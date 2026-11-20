"""Codex CLI subscription adapter for the shared policy edit protocol."""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .editors import EditorError, InvalidProposal, SYSTEM_PROMPT, _strict_json, parse_proposal, run_process


MODEL_ID = re.compile(r"gpt-[0-9]+(?:\.[0-9]+)?-[a-z0-9]+(?:-[a-z0-9]+)*")
CLI_VERSION = re.compile(r"codex-cli [0-9]+\.[0-9]+\.[0-9]+")
LAYER_ID = re.compile(r"fc(?:0|[1-9][0-9]*)")
LOGIN_CONFIG = 'forced_login_method="chatgpt"'
MODEL_EVIDENCE = "requested_cli_model_only"
USAGE_FIELDS = {
    "input_tokens", "cached_input_tokens", "cache_write_input_tokens",
    "output_tokens", "reasoning_output_tokens",
}
DISABLED_FEATURES = (
    "apps", "async_compaction", "browser_use", "browser_use_external", "computer_use", "fast_mode",
    "code_mode", "code_mode_host", "code_mode_only", "code_mode_prewarm",
    "goals", "hooks", "image_generation", "multi_agent", "multi_agent_v2",
    "plugins", "remote_plugin", "shell_tool", "shell_snapshot", "skill_search",
    "sleep_tool", "ultrafast_mode", "unbounded_connection_retries", "unified_exec", "view_image",
)


@dataclass(frozen=True)
class CodexOptions:
    model: str
    cli_version: str
    billing_assumption: str
    effort: str = "low"
    max_calls: int = 3
    retries: int = 0
    timeout_seconds: int = 120
    max_output_bytes: int = 1_000_000
    max_observed_input_tokens: int = 120_000
    max_observed_output_tokens: int = 4096

    def __post_init__(self):
        if not isinstance(self.model, str) or not MODEL_ID.fullmatch(self.model):
            raise ValueError("editor.model must be a full versioned Codex model ID")
        if not isinstance(self.cli_version, str) or not CLI_VERSION.fullmatch(self.cli_version):
            raise ValueError("editor.cli_version must be an exact codex-cli version string")
        if self.billing_assumption != "subscription_only":
            raise ValueError("Codex editor requires billing_assumption='subscription_only'")
        if self.effort not in ("low", "medium", "high", "xhigh"):
            raise ValueError("unsupported editor.effort")
        for name, low, high in (
            ("max_calls", 1, 100), ("retries", 0, 2), ("timeout_seconds", 1, 600),
            ("max_output_bytes", 8192, 8_000_000), ("max_observed_input_tokens", 256, 1_000_000),
            ("max_observed_output_tokens", 256, 32_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"editor.{name} must be an integer in [{low}, {high}]")


def _subscription_env():
    # Keep the native credential store and permit neither API keys nor endpoint overrides.
    return {
        key: os.environ[key] for key in ("HOME", "USER", "LOGNAME", "PATH", "LANG", "TMPDIR")
        if key in os.environ
    }


def _stream_evidence(stdout):
    """Collect only known stream types and typed usage for failed-run auditing."""
    events, kinds = [], []
    usage = None
    for line in stdout.splitlines():
        try:
            event = _strict_json(line)
        except (ValueError, RecursionError):
            event = None
        if not isinstance(event, dict):
            events.append({})
            kinds.append("malformed")
            continue
        events.append(event)
        kind = event.get("type")
        if kind in ("item.started", "item.updated", "item.completed"):
            item = event.get("item")
            item_kind = item.get("type") if isinstance(item, dict) else None
            if item_kind not in (
                "agent_message", "reasoning", "command_execution", "file_change",
                "mcp_tool_call", "web_search", "todo_list", "collab_tool_call", "error",
            ):
                item_kind = "unexpected"
            kinds.append(f"{kind}:{item_kind}")
        elif kind in ("thread.started", "turn.started", "turn.completed", "turn.failed", "error"):
            kinds.append(kind)
        else:
            kinds.append("unexpected")
        if kind == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = {
                key: value for key, value in event["usage"].items()
                if key in USAGE_FIELDS and type(value) is int
            }
    return events, kinds, usage


def _check_stream(events, usage):
    if len(events) != 4:
        raise EditorError("expected one Codex thread, turn, assistant message and completion")
    thread, start, message, completion = events
    if (
        set(thread) != {"type", "thread_id"} or thread["type"] != "thread.started"
        or not isinstance(thread["thread_id"], str) or not thread["thread_id"]
        or start != {"type": "turn.started"}
        or set(message) != {"type", "item"} or message["type"] != "item.completed"
        or set(completion) != {"type", "usage"} or completion["type"] != "turn.completed"
    ):
        raise EditorError("Codex reported unexpected events or model metadata")
    item = message["item"]
    if (
        not isinstance(item, dict) or set(item) != {"id", "type", "text"}
        or item["type"] != "agent_message" or not isinstance(item["id"], str) or not item["id"]
        or not isinstance(item["text"], str) or not item["text"].strip()
    ):
        raise EditorError("Codex reported tool activity or an invalid assistant message")
    required = USAGE_FIELDS - {"cache_write_input_tokens"}
    if (
        usage is None or usage != completion["usage"] or not required <= usage.keys()
        or any(count < 0 for count in usage.values())
        or not 0 <= usage["cached_input_tokens"] <= usage["input_tokens"]
        or usage.get("cache_write_input_tokens", 0) > usage["input_tokens"]
        or not 0 <= usage["reasoning_output_tokens"] <= usage["output_tokens"]
        or usage["input_tokens"] == 0 or usage["output_tokens"] == 0
    ):
        raise EditorError("Codex reported invalid or incomplete token usage")
    return item["text"]


class CodexEditor:
    """Launch one ephemeral ChatGPT-authenticated CLI session per editor action."""

    access = "codex_subscription_cli"

    def __init__(self, options, journal, *, allow_live=False, executable="codex"):
        if not allow_live:
            raise EditorError("Codex editor requires explicit live opt-in")
        self.options = options
        self.journal = journal
        self.executable = executable
        self.version = None

    def preflight(self):
        checks = (
            ("version", [self.executable, "--version"], self.options.cli_version),
            ("login", [self.executable, "login", "-c", LOGIN_CONFIG, "status"], "Logged in using ChatGPT"),
        )
        with TemporaryDirectory(prefix="weightspace-codex-") as directory:
            for check, command, expected in checks:
                try:
                    process = run_process(
                        command, prompt="", cwd=Path(directory), env=_subscription_env(),
                        timeout=min(15, self.options.timeout_seconds), output_cap=65536,
                    )
                except OSError as error:
                    self.journal.append(
                        "editor_preflight_failed", check=check, error_type=type(error).__name__,
                    )
                    raise EditorError(f"could not launch Codex CLI {check} check") from error
                streams = (process.stdout.strip(), process.stderr.strip())
                if (
                    process.returncode != 0 or process.interruption or process.invalid_utf8
                    or sorted(streams) != ["", expected]
                ):
                    self.journal.append(
                        "editor_preflight_failed", check=check, returncode=process.returncode,
                        interruption=process.interruption,
                    )
                    if check == "version":
                        raise EditorError("Codex CLI version differs from the configured version")
                    raise EditorError("Codex CLI must have a ChatGPT subscription login")
        self.version = self.options.cli_version
        self.journal.append(
            "editor_preflight", cli_version=self.version, access=self.access,
            requested_model=self.options.model, effort=self.options.effort,
            model_evidence=MODEL_EVIDENCE, actual_charge_usd=None,
            billing_assumption=self.options.billing_assumption,
        )

    def command(self, directory, instructions):
        overrides = (
            'approval_policy="never"', LOGIN_CONFIG, 'model_provider="openai"',
            "project_doc_max_bytes=0", "mcp_servers={}", 'web_search="disabled"',
            'shell_environment_policy.inherit="none"', "tools.update_plan.enabled=false",
            "hide_agent_reasoning=true", "include_environment_context=false",
            "include_permissions_instructions=false", "include_apps_instructions=false",
            "include_collaboration_mode_instructions=false", "skills.include_instructions=false",
            "check_for_update_on_startup=false", "features.skip_host_skill_discovery=true",
            "suppress_unstable_features_warning=true", "model_post_turn_compact_threshold_percent=0",
            f'model_reasoning_effort="{self.options.effort}"',
            f"model_instructions_file={json.dumps(str(instructions))}",
        )
        command = [
            self.executable, "--no-daemon", "exec", "--strict-config", "--ephemeral",
            "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
            "--color", "never", "--json", "--sandbox", "read-only",
            "--cd", str(directory), "--model", self.options.model,
        ]
        for override in overrides:
            command.extend(("-c", override))
        for feature in DISABLED_FEATURES:
            command.extend(("--disable", feature))
        return [*command, "-"]

    def request(self, prompt, *, call, attempt):
        if self.version is None:
            raise EditorError("Codex CLI preflight is required before editor requests")
        fields = {
            "call": call, "attempt": attempt, "access": self.access,
            "requested_model": self.options.model, "reported_models": [],
            "model_evidence": MODEL_EVIDENCE, "cli_version": self.version,
            "system_prompt": SYSTEM_PROMPT, "prompt": prompt,
            "prompt_sha256": sha256(prompt.encode()).hexdigest(), "actual_charge_usd": None,
        }
        with TemporaryDirectory(prefix="weightspace-codex-") as temporary:
            scratch = Path(temporary).resolve()
            directory = scratch / "cwd"
            instructions = scratch / "instructions.md"
            command = self.command(directory, instructions)
            fields["command"] = command
            try:
                directory.mkdir()
                instructions.write_text(SYSTEM_PROMPT, encoding="utf-8")
                process = run_process(
                    command, prompt=prompt, cwd=directory, env=_subscription_env(),
                    timeout=self.options.timeout_seconds, output_cap=self.options.max_output_bytes,
                )
            except OSError as error:
                self.journal.append("editor_process_error", **fields, error_type=type(error).__name__)
                raise EditorError("could not launch Codex CLI") from error
        events, kinds, usage = _stream_evidence(process.stdout)
        response, proposal, stream_error, proposal_error = None, None, None, None
        benign_stderr = not process.invalid_utf8 and process.stderr.strip() in ("", "Reading prompt from stdin...")
        try:
            if process.interruption:
                raise EditorError(f"Codex CLI stopped at its {process.interruption} limit")
            if process.returncode != 0:
                raise EditorError(f"Codex CLI exited with status {process.returncode}")
            if process.invalid_utf8:
                raise EditorError("Codex CLI emitted invalid UTF-8")
            if not benign_stderr:
                raise EditorError("Codex CLI reported unexpected diagnostics")
            response = _check_stream(events, usage)
            if (
                usage["input_tokens"] > self.options.max_observed_input_tokens
                or usage["output_tokens"] > self.options.max_observed_output_tokens
            ):
                raise EditorError("Codex reported token usage above the configured acceptance limit")
        except EditorError as exc:
            stream_error = exc
        if stream_error is None:
            try:
                proposal = parse_proposal(response)
                if any(not LAYER_ID.fullmatch(op["layer"]) for op in proposal.get("ops", [])):
                    raise InvalidProposal("expected a layer ID in fcN format")
                response.encode("utf-8")
            except InvalidProposal as exc:
                proposal_error = exc
            except UnicodeEncodeError:
                proposal_error = InvalidProposal("expected a UTF-8 JSON proposal")
        accepted = stream_error is None and proposal_error is None
        self.journal.append(
            "editor_turn", **fields, stream_kinds=kinds, usage=usage,
            raw_stdout=process.stdout if accepted else None,
            raw_stdout_sha256=process.stdout_sha256 or sha256(process.stdout.encode()).hexdigest(),
            raw_stderr=process.stderr if benign_stderr else None,
            raw_stderr_sha256=process.stderr_sha256 or sha256(process.stderr.encode()).hexdigest(),
            response=response if accepted else None, returncode=process.returncode,
            interruption=process.interruption, stream_error=str(stream_error) if stream_error else None,
            proposal_error=str(proposal_error) if proposal_error else None,
        )
        if stream_error:
            raise stream_error
        if proposal_error:
            raise proposal_error
        return proposal
