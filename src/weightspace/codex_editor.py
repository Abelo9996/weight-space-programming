"""Codex CLI subscription adapter for the shared policy edit protocol."""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .artifacts import canonical_json
from .editors import EditorError, InvalidProposal, SYSTEM_PROMPT, _strict_json, parse_proposal, run_process


MODEL_ID = re.compile(r"gpt-[0-9]+(?:\.[0-9]+)?-[a-z0-9]+(?:-[a-z0-9]+)*")
CLI_VERSION = re.compile(r"codex-cli [0-9]+\.[0-9]+\.[0-9]+")
LAYER_ID = re.compile(r"fc(?:0|[1-9][0-9]*)")
THREAD_ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
ITEM_ID = re.compile(r"item_[0-9]+")
LOGIN_CONFIG = 'forced_login_method="chatgpt"'
MODEL_EVIDENCE = "requested_cli_model_only"
CATALOG_SOURCE_FILE = "codex-model-source.json"
CATALOG_DIRECT_FILE = "codex-model-catalog.json"
MAX_CATALOG_BYTES = 1_000_000
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
CREDENTIAL_FIELD = re.compile(
    r"""(?ix)\b(?:(?:proxy-)?authorization|(?:set-)?cookie|password|passphrase
        |secret|client[_ -]?secret|(?:[a-z0-9]+_)?api[_ -]?key
        |(?:(?:openai|codex)_)?(?:access|refresh|id|oauth|session)[_ -]?token
        |oauth[_ -]?code|code[_ -]?verifier)["']?\s*[:=]
        |\b(?:bearer|basic)\s+[A-Za-z0-9._~+/-]+"""
)
DIAGNOSTIC_REDACTIONS = (
    (re.compile(r"(?i)(https?://)[^/\s]+@"), r"\1[redacted]@"),
    (re.compile(r"(?i)(https?://[^?\s#]+)[?#][^\s]*"), r"\1[query redacted]"),
    (re.compile(r"\b(?:sk|rk|sess)[-_][A-Za-z0-9_-]+", re.IGNORECASE), "[credential redacted]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+){1,2}"), "[credential redacted]"),
    (re.compile(r"\b[0-9a-fA-F]{24,}\b"), "[opaque value redacted]"),
    (re.compile(r"""(?ix)(?:/(?:users|home)/|/var/root\b|/root\b|[a-z]:\\users\\)
        [^\s"'\])}>]*"""), "[home path redacted]"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b"), "[email redacted]"),
)
OPAQUE_VALUE = re.compile(r"(?<![\w+/-])[A-Za-z0-9_+/-]{24,}={0,2}(?![\w+/-])")


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
    model_catalog_source: str | None = None
    model_catalog_source_sha256: str | None = None

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
        if self.model_catalog_source is not None or self.model_catalog_source_sha256 is not None:
            if not isinstance(self.model_catalog_source, str) or not Path(self.model_catalog_source).is_absolute():
                raise ValueError("editor.model_catalog_source must be an absolute descriptor path")
            digest = self.model_catalog_source_sha256
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("editor.model_catalog_source_sha256 must pin the source descriptor")


def _catalog_bytes(path):
    try:
        with Path(path).open("rb") as file:
            content = file.read(MAX_CATALOG_BYTES + 1)
    except OSError as error:
        raise ValueError("could not read the Codex model descriptor or archived catalog") from error
    if len(content) > MAX_CATALOG_BYTES:
        raise ValueError("Codex model descriptor or catalog exceeds its byte limit")
    return content


def _direct_catalog(source, options):
    if sha256(source).hexdigest() != options.model_catalog_source_sha256:
        raise ValueError("Codex model descriptor SHA-256 differs from the configured pin")
    try:
        model = _strict_json(source.decode("utf-8"))
    except (ValueError, RecursionError) as error:
        raise ValueError("expected one strict UTF-8 Codex model descriptor") from error
    if not isinstance(model, dict) or model.get("slug") != options.model or "models" in model or "identity" in model:
        raise ValueError("Codex model descriptor must contain only the configured model entry")
    try:
        return canonical_json({"models": [{**model, "tool_mode": "direct"}]}) + b"\n"
    except (ValueError, RecursionError) as error:
        raise ValueError("could not encode the derived Codex model catalog") from error


def stage_model_catalog(options, output):
    """Freeze an opt-in source descriptor and its Direct-only derivative for the whole comparison."""
    if options.model_catalog_source is None:
        return None
    source = _catalog_bytes(options.model_catalog_source)
    derived = _direct_catalog(source, options)
    for name, content in ((CATALOG_SOURCE_FILE, source), (CATALOG_DIRECT_FILE, derived)):
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
    return catalog_provenance(options, output)


def catalog_provenance(options, output):
    """Verify the archived catalog independently of the configured source path."""
    if options.model_catalog_source is None:
        return None
    if output is None:
        raise ValueError("Codex Direct catalog requires a staged comparison archive")
    source = _catalog_bytes(output / CATALOG_SOURCE_FILE)
    derived = _direct_catalog(source, options)
    if _catalog_bytes(output / CATALOG_DIRECT_FILE) != derived:
        raise ValueError("archived Codex catalog differs from the pinned Direct-mode derivative")
    return {
        "model": options.model, "tool_mode": "direct",
        "source_file": CATALOG_SOURCE_FILE, "source_sha256": options.model_catalog_source_sha256,
        "catalog_file": CATALOG_DIRECT_FILE, "catalog_sha256": sha256(derived).hexdigest(),
    }


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


def _diagnostic_preview(message):
    """Bound the preview and scrub recognizable credentials and local identifiers."""
    preview = "".join(character if character.isprintable() else " " for character in message)
    credential = CREDENTIAL_FIELD.search(preview)
    if credential:
        preview = preview[:credential.start()] + "[credential details redacted]"
    home = os.environ.get("HOME")
    if home and home != "/":
        preview = preview.replace(home, "[home]")
    for pattern, replacement in DIAGNOSTIC_REDACTIONS:
        preview = pattern.sub(replacement, preview)

    def redact_opaque(match):
        value = match.group()
        if any(character.isdigit() for character in value) or (
            any(character.isupper() for character in value) and any(character.islower() for character in value)
        ):
            return "[opaque value redacted]"
        return value

    preview = OPAQUE_VALUE.sub(redact_opaque, preview)
    return {"preview": preview[:512], "redacted": preview != message, "truncated": len(preview) > 512}


def _diagnostic_evidence(events, stderr):
    details = []
    for event in events:
        kind = event.get("type")
        payload = event
        if kind == "item.completed":
            payload = event.get("item")
            if not isinstance(payload, dict) or payload.get("type") != "error":
                continue
            kind = "item.completed:error"
        elif kind == "turn.failed":
            payload = event.get("error")
        elif kind != "error":
            continue
        message = payload.get("message") if isinstance(payload, dict) else None
        if isinstance(message, str) and message.strip():
            details.append({"source": kind, **_diagnostic_preview(message)})
            if len(details) == 8:
                break
    stderr_lines = [line for line in stderr.splitlines() if line.strip()]
    details.extend(
        {"source": "stderr", **_diagnostic_preview(line)}
        for line in stderr_lines[:8]
    )
    return details


def _check_stream(events, usage):
    if len(events) != 4:
        raise EditorError("expected one Codex thread, turn, assistant message and completion")
    thread, start, message, completion = events
    if (
        set(thread) != {"type", "thread_id"} or thread["type"] != "thread.started"
        or not isinstance(thread["thread_id"], str) or not THREAD_ID.fullmatch(thread["thread_id"])
        or start != {"type": "turn.started"}
        or set(message) != {"type", "item"} or message["type"] != "item.completed"
        or set(completion) != {"type", "usage"} or completion["type"] != "turn.completed"
    ):
        raise EditorError("Codex reported unexpected events or model metadata")
    item = message["item"]
    if (
        not isinstance(item, dict) or set(item) != {"id", "type", "text"}
        or item["type"] != "agent_message" or not isinstance(item["id"], str) or not ITEM_ID.fullmatch(item["id"])
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

    def __init__(self, options, journal, *, allow_live=False, executable="codex", catalog_archive=None):
        if not allow_live:
            raise EditorError("Codex editor requires explicit live opt-in")
        self.options = options
        self.journal = journal
        self.executable = executable
        self.version = None
        self.catalog_archive = catalog_archive
        self.catalog = None

    def _verify_catalog(self):
        try:
            return catalog_provenance(self.options, self.catalog_archive)
        except ValueError as error:
            raise EditorError(str(error)) from error

    def preflight(self):
        try:
            self.catalog = self._verify_catalog()
        except EditorError:
            self.journal.append("editor_preflight_failed", check="model_catalog")
            raise
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
            billing_assumption=self.options.billing_assumption, model_catalog=self.catalog,
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
        if self.catalog is not None:
            catalog = (self.catalog_archive / CATALOG_DIRECT_FILE).resolve()
            command.extend(("-c", f"model_catalog_json={json.dumps(str(catalog))}"))
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
            "model_catalog": self.catalog,
        }
        try:
            if self._verify_catalog() != self.catalog:
                raise EditorError("Codex model catalog provenance changed after preflight")
        except EditorError:
            self.journal.append("editor_process_error", **fields, error_type="model_catalog")
            raise
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
        thread_id = events[0].get("thread_id") if events and events[0].get("type") == "thread.started" else None
        if not isinstance(thread_id, str) or not THREAD_ID.fullmatch(thread_id):
            thread_id = None
        usage_limits = {
            "input_tokens": self.options.max_observed_input_tokens,
            "output_tokens": self.options.max_observed_output_tokens,
        }
        usage_excesses = {
            name: {"observed": usage[name], "limit": limit}
            for name, limit in usage_limits.items()
            if usage is not None and usage.get(name, 0) > limit
        }
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
            if usage_excesses:
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
            "editor_turn", **fields, stream_kinds=kinds, thread_id=thread_id, usage=usage,
            usage_excesses=usage_excesses,
            diagnostics=_diagnostic_evidence(events, process.stderr if not benign_stderr else ""),
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
