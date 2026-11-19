"""Tool-free, stateless Claude Code subscription proposals on broker dev feedback."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic


SYSTEM_PROMPT = """You edit a neural policy by proposing JSON to a trusted evaluator.
Use only the provided task specification, checkpoint, and measured development results.
Every probe is a complete ordered edit list applied to the starting checkpoint.
Return one JSON object with no Markdown: {"action":"probe","ops":[...]} or
{"action":"select","candidate_id":0}. Select only a scored candidate in the history.
The development objective averages failure recovery and normal behavior retention.
A failed or rejected candidate cannot be selected. No executables or prose are accepted."""

FULL_MODEL = re.compile(r"claude-(?:haiku|sonnet|opus|fable)-[0-9]+-[0-9]+(?:-[0-9]{8})?")
ALLOWED_ENV = {"HOME", "USER", "LOGNAME", "PATH", "LANG", "TMPDIR", "TMP", "TEMP", "SHELL"}


@dataclass(frozen=True)
class ClaudeOptions:
    model: str
    billing_assumption: str
    effort: str = "low"
    max_calls: int = 3
    retries: int = 0
    timeout_seconds: int = 120
    max_output_bytes: int = 1_000_000
    max_output_tokens: int = 4096
    max_api_equivalent_usd_per_call: float = 0.5

    def __post_init__(self):
        if not isinstance(self.model, str) or not FULL_MODEL.fullmatch(self.model):
            raise ValueError("editor.model must be a full versioned Claude model ID")
        if self.billing_assumption not in ("extra_usage_disabled", "extra_usage_authorized"):
            raise ValueError("editor.billing_assumption must describe the account's extra-usage authorization")
        if self.effort not in ("low", "medium", "high", "max"):
            raise ValueError("unsupported editor.effort")
        for name, low, high in (
            ("max_calls", 1, 100), ("retries", 0, 2), ("timeout_seconds", 1, 600),
            ("max_output_bytes", 8192, 8_000_000), ("max_output_tokens", 256, 32_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"editor.{name} must be an integer in [{low}, {high}]")
        limit = self.max_api_equivalent_usd_per_call
        if type(limit) not in (float, int) or not 0 < limit <= 10:
            raise ValueError("editor.max_api_equivalent_usd_per_call must be in (0, 10]")


class EditorError(RuntimeError):
    """The pinned, isolated editor protocol did not complete."""


class InvalidProposal(EditorError):
    """A well-isolated request returned a malformed proposal."""


@dataclass(frozen=True)
class ProcessResult:
    stdout: str
    stderr: str
    returncode: int | None
    interruption: str | None


def subscription_environment(output_tokens):
    """Inherit shell essentials and leave login entirely to Claude Code."""
    result = {
        key: value for key, value in os.environ.items()
        if key in ALLOWED_ENV or key.startswith("LC_")
    }
    result.update(
        CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(output_tokens),
        DISABLE_AUTOUPDATER="1",
        DISABLE_EXTRA_USAGE_COMMAND="1",
    )
    return result


def run_process(command, *, prompt, cwd, env, timeout, output_cap):
    """Capture a bounded CLI stream while feeding the prompt on standard input."""
    stdout, stderr = bytearray(), bytearray()
    remaining = memoryview(prompt.encode("utf-8"))
    deadline = monotonic() + timeout
    interruption = None
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=cwd, env=env, start_new_session=True, shell=False,
    ) as process, selectors.DefaultSelector() as selector:
        try:
            for stream, target in ((process.stdout, stdout), (process.stderr, stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, target)
            if remaining:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE)
            else:
                process.stdin.close()
            while selector.get_map():
                seconds = deadline - monotonic()
                if seconds <= 0:
                    interruption = "timeout"
                    break
                for key, direction in selector.select(seconds):
                    if direction == selectors.EVENT_WRITE:
                        try:
                            written = os.write(key.fd, remaining[:65536])
                            remaining = remaining[written:]
                        except BrokenPipeError:
                            remaining = remaining[:0]
                        except BlockingIOError:
                            continue
                        if not remaining:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    room = output_cap - len(stdout) - len(stderr)
                    key.data.extend(chunk[:room])
                    if len(chunk) > room:
                        interruption = "output_cap"
                        break
                if interruption:
                    break
            if interruption is None:
                try:
                    process.wait(timeout=max(0, deadline - monotonic()))
                except subprocess.TimeoutExpired:
                    interruption = "timeout"
        finally:
            if interruption or process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        return ProcessResult(
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            process.returncode, interruption,
        )


def _strict_json(text):
    def object_pairs(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate object field")
        return result

    def invalid_constant(_value):
        raise ValueError("invalid JSON number")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON number")
        return number

    return json.loads(
        text, object_pairs_hook=object_pairs, parse_constant=invalid_constant, parse_float=finite_float,
    )


def _operation_is_json(op):
    fields = {
        "scale_neuron": {"op", "layer", "neuron", "factor"},
        "zero_neuron": {"op", "layer", "neuron"},
        "shift_bias": {"op", "layer", "neuron", "shift"},
        "rank_one": {"op", "layer", "alpha", "u", "v"},
    }
    if not isinstance(op, dict) or not isinstance(op.get("op"), str):
        return False
    if set(op) != fields.get(op["op"]) or not isinstance(op["layer"], str):
        return False
    if "neuron" in op and type(op["neuron"]) is not int:
        return False

    def numeric(value):
        return type(value) is int or (type(value) is float and math.isfinite(value))

    if any(key in op and not numeric(op[key]) for key in ("factor", "shift", "alpha")):
        return False
    return all(
        key not in op or isinstance(op[key], list) and all(map(numeric, op[key]))
        for key in ("u", "v")
    )


def parse_proposal(response):
    """Check the wire shape before the shared edit validator checks semantics."""
    try:
        proposal = _strict_json(response)
    except (ValueError, RecursionError) as error:
        raise InvalidProposal("expected one strict JSON proposal") from error
    if isinstance(proposal, dict):
        if (
            proposal.get("action") == "select" and set(proposal) == {"action", "candidate_id"}
            and type(proposal["candidate_id"]) is int and proposal["candidate_id"] >= 0
        ):
            return proposal
        if (
            proposal.get("action") == "probe" and set(proposal) == {"action", "ops"}
            and isinstance(proposal["ops"], list) and all(map(_operation_is_json, proposal["ops"]))
        ):
            return proposal
    raise InvalidProposal("expected probe operations or a nonnegative scored candidate ID")


def _stream_fields(stdout):
    events = []
    malformed = False
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = _strict_json(line)
            if not isinstance(event, dict):
                raise ValueError("expected JSON object")
            events.append(event)
        except (ValueError, RecursionError):
            malformed = True
    inits = [event for event in events if event.get("type") == "system" and event.get("subtype") == "init"]
    results = [event for event in events if event.get("type") == "result"]
    assistant = [event for event in events if event.get("type") == "assistant"]
    messages = [event.get("message") for event in assistant]
    models = sorted({
        message["model"] for message in messages
        if isinstance(message, dict) and isinstance(message.get("model"), str)
    })
    return events, malformed, inits, results, assistant, models


def _check_stream(events, malformed, inits, results, assistant, models, options):
    if malformed or len(inits) != 1 or len(results) != 1:
        raise EditorError("expected one well-formed Claude initialization and result")
    init, result = inits[0], results[0]
    if events[0] is not init or events[-1] is not result or not assistant:
        raise EditorError("Claude stream is missing its ordered initialization, answer, or result")
    session = init.get("session_id")
    if not isinstance(session, str) or not session or any(event.get("session_id") != session for event in events):
        raise EditorError("Claude stream spans multiple or missing sessions")
    if init.get("tools") != [] or init.get("mcp_servers") != []:
        raise EditorError("Claude reported a nonempty or missing tool or MCP inventory")
    if any(init.get(field) != [] for field in ("plugins", "skills")):
        raise EditorError("Claude reported a nonempty or missing plugin or skill inventory")
    if init.get("apiKeySource") != "none":
        raise EditorError("Claude reported an unexpected API key source")
    for event in events[1:-1]:
        if event.get("type") == "assistant":
            continue
        if event.get("type") == "rate_limit_event":
            rate = event.get("rate_limit_info")
            if not isinstance(rate, dict) or rate.get("status") not in ("allowed", "allowed_warning"):
                raise EditorError("Claude reported a rejected or malformed rate limit event")
            if options.billing_assumption == "extra_usage_disabled" and (
                rate.get("overageEnabled") is True or rate.get("isUsingOverage") is True
            ):
                raise EditorError("Claude rate-limit telemetry contradicts the billing assumption")
            continue
        if (
            event.get("type") == "system" and event.get("subtype") == "thinking_tokens"
            and all(type(event.get(key)) is int for key in ("estimated_tokens", "estimated_tokens_delta"))
        ):
            continue
        raise EditorError("Claude reported an unexpected stream event")
    message_ids = set()
    text = []
    for event in assistant:
        if any(event.get(key) for key in ("parent_tool_use_id", "agent_id", "error", "aborted")):
            raise EditorError("Claude did not return a successful main-thread response")
        message = event.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("id"), str) or not message["id"]:
            raise EditorError("Claude returned a malformed assistant message")
        if message.get("model") != options.model:
            raise EditorError("Claude returned an assistant message for a different or missing model")
        message_ids.add(message["id"])
        content = message.get("content")
        if not isinstance(content, list) or any(
            not isinstance(block, dict) or block.get("type") not in ("text", "thinking", "redacted_thinking")
            for block in content
        ):
            raise EditorError("Claude returned an unexpected assistant content block")
        for block in content:
            if block["type"] == "text":
                if not isinstance(block.get("text"), str):
                    raise EditorError("Claude returned a malformed text block")
                text.append(block["text"])
    if len(message_ids) != 1 or type(result.get("num_turns")) is not int or result["num_turns"] != 1:
        raise EditorError("Claude did not confirm a single model turn")
    if (
        result.get("permission_denials") != [] or result.get("deferred_tool_use") is not None
        or result.get("queued_turn_count", 0) != 0
        or result.get("stop_reason") in ("tool_use", "pause_turn")
    ):
        raise EditorError("Claude reported tool activity, permission denials, or a queued turn")
    if init.get("model") != options.model or models != [options.model]:
        raise EditorError("Claude did not confirm the requested model identity")
    if result.get("subtype") != "success" or result.get("is_error") is not False:
        raise EditorError("Claude returned an unsuccessful result")
    if "local_command" in result or result.get("terminal_reason", "completed") != "completed":
        raise EditorError("Claude did not complete the model turn normally")
    model_usage = result.get("modelUsage", {})
    if not isinstance(model_usage, dict) or any(
        model != options.model or not isinstance(usage, dict)
        or usage.get("provider", "firstParty") != "firstParty"
        or usage.get("webSearchRequests", 0)
        for model, usage in model_usage.items()
    ):
        raise EditorError("Claude reported usage for a different model or provider")
    if not text or not isinstance(result.get("result"), str) or result["result"].strip() != "".join(text).strip():
        raise EditorError("Claude result text does not match the assistant transcript")


class ClaudeEditor:
    """Own the CLI preflight and append the full scientific turn evidence."""

    def __init__(self, options, journal, *, allow_live=False, executable="claude"):
        if not allow_live:
            raise EditorError("Claude editor requires explicit live opt-in")
        self.options = options
        self.journal = journal
        self.executable = executable
        self.version = None

    def _invoke(self, command, prompt="", *, preflight=False):
        with TemporaryDirectory(prefix="weightspace-editor-") as directory:
            return run_process(
                command, prompt=prompt, cwd=Path(directory),
                env=subscription_environment(self.options.max_output_tokens),
                timeout=min(15, self.options.timeout_seconds) if preflight else self.options.timeout_seconds,
                output_cap=65536 if preflight else self.options.max_output_bytes,
            )

    def preflight(self):
        version = self._invoke([self.executable, "--version"], preflight=True)
        if version.returncode != 0 or version.interruption or not re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+ \(Claude Code\)", version.stdout.strip()
        ):
            raise EditorError("could not identify the Claude Code CLI version")
        self.version = version.stdout.strip()
        auth = self._invoke([self.executable, "auth", "status", "--json"], preflight=True)
        try:
            account = _strict_json(auth.stdout)
        except ValueError as error:
            raise EditorError("could not read the Claude Code login status") from error
        if (
            auth.returncode != 0 or auth.interruption or not isinstance(account, dict)
            or account.get("loggedIn") is not True or account.get("authMethod") != "claude.ai"
            or account.get("apiProvider") != "firstParty" or not account.get("subscriptionType")
        ):
            raise EditorError("Claude Code must have a first-party Claude subscription login")
        self.journal.append(
            "editor_preflight", cli_version=self.version, access="claude_subscription_cli",
            requested_model=self.options.model, effort=self.options.effort,
            subscription_type=account.get("subscriptionType"), actual_charge_usd=None,
            billing_assumption=self.options.billing_assumption,
        )

    def command(self):
        return [
            self.executable, "--print", "--model", self.options.model,
            "--effort", self.options.effort, "--safe-mode", "--restricted",
            "--tools", "", "--disallowed-tools", "*", "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}', "--setting-sources", "",
            "--settings", '{"disableAllHooks":true,"disableClaudeAiConnectors":true}',
            "--disable-slash-commands", "--no-session-persistence", "--no-chrome",
            "--permission-mode", "dontAsk", "--permission-prompts", "none",
            "--max-turns", "1",
            "--max-budget-usd", str(self.options.max_api_equivalent_usd_per_call),
            "--output-format", "stream-json", "--verbose", "--include-hook-events",
            "--prompt-suggestions", "off",
            "--system-prompt", SYSTEM_PROMPT,
        ]

    def request(self, prompt, *, call, attempt):
        command = self.command()
        try:
            process = self._invoke(command, prompt)
        except OSError as error:
            self.journal.append(
                "editor_process_error", call=call, attempt=attempt, command=command,
                prompt=prompt, prompt_sha256=sha256(prompt.encode()).hexdigest(),
                error_type=type(error).__name__, actual_charge_usd=None,
            )
            raise EditorError("could not launch Claude Code") from error
        events, malformed, inits, results, assistant, models = _stream_fields(process.stdout)
        result = results[-1] if results else {}
        self.journal.append(
            "editor_turn", call=call, attempt=attempt, access="claude_subscription_cli",
            requested_model=self.options.model, reported_models=models,
            cli_version=self.version, command=command, system_prompt=SYSTEM_PROMPT,
            prompt=prompt, prompt_sha256=sha256(prompt.encode()).hexdigest(),
            raw_stdout=process.stdout, raw_stderr=process.stderr,
            returncode=process.returncode, interruption=process.interruption,
            response=result.get("result"), usage=result.get("usage"),
            model_usage=result.get("modelUsage"), actual_charge_usd=None,
            cli_api_equivalent_estimate_usd=result.get("total_cost_usd"),
        )
        if process.interruption:
            raise EditorError(f"Claude Code stopped at its {process.interruption} limit")
        if process.returncode != 0:
            raise EditorError(f"Claude Code exited with status {process.returncode}")
        _check_stream(events, malformed, inits, results, assistant, models, self.options)
        return parse_proposal(result["result"])


def feedback(candidate, ops):
    score = candidate.score
    return {
        "candidate_id": candidate.candidate_id, "status": candidate.status,
        "ops": ops, "reason": candidate.reason, "changed_entries": candidate.changed_entries,
        "dev_score": None if score is None else {
            "objective": score.objective, "failure_success": score.failure_success,
            "normal_success": score.normal_success,
        },
    }


def build_prompt(dev, policy, history, calls_remaining, issue):
    task = dev.task
    payload = {
        "task": {
            "id": task.id, "name": task.name, "sizes": task.sizes, "horizon": task.horizon,
            "description": task.description, "normalization": task.normalization,
            "action_rule": task.action_rule, "success_rule": task.success_rule,
        },
        "checkpoint": {key: value.tolist() for key, value in policy.items()},
        "edit_limits": asdict(dev.limits),
        "operations": [
            {"op": "scale_neuron", "layer": "fcN", "neuron": "hidden row index", "factor": "positive"},
            {"op": "zero_neuron", "layer": "fcN", "neuron": "hidden index; zero outgoing column"},
            {"op": "shift_bias", "layer": "fcN", "neuron": "bias index", "shift": "additive"},
            {"op": "rank_one", "layer": "fcN", "alpha": "multiplier", "u": "row axis vector", "v": "column axis vector"},
        ],
        "layout": "Each layer is affine W @ x + b. ReLU follows every hidden layer. "
        "Scale multiplies that neuron's incoming row and bias without compensation.",
        "candidate_slots_remaining": dev.budget.candidates_remaining,
        "editor_calls_remaining_including_this": calls_remaining,
        "previous_protocol_error": issue,
        "history": history,
    }
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))


def run_editor(dev, initial, editor):
    """Allow only broker-mediated probes and selection from scored candidates."""
    editor.preflight()
    policy = dev.policy()
    history = [feedback(initial, [])]
    scored = {initial.candidate_id}
    issue = None
    invalid = 0
    for call in range(1, editor.options.max_calls + 1):
        remaining = editor.options.max_calls - call + 1
        prompt = build_prompt(dev, policy, history, remaining, issue)
        try:
            proposal = editor.request(prompt, call=call, attempt=invalid + 1)
            if proposal["action"] == "select":
                if proposal["candidate_id"] in scored:
                    return proposal["candidate_id"]
                raise InvalidProposal("candidate ID has no successful development score")
            if not dev.budget.candidates_remaining or remaining <= 1:
                raise InvalidProposal("the remaining action must select a scored candidate")
        except InvalidProposal as error:
            issue = str(error)
            invalid += 1
            editor.journal.append("editor_protocol_error", call=call, reason=issue)
            if invalid > editor.options.retries:
                raise
            continue
        invalid = 0
        issue = None
        candidate = dev.probe(proposal["ops"])
        history.append(feedback(candidate, proposal["ops"]))
        if candidate.status == "scored":
            scored.add(candidate.candidate_id)
    raise EditorError("Claude did not select a scored candidate within its call cap")
