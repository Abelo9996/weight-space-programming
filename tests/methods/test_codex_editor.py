from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

from weightspace import codex_editor
from weightspace.codex_editor import CodexEditor, CodexOptions
from weightspace.editors import EditorError, InvalidProposal, ProcessResult, SYSTEM_PROMPT


VERSION = "codex-cli 0.169.0"
MODEL = "gpt-6.2-sol"
SETTINGS = {"model": MODEL, "cli_version": VERSION, "billing_assumption": "subscription_only"}
USAGE = {
    "input_tokens": 15_317, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
    "output_tokens": 15, "reasoning_output_tokens": 0,
}
SELECT = '{"action":"select","candidate_id":0}'
THREAD = "019ac000-0000-7000-8000-000000000001"


class Journal:
    def __init__(self):
        self.events = []

    def append(self, event, **fields):
        # Match the real journal's UTF-8 and finite JSON requirements.
        json.dumps(fields, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.events.append({"event": event, **fields})


def stream(response=SELECT, *, extras=(), usage=USAGE, thread_fields=None):
    events = [
        {"type": "thread.started", "thread_id": THREAD, **(thread_fields or {})},
        {"type": "turn.started"},
        *extras,
        {"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": response}},
        {"type": "turn.completed", "usage": usage},
    ]
    return "\n".join(map(json.dumps, events)) + "\n"


def mock_cli(
    monkeypatch, responses, *,
    version=ProcessResult(VERSION, "", 0, None),
    auth=ProcessResult("", "Logged in using ChatGPT", 0, None),
):
    launches = []

    def invoke(command, *, prompt, cwd, env, timeout, output_cap):
        assert cwd.is_dir() and not list(cwd.iterdir())
        assert set(env) <= {"HOME", "USER", "LOGNAME", "PATH", "LANG", "TMPDIR"}
        if command == ["codex", "--version"]:
            return version
        if command == ["codex", "login", "-c", 'forced_login_method="chatgpt"', "status"]:
            return auth
        assert command[:3] == ["codex", "--no-daemon", "exec"]
        assert command[-1] == "-" and prompt not in command
        assert command[command.index("--model") + 1] == MODEL
        assert command[command.index("--cd") + 1] == str(cwd)
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert {
            "--ephemeral", "--ignore-user-config", "--ignore-rules", "--strict-config",
            "--skip-git-repo-check", "--json", 'model_provider="openai"',
            'forced_login_method="chatgpt"', 'web_search="disabled"', 'approval_policy="never"',
            'shell_environment_policy.inherit="none"', "project_doc_max_bytes=0",
            "mcp_servers={}", "include_environment_context=false",
            "suppress_unstable_features_warning=true",
        } <= set(command)
        for feature in ("shell_tool", "code_mode", "apps", "hooks", "plugins", "unified_exec"):
            assert command[command.index(feature) - 1] == "--disable"
        setting = next(item for item in command if item.startswith("model_instructions_file="))
        instructions = Path(json.loads(setting.split("=", 1)[1]))
        assert instructions.parent == cwd.parent
        assert instructions.read_text() == SYSTEM_PROMPT
        assert timeout == 120 and output_cap == 1_000_000
        launches.append(prompt)
        result = next(responses)
        return result if isinstance(result, ProcessResult) else ProcessResult(result, "", 0, None)

    monkeypatch.setattr(codex_editor, "run_process", invoke)
    return launches


def test_codex_native_command_and_audit_evidence(monkeypatch):
    for secret_setting in (
        "OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_HOME",
        "OPENAI_IDENTITY_TOKEN_FILE", "RUST_LOG",
    ):
        monkeypatch.setenv(secret_setting, "private-override")
    raw = stream()
    launches = mock_cli(monkeypatch, iter([raw]))
    journal = Journal()
    editor = CodexEditor(CodexOptions(**SETTINGS), journal, allow_live=True)
    editor.preflight()
    assert editor.request('{"checkpoint":{}}', call=1, attempt=1) == json.loads(SELECT)
    assert launches == ['{"checkpoint":{}}']
    preflight, turn = journal.events
    assert preflight["cli_version"] == VERSION
    assert preflight["access"] == "codex_subscription_cli"
    assert preflight["model_evidence"] == turn["model_evidence"] == "requested_cli_model_only"
    assert turn["requested_model"] == MODEL and turn["reported_models"] == []
    assert turn["usage"] == USAGE and turn["actual_charge_usd"] is None
    assert turn["thread_id"] == THREAD and turn["diagnostics"] == [] and turn["usage_excesses"] == {}
    assert turn["raw_stdout"] == raw
    assert turn["raw_stdout_sha256"] == sha256(raw.encode()).hexdigest()
    assert "private-override" not in str(journal.events)


@pytest.mark.parametrize(("changed_check", "result"), [
    ("version", ProcessResult("codex-cli 0.170.0", "", 0, None)),
    ("login", ProcessResult("", "Logged in using an API key private-token", 0, None)),
    ("login", ProcessResult("", "", 20, "timeout")),
])
def test_codex_rejects_unpinned_version_or_non_subscription(monkeypatch, changed_check, result):
    kwargs = {"auth" if changed_check == "login" else changed_check: result}
    launches = mock_cli(monkeypatch, iter([]), **kwargs)
    journal = Journal()
    editor = CodexEditor(CodexOptions(**SETTINGS), journal, allow_live=True)
    with pytest.raises(EditorError):
        editor.preflight()
    assert not launches
    assert journal.events[-1]["check"] == changed_check
    assert "private-token" not in str(journal.events)


@pytest.mark.parametrize("raw", [
    "not jsonl",
    stream(extras=[{"type": "item.started", "item": {"type": "command_execution", "command": "private-token"}}]),
    stream(extras=[{"type": "item.completed", "item": {"type": "file_change"}}]),
    stream(extras=[{"type": "item.completed", "item": {"type": "error"}}]),
    stream(thread_fields={"model": "different-model"}),
    stream(thread_fields={"thread_id": "private-token"}),
    stream(usage={**USAGE, "output_tokens": True}),
    stream(usage={**USAGE, "reasoning_output_tokens": 16}),
    stream(usage={**USAGE, "input_tokens": 120_001}),
    stream(usage={**USAGE, "output_tokens": 4097}),
    stream(usage={**USAGE, "different_model_tokens": 9}),
    ProcessResult(stream(), "Model changed sk-private-token", 0, None),
    ProcessResult(stream(), "", 0, None, invalid_utf8=True),
    ProcessResult(stream(), "", -9, "output_cap"),
    ProcessResult(stream(), "", 1, None),
])
def test_codex_rejects_tool_activity_usage_drift_and_unexpected_cli_output(monkeypatch, raw):
    mock_cli(monkeypatch, iter([raw]))
    journal = Journal()
    editor = CodexEditor(CodexOptions(**SETTINGS), journal, allow_live=True)
    editor.preflight()
    with pytest.raises(EditorError):
        editor.request("{}", call=1, attempt=1)
    turn = journal.events[-1]
    assert turn["stream_error"] is not None
    assert turn["raw_stdout"] is None and turn["response"] is None
    assert len(turn["raw_stdout_sha256"]) == 64
    assert "private-token" not in str(journal.events)


def test_codex_records_redacted_startup_diagnostics_and_independent_usage_excess(monkeypatch):
    lines = stream(usage={**USAGE, "input_tokens": 181_919}).splitlines()
    lines.insert(1, json.dumps({"type": "item.completed", "item": {
        "type": "error", "id": "item_0",
        "message": "Config warning: model rerouted from gpt-6.2-sol to gpt-6.2-astra (/Users/example/.codex/config.toml)",
    }}))
    raw = "\n".join(lines) + "\n"
    stderr = "WARN failed to connect: HTTP 429 Authorization: Bearer private-secret default openai"
    mock_cli(monkeypatch, iter([ProcessResult(raw, stderr, 0, None)]))
    journal = Journal()
    editor = CodexEditor(CodexOptions(**SETTINGS), journal, allow_live=True)
    editor.preflight()
    with pytest.raises(EditorError, match="unexpected diagnostics"):
        editor.request("{}", call=1, attempt=1)
    turn = journal.events[-1]
    assert turn["thread_id"] == THREAD
    assert turn["usage_excesses"] == {"input_tokens": {"observed": 181_919, "limit": 120_000}}
    assert turn["diagnostics"] == [
        {
            "source": "item.completed:error", "redacted": True, "truncated": False,
            "preview": "Config warning: model rerouted from gpt-6.2-sol to gpt-6.2-astra ([home path redacted])",
        },
        {
            "source": "stderr", "redacted": True, "truncated": False,
            "preview": "WARN failed to connect: HTTP 429 [credential details redacted]",
        },
    ]
    assert turn["raw_stdout"] is None and turn["raw_stderr"] is None
    assert turn["raw_stdout_sha256"] == sha256(raw.encode()).hexdigest()
    assert turn["raw_stderr_sha256"] == sha256(stderr.encode()).hexdigest()
    assert not any(word in str(turn["diagnostics"]) for word in (
        "private-secret", "/Users/example", "default openai",
    ))


def test_codex_diagnostic_previews_bound_payloads_and_exclude_tool_details():
    message = "Unexpected response " + "sk-private-secret " * 1000
    events = [
        {"type": "item.completed", "item": {"type": "command_execution", "message": message}},
        *({"type": "turn.failed", "error": {"message": message}} for _ in range(20)),
    ]
    evidence = codex_editor._diagnostic_evidence(events, "\n".join([message] * 20))
    assert len(evidence) == 16
    assert [entry["source"] for entry in evidence] == ["turn.failed"] * 8 + ["stderr"] * 8
    assert all(
        entry["preview"].startswith("Unexpected response [credential redacted]")
        and len(entry["preview"]) == 512 and entry["redacted"] and entry["truncated"]
        and "private-secret" not in entry["preview"]
        for entry in evidence
    )


@pytest.mark.parametrize(("text", "private_value"), [
    ("Warning: OPENAI_API_KEY=example-private-value", "example-private-value"),
    ('Warning: {"CODEX_ACCESS_TOKEN":"example-private-value"}', "example-private-value"),
    ("Warning: oauth_code=example-private-value", "example-private-value"),
    ("Warning: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmb28ifQ.testSignature", "testSignature"),
    ("Warning: sk-proj-example-private-value", "example-private-value"),
    ("Warning: AbCdEfGhIjKlMnOpQrStUv56789012", "AbCdEfGhIjKlMnOpQrStUv56789012"),
    ("Warning: access to /home/private-user/.codex was denied", "private-user"),
    ("Warning: https://example.test/api?secret-example=private-value", "private-value"),
])
def test_codex_diagnostics_redact_recognizable_private_values(text, private_value):
    result = codex_editor._diagnostic_preview(text)
    assert result["redacted"] and result["preview"].startswith("Warning: ")
    assert private_value not in result["preview"]


def test_codex_diagnostic_keeps_unrecognized_warning_explanation():
    message = "Warning: Experimental posture resolved to an unsupported enforcement mode"
    assert codex_editor._diagnostic_preview(message) == {
        "preview": message, "redacted": False, "truncated": False,
    }


@pytest.mark.parametrize("proposal", [
    "commentary with private-token",
    '{"action":"select","candidate_id":0,"candidate_id":1}',
    '{"action":"probe","ops":[{"op":"zero_neuron","layer":"\\ud800","neuron":0}]}',
    '{"action":"probe","ops":[{"op":"zero_neuron","layer":"private-token","neuron":0}]}',
])
def test_codex_proposal_failure_is_separate_from_stream_failure(monkeypatch, proposal):
    mock_cli(monkeypatch, iter([stream(proposal)]))
    journal = Journal()
    editor = CodexEditor(CodexOptions(**SETTINGS), journal, allow_live=True)
    editor.preflight()
    with pytest.raises(InvalidProposal):
        editor.request("{}", call=1, attempt=1)
    turn = journal.events[-1]
    assert turn["stream_error"] is None and turn["proposal_error"]
    assert turn["raw_stdout"] is None and turn["response"] is None
    assert "private-token" not in str(journal.events)


def test_codex_requires_live_opt_in():
    with pytest.raises(EditorError, match="live opt-in"):
        CodexEditor(CodexOptions(**SETTINGS), Journal())


def test_model_catalog_must_match_pin_and_selected_model(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "archive"
    output.mkdir()
    source.write_text('{"slug":"wrong-model","tool_mode":"code_mode_only"}')
    settings = {**SETTINGS, "model_catalog_source": str(source)}
    with pytest.raises(ValueError, match="pin the source"):
        CodexOptions(**settings)
    with pytest.raises(ValueError, match="differs from the configured pin"):
        codex_editor.stage_model_catalog(CodexOptions(**settings, model_catalog_source_sha256="0" * 64), output)
    with pytest.raises(ValueError, match="only the configured model entry"):
        codex_editor.stage_model_catalog(
            CodexOptions(**settings, model_catalog_source_sha256=sha256(source.read_bytes()).hexdigest()), output,
        )
    assert not list(output.iterdir())


def test_catalog_archive_drift_fails_before_launch(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text('{"slug":"gpt-6.2-sol","tool_mode":"code_mode_only"}')
    options = CodexOptions(
        **SETTINGS, model_catalog_source=str(source), model_catalog_source_sha256=sha256(source.read_bytes()).hexdigest(),
    )
    output = tmp_path / "archive"
    output.mkdir()
    archived = codex_editor.stage_model_catalog(options, output)
    assert archived["tool_mode"] == "direct"
    launches = mock_cli(monkeypatch, iter([]))
    journal = Journal()
    editor = CodexEditor(options, journal, allow_live=True, catalog_archive=output)
    editor.preflight()
    assert journal.events[-1]["model_catalog"] == archived
    (output / archived["catalog_file"]).write_text('{"models":[]}')
    with pytest.raises(EditorError, match="pinned Direct-mode derivative"):
        editor.request("{}", call=1, attempt=1)
    assert not launches
    assert journal.events[-1]["error_type"] == "model_catalog"


def test_process_reports_invalid_utf8_and_hashes_the_original_capture(tmp_path):
    result = codex_editor.run_process(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes([255]))"],
        prompt="", cwd=tmp_path, env={}, timeout=5, output_cap=8192,
    )
    assert result.returncode == 0 and result.invalid_utf8
    assert result.stdout_sha256 == sha256(bytes([255])).hexdigest()
