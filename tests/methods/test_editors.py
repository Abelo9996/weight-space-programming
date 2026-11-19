import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from weightspace import editors
from weightspace.editors import ClaudeEditor, ClaudeOptions, EditorError, InvalidProposal, ProcessResult
from weightspace.edits import EditLimits


MODEL = "claude-haiku-5-5"
AUTH = {
    "loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
    "subscriptionType": "max", "email": "private@example.test",
}


class Journal:
    def __init__(self):
        self.events = []

    def append(self, event, **fields):
        self.events.append({"event": event, **fields})


def options(**changes):
    return ClaudeOptions(**({"model": MODEL, "billing_assumption": "extra_usage_disabled"} | changes))


def cli_stream(
    response, *, init_fields=None, assistant_fields=None, result_fields=None,
    extra_event=None, wrong_order=False, alternate_message=None,
):
    events = [
        {"type": "system", "subtype": "init", "session_id": "session", "model": MODEL, "apiKeySource": "none",
         "tools": [], "mcp_servers": [], "skills": [], "plugins": [], **(init_fields or {})},
        {"type": "assistant", "session_id": "session", "parent_tool_use_id": None, "message": {
            "id": "message", "model": MODEL, "content": [{"type": "text", "text": response}],
            **(assistant_fields or {}),
        }},
        {"type": "result", "subtype": "success", "session_id": "session", "is_error": False,
         "result": response, "num_turns": 1, "permission_denials": [],
         "usage": {"input_tokens": 12, "output_tokens": 3}, "total_cost_usd": 0.002, **(result_fields or {})},
    ]
    if alternate_message:
        events[1]["message"] = alternate_message
    if extra_event:
        events.insert(1, {"session_id": "session", **extra_event})
    if wrong_order:
        events[-1], events[-2] = events[-2], events[-1]
    return "\n".join(map(json.dumps, events)) + "\n"


def mocked_cli(monkeypatch, responses):
    inputs = []

    def invoke(command, *, prompt, cwd, env, timeout, output_cap):
        assert isinstance(command, list)
        assert cwd.is_dir() and not list(cwd.iterdir())
        assert "ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
        assert "CLAUDE_CODE_USE_BEDROCK" not in env
        assert "ANTHROPIC_BASE_URL" not in env
        if command == ["claude", "--version"]:
            return ProcessResult("2.1.326 (Claude Code)\n", "", 0, None)
        if command == ["claude", "auth", "status", "--json"]:
            return ProcessResult(json.dumps(AUTH), "", 0, None)
        for name, value in (
            ("--tools", ""), ("--setting-sources", ""), ("--mcp-config", '{"mcpServers":{}}'),
            ("--permission-prompts", "none"), ("--disallowed-tools", "*"),
            ("--max-turns", "1"), ("--model", MODEL), ("--output-format", "stream-json"),
        ):
            assert command[command.index(name) + 1] == value
        for name in (
            "--safe-mode", "--restricted", "--strict-mcp-config", "--no-session-persistence",
            "--disable-slash-commands", "--system-prompt",
        ):
            assert name in command
        assert "--bare" not in command and "--plugin-dir" not in command
        assert isinstance(prompt, str) and prompt not in command
        assert timeout == 120 and output_cap == 1_000_000
        inputs.append(prompt)
        return ProcessResult(next(responses), "", 0, None)

    monkeypatch.setattr(editors, "run_process", invoke)
    return inputs


def candidate(i, ops):
    return SimpleNamespace(
        candidate_id=i, changed_entries=len(ops), status="scored", reason=None,
        score=SimpleNamespace(objective=0.5, failure_success=0, normal_success=1),
    )


def test_native_cli_receives_only_weights_task_and_development_feedback(monkeypatch):
    for key in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK", "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.setenv(key, "secret-override")
    ops = [{"op": "shift_bias", "layer": "fc1", "neuron": 0, "shift": 0.5}]
    raw = [
        cli_stream(json.dumps({"action": "probe", "ops": ops})),
        cli_stream(json.dumps({"action": "select", "candidate_id": 1})),
    ]
    inputs = mocked_cli(monkeypatch, iter(raw))
    calls = []

    def probe(edits):
        calls.append(edits)
        return candidate(1, edits)

    dev = SimpleNamespace(
        limits=EditLimits(), budget=SimpleNamespace(candidates_remaining=1), probe=probe,
        policy=lambda: {"fc0.bias": np.array([0.5, 2.0])},
        task=SimpleNamespace(
            id="fixture", name="Fixture", sizes=(1, 2, 1), horizon=8,
            description="Repair a small policy.", normalization="identity", action_rule="argmax",
            success_rule="survive", evaluator_source="private-marker", private_path=Path("private/arrays"),
            heldout=np.array([99999]), reference_delta="private-reference",
        ),
    )
    journal = Journal()
    editor = ClaudeEditor(options(max_calls=2), journal, allow_live=True)
    assert editors.run_editor(dev, candidate(0, []), editor) == 1
    assert calls == [ops]
    assert [json.loads(prompt)["history"][-1]["candidate_id"] for prompt in inputs] == [0, 1]
    for prompt in inputs:
        assert "private" not in prompt and "heldout" not in prompt and "99999" not in prompt
        assert "secret-override" not in prompt
    assert [event["raw_stdout"] for event in journal.events if event["event"] == "editor_turn"] == raw
    assert "private@example.test" not in str(journal.events)
    assert journal.events[-1]["actual_charge_usd"] is None
    assert journal.events[-1]["cli_api_equivalent_estimate_usd"] == 0.002


@pytest.mark.parametrize("response", [
    'Here is {"action":"select","candidate_id":0}',
    '{"action":"select","candidate_id":0,"candidate_id":1}',
    '{"action":"select","candidate_id":true}',
    '{"action":"probe","ops":[{"op":"shift_bias","layer":"fc0","neuron":0,"shift":NaN}]}',
    '{"action":"probe","ops":[{"op":"shift_bias","layer":"fc0","neuron":0,"shift":1e999}]}',
    '{"action":"probe","ops":[{"op":"run_code","code":"print(1)"}]}',
])
def test_strict_proposal_parser(response):
    with pytest.raises(InvalidProposal):
        editors.parse_proposal(response)


@pytest.mark.parametrize("override", [
    {"init_fields": {"tools": ["Bash"]}},
    {"init_fields": {"plugins": [{"name": "unwanted"}]}},
    {"init_fields": {"apiKeySource": "ANTHROPIC_API_KEY"}},
    {"assistant_fields": {"model": "another-model"}},
    {"assistant_fields": {"content": [{"type": "tool_use", "name": "Read"}]}},
    {"extra_event": {"type": "user", "message": {"content": [{"type": "tool_result"}]}}},
    {"extra_event": {"type": "system", "subtype": "hook_response"}},
    {"extra_event": {"type": "unexpected"}},
    {"extra_event": {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed_warning", "overageEnabled": True}}},
    {"extra_event": {"type": "assistant", "message": {"id": "second-message", "model": MODEL, "content": []}}},
    {"wrong_order": True},
    {"alternate_message": "invalid-value"},
    {"result_fields": {"num_turns": 2}},
    {"result_fields": {"session_id": "another-session"}},
    {"result_fields": {"deferred_tool_use": {"name": "Read"}}},
    {"result_fields": {"result": '{"action":"select","candidate_id":1}'}},
    {"result_fields": {"terminal_reason": "hook_stopped"}},
    {"result_fields": {"terminal_reason": "stop_hook_prevented"}},
    {"result_fields": {"terminal_reason": None}},
    {"result_fields": {"local_command": "reload_plugins"}},
    {"result_fields": {"local_command": None}},
])
def test_cli_tool_and_model_drift_fail_closed(monkeypatch, override):
    raw = cli_stream('{"action":"select","candidate_id":0}', **override)
    mocked_cli(monkeypatch, iter([raw]))
    journal = Journal()
    editor = ClaudeEditor(options(), journal, allow_live=True)
    editor.preflight()
    with pytest.raises(EditorError):
        editor.request("{}", call=1, attempt=1)
    assert journal.events[-1]["raw_stdout"] == raw


def test_claude_reports_allowed_rate_limit_event(monkeypatch):
    raw = cli_stream(
        '{"action":"select","candidate_id":0}',
        extra_event={"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}},
        result_fields={"terminal_reason": "completed"},
    )
    mocked_cli(monkeypatch, iter([raw]))
    editor = ClaudeEditor(options(), Journal(), allow_live=True)
    editor.preflight()
    assert editor.request("{}", call=1, attempt=1) == {"action": "select", "candidate_id": 0}


def test_process_output_cap_uses_no_claude(tmp_path):
    result = editors.run_process(
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read() + 'x' * 50000)"],
        prompt="payload", cwd=tmp_path, env={}, timeout=5, output_cap=512,
    )
    assert result.interruption == "output_cap"
    assert result.stdout.startswith("payload")
    assert len(result.stdout) + len(result.stderr) == 512


def test_unapproved_editor_cannot_start(monkeypatch):
    monkeypatch.setattr(editors, "run_process", lambda *args, **kwargs: pytest.fail("unexpected process"))
    with pytest.raises(EditorError):
        ClaudeEditor(options(), Journal())
