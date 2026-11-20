from decimal import Decimal
from hashlib import sha256
import json

import pytest

from weightspace.cli import main
from weightspace.readers.analysis import summarize
from weightspace.readers.experiment import load_config, prepare_plan, run_plan
from weightspace.readers.providers import WireResponse


KEYS = {"openai": "non-live-openai-credential-917", "anthropic": "non-live-anthropic-credential-283"}


@pytest.fixture
def fake_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEYS["openai"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEYS["anthropic"])


def wire_response(provider, model, prediction, *, input_tokens=100, id="test-response", with_usage=True):
    content = json.dumps({"accuracy": prediction})
    if provider == "openai":
        response = {
            "object": "response", "status": "completed", "model": model, "id": id,
            "output": [{"type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": content}]}],
            "usage": {"input_tokens": input_tokens, "output_tokens": 8, "total_tokens": input_tokens + 8,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}},
        }
    else:
        response = {
            "model": model, "id": id, "type": "message", "role": "assistant", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": content}],
            "usage": {"input_tokens": input_tokens, "output_tokens": 8,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        }
    if not with_usage:
        del response["usage"]
    return WireResponse(200, json.dumps(response).encode(), "test-http-request-id")


@pytest.mark.parametrize(("settings", "blocker"), [
    ({"budget": "0"}, "budget_usd"),
    ({"budget": "0.00001"}, "budget_usd"),
    ({"credits": {}}, "confirmed_credits_usd"),
    ({"credits": {"openai": "0.00001", "anthropic": "5"}}, "confirmed_credits_usd"),
    ({"context": 1200}, "context window"),
])
def test_reading_live_run_checks_entire_budget_before_transport(
    panel_factory, config_factory, fake_keys, tmp_path, settings, blocker,
):
    panel, _ = panel_factory(test=1)
    config = load_config(config_factory(**settings))
    plan = prepare_plan(config, panel)
    output = tmp_path / "run"
    assert any(blocker in message for message in plan.metadata["live_preflight_blockers"])
    with pytest.raises(ValueError, match=blocker):
        run_plan(plan, output, allow_api=True, confirm_credits=True, transport=lambda *_: pytest.fail("transport ran"))
    assert not output.exists()


def test_cli_offline_plan_and_no_accidental_api_execution(panel_factory, config_factory, tmp_path, monkeypatch):
    panel, _ = panel_factory(test=1)
    config = config_factory()
    monkeypatch.setattr("weightspace.readers.experiment.send_request", lambda *_: pytest.fail("transport ran"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = ["--config", str(config), "--panel", str(panel)]
    assert main(["reading", "plan", *args, "--output", str(tmp_path / "plan.json")]) == 0
    for opts in ([], ["--allow-api"], ["--confirm-credits"], ["--allow-api", "--confirm-credits"]):
        with pytest.raises(SystemExit) as error:
            main(["reading", "run", *args, "--output", str(tmp_path / "run"), *opts])
        assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_offline_plan_does_not_store_embedded_environment_key(
    panel_factory, config_factory, fake_keys, tmp_path, capsys,
):
    panel, _ = panel_factory(test=1)
    config = config_factory()
    raw = json.loads(config.read_text())
    raw["readers"][0]["price"]["source"] += "/" + KEYS["openai"]
    config.write_text(json.dumps(raw))
    output = tmp_path / "private-plan.json"
    with pytest.raises(SystemExit) as error:
        main(["reading", "plan", "--config", str(config), "--panel", str(panel), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()
    assert KEYS["openai"] not in capsys.readouterr().err


@pytest.mark.parametrize(("provider", "variable", "token"), [
    ("anthropic", "ANTHROPIC_API_KEY", "sk-ant-oat01-test-only"),
    ("openai", "OPENAI_API_KEY", "eyJfake.payload.signature"),
    ("openai", "OPENAI_API_KEY", "sess-test-only"),
])
def test_recognizable_access_and_session_tokens_are_rejected(
    panel_factory, config_factory, fake_keys, tmp_path, monkeypatch, provider, variable, token,
):
    panel, _ = panel_factory(test=1)
    plan = prepare_plan(load_config(config_factory(providers=(provider,))), panel)
    monkeypatch.setenv(variable, token)
    output = tmp_path / "invalid-credential-run"
    with pytest.raises(ValueError, match="must contain a first-party API key"):
        run_plan(plan, output, allow_api=True, confirm_credits=True, transport=lambda *_: pytest.fail("transport ran"))
    assert not output.exists()


def test_large_job_requires_explicit_approval(panel_factory, config_factory, fake_keys, tmp_path):
    panel, _ = panel_factory(test=1)
    path = config_factory(providers=("openai",), budget="1000", credits={"openai": "1000"})
    raw = json.loads(path.read_text())
    raw["readers"][0]["price"]["standard"]["output"] = "50000"
    path.write_text(json.dumps(raw))
    plan = prepare_plan(load_config(path), panel)
    assert not plan.metadata["live_preflight_blockers"]
    assert plan.metadata["requires_over_threshold_approval"]
    output = tmp_path / "high-budget"
    with pytest.raises(ValueError, match="above \\$40"):
        run_plan(plan, output, allow_api=True, confirm_credits=True, transport=lambda *_: pytest.fail("transport ran"))
    assert not output.exists()
    result = run_plan(plan, output, allow_api=True, confirm_credits=True, approve_over_40=True,
                      transport=lambda *_: WireResponse(401, b"test approval path"))
    assert result["attempted"] == 1
    assert json.loads((output / "manifest.json").read_text())["metadata"]["approved_over_threshold"] is True


def test_successful_shapes_budget_and_scored_complete_panels(
    panel_factory, config_factory, fake_keys, tmp_path,
):
    panel, _ = panel_factory(test=2)
    plan = prepare_plan(load_config(config_factory(repeats=3)), panel)
    assert not plan.metadata["live_preflight_blockers"]
    ordered = iter(plan.metadata["requests"])
    bodies = []
    responses = []
    network_ids = sorted(plan.variants)

    def send(endpoint, body, headers, timeout):
        expected = next(ordered)
        provider = expected["provider"]
        assert endpoint.url == expected["endpoint"]
        assert timeout == 10
        assert headers.get("Authorization", headers.get("x-api-key")) in (KEYS[provider], f"Bearer {KEYS[provider]}")
        assert expected["request_sha256"] == sha256(body).hexdigest()
        request = json.loads(body)
        text = request["input"] if provider == "openai" else request["messages"][0]["content"]
        assert text == plan.prompt(expected)
        assert all(key not in text for key in KEYS.values())
        prediction = 0.3 + 0.1 * network_ids.index(expected["network_sha256"])
        if expected["condition"] == "shuffle":
            prediction += 0.4
        elif expected["condition"] == "permutation":
            prediction += 0.2
        bodies.append(expected["request_sha256"])
        response = wire_response(provider, request["model"], prediction)
        responses.append(response.body.decode())
        return response

    output = tmp_path / "reader-run"
    outcome = run_plan(plan, output, allow_api=True, confirm_credits=True, transport=send)
    assert outcome["attempted"] == outcome["completed"] == 60
    assert outcome["attempted_reservation_usd"] == plan.metadata["projected_reservation_usd"]
    assert outcome["actual_charge_usd"] is None
    assert len(set(bodies)) == 20
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    finished = [event for event in events if event["event"] == "call_finished"]
    assert [event["raw_response"] for event in finished] == responses
    for event in finished:
        raw = event["raw_response"].encode()
        assert event["wire_response_bytes_received"] == len(raw)
        assert event["response_sha256"] == event["wire_response_sha256"] == sha256(raw).hexdigest()
    assert not any(key in (output / "events.jsonl").read_text() for key in KEYS.values())
    with pytest.raises(FileExistsError):
        run_plan(plan, output, allow_api=True, confirm_credits=True, transport=lambda *_: pytest.fail("retried"))
    summary = summarize(output, bootstrap_replicates=20)
    assert summary["counts"] == {"completed": 60, "failed": 0, "interrupted": 0, "unattempted": 0}
    for group in (group for group in summary["groups"] if group["activation"] == "all"):
        assert group["accuracy"]["original"]["n_networks"] == 2
        assert set(group["accuracy"]) == {"original", "permutation", "activation_symmetry", "composition"}
        assert group["symmetry"]["permutation"]["symmetry_gap"] == {"estimate": 1, "n_networks": 2, "ci95": None}
        assert group["symmetry"]["all_equivalent"]["symmetry_gap"]["estimate"] == pytest.approx(1 / 3)
        assert group["shuffle_response_control"]["condition_agreement"] == 0
        assert group["shuffle_response_control"]["condition_mean_absolute_difference"] == pytest.approx(0.4)
    finished[0]["raw_response"] += " "
    (output / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(ValueError, match="response digest"):
        summarize(output)


def test_failed_attempts_keep_reservations_and_redact_keys(panel_factory, config_factory, fake_keys, tmp_path):
    panel, _ = panel_factory(test=1)
    plan = prepare_plan(load_config(config_factory()), panel)
    schedule = iter(plan.metadata["requests"])
    transport_calls = []

    def send(endpoint, body, headers, timeout):
        cell = next(schedule)
        transport_calls.append(cell)
        key1, key2 = KEYS.values()
        if len(transport_calls) == 1:
            raise TimeoutError(f"failed with credential {key1}\nAuthorization: Bearer {key2}")
        if len(transport_calls) == 2:
            escaped = key2.replace("n", r"\u006e")
            return WireResponse(500, f'{{"error":"{key1},{escaped}","Authorization":"test"}}'.encode())
        if len(transport_calls) == 3:
            return WireResponse(200, b"")  # malformed provider envelope
        if len(transport_calls) == 4:
            return wire_response(cell["provider"], cell["requested_model"], 1.5)
        if len(transport_calls) == 5:
            return wire_response(cell["provider"], cell["requested_model"], 0.71, with_usage=False)
        return wire_response(cell["provider"], cell["requested_model"], 0.72)

    output = tmp_path / "reader-run-failures"
    outcome = run_plan(plan, output, allow_api=True, confirm_credits=True, transport=send)
    assert outcome["failed"] == outcome["unknown_usage"] == 4
    assert outcome["unattempted"] == 0
    assert outcome["attempted_reservation_usd"] == plan.metadata["projected_reservation_usd"]
    assert Decimal(outcome["known_quoted_usage_usd"]) < Decimal(outcome["attempted_reservation_usd"])
    assert len({call["call_id"] for call in transport_calls}) == 20
    raw = "".join(path.read_text() for path in output.iterdir())
    assert all(key not in raw for key in KEYS.values())
    assert "Authorization" not in raw
    assert "[redacted credential]" in raw
    summary = summarize(output)
    assert summary["counts"]["failed"] == 4
    assert summary["billing"]["unknown_usage_attempts"] == 4
    assert len([row for row in summary["availability"] if row["usage_measured"] is False]) == 4
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    changed = [event for event in events if event.get("raw_response_changed")]
    assert len(changed) == 1 and changed[0]["wire_response_sha256"] is None
    assert changed[0]["response_sha256"] == sha256(changed[0]["raw_response"].encode()).hexdigest()


@pytest.mark.parametrize("drift", ["token overage", "different model", "credential rejection", "service tier", "region", "tools"])
def test_request_stops_after_observed_drift(panel_factory, config_factory, fake_keys, tmp_path, drift):
    panel, _ = panel_factory(test=1)
    provider = "anthropic" if drift in ("region", "tools") else "openai"
    plan = prepare_plan(load_config(config_factory(providers=(provider,))), panel)
    first = plan.metadata["requests"][0]
    calls = []

    def send(endpoint, body, headers, timeout):
        calls.append(body)
        if drift == "credential rejection":
            return WireResponse(401, b"denied")
        model = "other-model" if drift == "different model" else first["requested_model"]
        tokens = first["input_tokens_reserved_estimate"] + 1 if drift == "token overage" else 100
        wire = wire_response(first["provider"], model, 0.7, input_tokens=tokens)
        response = json.loads(wire.body)
        if drift == "service tier":
            response["service_tier"] = "priority"
        if drift == "region":
            response["usage"]["inference_geo"] = "us"
        if drift == "tools":
            response["usage"]["server_tool_use"] = {"web_fetch_requests": 1}
        return WireResponse(wire.status, json.dumps(response).encode())

    output = tmp_path / "reader-run-drift"
    result = run_plan(plan, output, allow_api=True, confirm_credits=True, transport=send)
    assert len(calls) == 1
    assert result["attempted"] == 1
    assert result["unattempted"] == plan.metadata["planned_calls"] - 1
    assert result["attempted_reservation_usd"] == first["reserved_usd"]
    assert result["stop_reason"] is not None
    if drift in ("different model", "service tier", "region", "tools"):
        assert result["failed"] == result["unknown_usage"] == 1
    assert summarize(output)["stop_reason"] == result["stop_reason"]
