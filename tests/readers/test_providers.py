from decimal import Decimal
import json

import pytest

from weightspace.readers.pricing import RateCard
from weightspace.readers.providers import (
    ENDPOINTS, CredentialFilter, WireResponse, normalize_usage, parse_accuracy, parse_response, request_body,
    send_request,
)


@pytest.mark.parametrize("payload", [
    "", "0.5", '{"accuracy":true}', '{"accuracy":"0.5"}', '{"accuracy":NaN}', '{"accuracy":Infinity}',
    '{"accuracy":1e6000}', '{"accuracy":-0.0001}', '{"accuracy":1.0001}', '{"accuracy":0.5,"note":"x"}',
    '{"accuracy":-1e-999}', '{"accuracy":1.00000000000000000001}',
    '{"accuracy":0.5,"accuracy":0.2}', '```json\n{"accuracy":0.5}\n```', '{"accuracy":0.5} more',
])
def test_accuracy_response_contract(payload):
    with pytest.raises(ValueError):
        parse_accuracy(payload)


def test_provider_request_shape():
    prompt = "identical target architecture and weights"
    a = json.loads(request_body("openai", "gpt-6-luna", prompt, 512, "low"))
    b = json.loads(request_body("anthropic", "claude-haiku-5-5", prompt, 512, "low"))
    assert a["input"] == b["messages"][0]["content"] == prompt
    assert a["max_output_tokens"] == b["max_tokens"] == 512
    assert a["text"]["format"]["schema"] == b["output_config"]["format"]["schema"]
    assert a["text"]["format"]["strict"] and a["store"] is False
    assert a["truncation"] == "disabled" and a["service_tier"] == "default"
    assert b["inference_geo"] == "global" and b["service_tier"] == "standard_only"
    assert "tools" not in a and "tools" not in b
    assert ENDPOINTS["openai"].url == "https://api.openai.com/v1/responses"
    assert ENDPOINTS["anthropic"].url == "https://api.anthropic.com/v1/messages"


def test_response_requires_expected_model_and_complete_assistant_output():
    message = {
        "type": "message", "status": "completed", "role": "assistant",
        "content": [{"type": "output_text", "text": '{"accuracy":0}'}],
    }
    body = {"model": "m", "object": "response", "status": "completed", "output": [{"type": "reasoning"}, message]}
    assert parse_response("openai", body, "m") == 0
    with pytest.raises(ValueError, match="different|does not match"):
        parse_response("openai", body, "other")
    with pytest.raises(ValueError, match="did not complete"):
        parse_response("openai", body | {"status": "incomplete"}, "m")
    with pytest.raises(ValueError, match="unsupported"):
        parse_response("openai", body | {"output": [message, {"type": "function_call"}]}, "m")
    with pytest.raises(ValueError, match="unsupported"):
        parse_response("openai", body | {"output": [message | {"content": [{"type": "refusal"}]}]}, "m")

    body = {
        "type": "message", "role": "assistant", "stop_reason": "end_turn", "model": "m",
        "content": [{"type": "thinking"}, {"type": "text", "text": '{"accuracy":1}'}],
    }
    assert parse_response("anthropic", body, "m") == 1
    with pytest.raises(ValueError, match="did not end normally"):
        parse_response("anthropic", body | {"stop_reason": "max_tokens"}, "m")
    with pytest.raises(ValueError, match="unsupported"):
        parse_response("anthropic", body | {"content": body["content"] + [{"type": "tool_use"}]}, "m")


def test_billable_usage_categories_and_long_context_quote():
    op = normalize_usage("openai", {
        "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
        "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 20},
        "output_tokens_details": {"reasoning_tokens": 7},
    })
    an = normalize_usage("anthropic", {
        "input_tokens": 40, "output_tokens": 10, "cache_read_input_tokens": 40, "cache_creation_input_tokens": 20,
        "cache_creation": {"ephemeral_5m_input_tokens": 15, "ephemeral_1h_input_tokens": 5},
    })
    assert op == {"input": 40, "cache_read": 40, "cache_write": 20, "cache_write_long": 0,
                  "output": 10, "input_total": 100}
    assert an == op | {"cache_write": 15, "cache_write_long": 5}
    standard = {"input": "1", "cache_read": "0.1", "cache_write": "1.25", "cache_write_long": "2", "output": "5"}
    rates = RateCard.read({
        "source": "https://example.invalid", "as_of": "2026-01-01", "standard": standard,
        "long_context_above_input_tokens": 99, "long_context": standard | {"output": "10"},
    }, 1000)
    assert rates.reservation(100, 10) == Decimal("0.0003")
    assert rates.estimate(op) == Decimal("0.000169")
    assert rates.estimate(an) == Decimal("0.00017275")
    with pytest.raises(ValueError, match="usage"):
        normalize_usage("anthropic", {"input_tokens": 1, "output_tokens": True})
    with pytest.raises(ValueError, match="unknown"):
        normalize_usage("anthropic", {
            "input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 1,
        })


def test_filter_preserves_ordinary_response_and_redacts_escapes():
    key = "test-credential-zx298"
    redactor = CredentialFilter([key])
    raw = b'{ "message":"plain provider response" }'
    assert redactor.response(raw)[0].encode() == raw
    escaped = key.replace("t", r"\u0074")
    for payload in (
        f'{{"error":"{escaped}"}}',
        f'{{"error":"{escaped}","error":"x"}}',
        f'{{"error":"{escaped}',  # Truncated JSON.
        json.dumps({"error": f'{{"echo":"{escaped}"}}'}),
        f'{{"Authorization":"Bearer {key}","x-api-key":"{escaped}"}}',
        f'failed with headers {{"Authorization":"Bearer {key}"}}',
        f'Authorization: Bearer {key}',
    ):
        safe, _ = redactor.response(payload.encode())
        assert key not in safe
        assert escaped not in safe
        assert "Authorization" not in safe


def test_fixed_direct_http_transport_one_attempt(monkeypatch):
    calls = []
    class Response:
        status = 307
        length = None
        def __enter__(self):
            return self
        def __exit__(self, *error):
            pass
        def getheader(self, name):
            return "fake-request-id" if name == "x-request-id" else None
        def read(self, _size):
            return b"do not follow"
    class Connection:
        def __init__(self, host, timeout):
            calls.append((host, timeout))
        def request(self, method, path, *, body, headers):
            calls.append((method, path, body, headers["Authorization"]))
        def getresponse(self):
            return Response()
        def close(self):
            calls.append("close")

    monkeypatch.setattr("weightspace.readers.providers.http.client.HTTPSConnection", Connection)
    endpoint = ENDPOINTS["openai"]
    result = send_request(endpoint, b"request", endpoint.headers("fake-key"), 10)
    assert result == WireResponse(307, b"do not follow", "fake-request-id")
    assert calls == [("api.openai.com", 10), ("POST", "/v1/responses", b"request", "Bearer fake-key"), "close"]
    Response.length = 120
    assert send_request(endpoint, b"request", endpoint.headers("fake-key"), 10).truncated
