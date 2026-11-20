"""First-party API request and response formats for the shared reader prompt."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import http.client
import math
import re

from ..artifacts import canonical_json
from .data import loads


@dataclass(frozen=True)
class Endpoint:
    host: str
    path: str
    key_variable: str
    request_id_header: str
    version: str | None = None

    @property
    def url(self):
        return f"https://{self.host}{self.path}"

    def headers(self, key):
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.version is None:
            headers["Authorization"] = f"Bearer {key}"
        else:
            headers.update({"anthropic-version": self.version, "x-api-key": key})
        return headers


ENDPOINTS = {
    "openai": Endpoint("api.openai.com", "/v1/responses", "OPENAI_API_KEY", "x-request-id"),
    "anthropic": Endpoint("api.anthropic.com", "/v1/messages", "ANTHROPIC_API_KEY", "request-id", "2026-10-21"),
}
ANSWER_SCHEMA = {
    "type": "object", "properties": {"accuracy": {"type": "number"}},
    "required": ["accuracy"], "additionalProperties": False,
}
MAX_RESPONSE_BYTES = 4_000_000


@dataclass(frozen=True)
class WireResponse:
    status: int
    body: bytes
    request_id: str | None = None
    truncated: bool = False


def send_request(endpoint, body, headers, timeout):
    """Make one direct HTTPS request, with no SDK retries, redirects, or proxy."""
    connection = http.client.HTTPSConnection(endpoint.host, timeout=timeout)
    try:
        connection.request("POST", endpoint.path, body=body, headers=headers)
        with connection.getresponse() as response:
            request_id = response.getheader(endpoint.request_id_header)
            try:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            except http.client.IncompleteRead as error:
                return WireResponse(response.status, error.partial, request_id, truncated=True)
            truncated = len(raw) > MAX_RESPONSE_BYTES or response.length not in (None, 0)
            return WireResponse(
                response.status, raw[:MAX_RESPONSE_BYTES], request_id, truncated=truncated,
            )
    finally:
        connection.close()


def request_body(provider, model, prompt, max_tokens, effort):
    if provider == "openai":
        body = {
            "model": model, "input": prompt, "max_output_tokens": max_tokens,
            "reasoning": {"effort": effort}, "service_tier": "default", "store": False, "truncation": "disabled",
            "text": {"format": {
                "type": "json_schema", "name": "cnn_accuracy", "strict": True, "schema": ANSWER_SCHEMA,
            }},
        }
    elif provider == "anthropic":
        body = {
            "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "inference_geo": "global", "service_tier": "standard_only", "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        }
    else:
        raise ValueError("provider must be openai or anthropic")
    return canonical_json(body)


def parse_accuracy(text):
    if not isinstance(text, str):
        raise ValueError("model output must be JSON text")
    try:
        obj = loads(text, parse_float=Decimal, parse_int=Decimal, parse_constant=Decimal)
    except InvalidOperation:
        raise ValueError("model output contains an invalid accuracy number") from None
    if not isinstance(obj, dict) or set(obj) != {"accuracy"}:
        raise ValueError("model output must have exactly one accuracy field")
    accuracy = obj["accuracy"]
    if not isinstance(accuracy, Decimal) or not accuracy.is_finite() or not 0 <= accuracy <= 1:
        raise ValueError("predicted accuracy must be a finite number in [0, 1]")
    return float(accuracy)


def _one_output(blocks, *, ignore_types, output_type):
    if not isinstance(blocks, list) or any(not isinstance(block, dict) for block in blocks):
        raise ValueError("provider content must be a list of blocks")
    outputs = [block for block in blocks if block.get("type") not in ignore_types]
    if len(outputs) != 1 or outputs[0].get("type") != output_type:
        raise ValueError(f"expected one {output_type} block with no unsupported outputs")
    return outputs[0]


def parse_response(provider, response, requested_model):
    if not isinstance(response, dict):
        raise ValueError("provider response must be a JSON object")
    if response.get("model") != requested_model:
        raise ValueError("reported model does not match requested model")
    if provider == "openai":
        if response.get("object") != "response" or response.get("status") != "completed" or response.get("error"):
            raise ValueError("OpenAI response did not complete")
        message = _one_output(response.get("output"), ignore_types=("reasoning",), output_type="message")
        if message.get("role") != "assistant" or message.get("status") != "completed":
            raise ValueError("expected a completed OpenAI assistant message")
        block = _one_output(message.get("content"), ignore_types=(), output_type="output_text")
    elif provider == "anthropic":
        if (response.get("type") != "message" or response.get("role") != "assistant"
                or response.get("stop_reason") != "end_turn"):
            raise ValueError("Anthropic message did not end normally")
        block = _one_output(
            response.get("content"), ignore_types=("thinking", "redacted_thinking"), output_type="text",
        )
    else:
        raise ValueError("provider must be openai or anthropic")
    return parse_accuracy(block.get("text"))


def _counts(*values):
    if any(type(value) is not int or value < 0 or value > 1_000_000_000 for value in values):
        raise ValueError("provider usage counts are missing or invalid")


def normalize_usage(provider, usage):
    """Count billable categories once and reject malformed or incomplete usage."""
    if not isinstance(usage, dict):
        raise ValueError("provider response did not include usage")
    input_tokens, output = usage.get("input_tokens"), usage.get("output_tokens")
    _counts(input_tokens, output)
    if provider == "openai":
        details = usage.get("input_tokens_details")
        if not isinstance(details, dict):
            raise ValueError("OpenAI response lacked input token details")
        read, write = details.get("cached_tokens"), details.get("cache_write_tokens", 0)
        _counts(read, write, usage.get("total_tokens"))
        if read + write > input_tokens or usage.get("total_tokens") != input_tokens + output:
            raise ValueError("contradictory OpenAI token counts")
        ordinary, long_write = input_tokens - read - write, 0
    elif provider == "anthropic":
        read, written = usage.get("cache_read_input_tokens"), usage.get("cache_creation_input_tokens")
        _counts(read, written)
        write_type = usage.get("cache_creation")
        if written and not isinstance(write_type, dict):
            raise ValueError("Anthropic cache write duration is unknown")
        write_type = write_type if isinstance(write_type, dict) else {}
        write = write_type.get("ephemeral_5m_input_tokens", 0)
        long_write = write_type.get("ephemeral_1h_input_tokens", 0)
        _counts(write, long_write)
        if write + long_write != written:
            raise ValueError("contradictory Anthropic cache token counts")
        ordinary = input_tokens
    else:
        raise ValueError("provider must be openai or anthropic")
    return {"input": ordinary, "cache_read": read, "cache_write": write, "cache_write_long": long_write,
            "output": output, "input_total": ordinary + read + write + long_write}


_AUTH_HEADER = re.compile(r"(?im)^\s*(?:authorization|x-api-key)\s*:[^\r\n]*")
_AUTH_FIELD = re.compile(
    r'(?i)"(?:authorization|x-api-key)"\s*:\s*(?:"(?:[^"\\]|\\.)*(?:"|$)|[^,}\r\n]*)',
)


def _credential_pattern(key):
    # Providers can echo a credential within JSON strings, including nested or partial JSON.
    def encoded_character(character):
        options = [re.escape(character), r"\\+u" + f"{ord(character):04x}"]
        if character in '"\\/':
            options.append(r"\\+" + re.escape(character))
        return "(?:" + "|".join(options) + ")"

    return re.compile("".join(map(encoded_character, key)), re.IGNORECASE | re.ASCII)


class CredentialFilter:
    """Remove known API key values before recording provider-controlled data."""

    def __init__(self, keys):
        self._patterns = tuple(_credential_pattern(key) for key in sorted(set(keys), key=len, reverse=True))

    def text(self, value):
        for pattern in self._patterns:
            value = pattern.sub("[redacted credential]", value)
        value = _AUTH_HEADER.sub("[redacted header]", value)
        return _AUTH_FIELD.sub('"redacted_header":"[redacted header]"', value)

    def clean(self, value):
        if isinstance(value, str):
            return self.text(value).encode("utf-8", errors="backslashreplace").decode("utf-8")
        if isinstance(value, list):
            return [self.clean(item) for item in value]
        if isinstance(value, dict):
            return {
                self.clean(key): self.clean(item)
                for key, item in value.items() if key.casefold() not in ("authorization", "x-api-key")
            }
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def response(self, raw):
        decoded = raw.decode("utf-8", errors="replace")
        safe_text = self.text(decoded)
        try:
            body = loads(decoded)
        except (ValueError, RecursionError):
            return safe_text, None
        try:
            cleaned = self.clean(body)
        except RecursionError:
            return safe_text, None
        if cleaned != body:
            safe_text = canonical_json(cleaned).decode("utf-8")
        return safe_text, cleaned
