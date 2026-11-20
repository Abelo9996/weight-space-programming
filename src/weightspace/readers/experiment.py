"""Preflight and journal bounded CNN reader comparisons."""

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import math
import os
from pathlib import Path
import platform
from random import Random
import re

import numpy as np
import torch

from ..artifacts import RunJournal, canonical_json, json_digest
from ..experiment import _source_provenance
from .data import fields, finite_range, integer, load_panel, loads
from .pricing import RateCard, dollars
from .prompts import CONDITIONS, PROMPT_VERSION, example_prefix, variants_for
from .providers import (
    ENDPOINTS, CredentialFilter, normalize_usage, parse_response, request_body, send_request,
)


JOB_APPROVAL_THRESHOLD_USD = Decimal(40)


def _environment_keys():
    return {provider: value for provider, endpoint in ENDPOINTS.items()
            if (value := os.environ.get(endpoint.key_variable))}


def reject_embedded_credentials(value, keys=None):
    """Prevent offline output or a live request from recording an environment key."""
    values = _environment_keys().values() if keys is None else keys
    encoded = value if isinstance(value, bytes) else canonical_json(value)
    if any(key.encode("utf-8") in encoded for key in values):
        raise ValueError("an API credential appears in planned data")


@dataclass(frozen=True)
class Reader:
    provider: str
    model: str
    effort: str
    context_window: int
    max_output_tokens: int
    input_tokens_per_byte: float
    framing_tokens: int
    rates: RateCard

    @classmethod
    def read(cls, raw):
        fields(raw, ("provider", "model", "effort", "context_window", "max_output_tokens",
                     "input_tokens_per_byte", "framing_tokens", "price"))
        provider, model, effort = (raw[key] for key in ("provider", "model", "effort"))
        if provider not in tuple(ENDPOINTS):
            raise ValueError("provider must be openai or anthropic")
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", model):
            raise ValueError("invalid provider model ID")
        if effort not in ("low", "medium", "high"):
            raise ValueError("effort must be low, medium or high")
        window = integer(raw["context_window"], 64, 10_000_000, "context_window")
        output = integer(raw["max_output_tokens"], 16, min(window - 1, 128_000), "max_output_tokens")
        factor = finite_range(raw["input_tokens_per_byte"], 0.01, 2, "input_tokens_per_byte")
        framing = integer(raw["framing_tokens"], 1, window, "framing_tokens")
        return cls(provider, model, effort, window, output, factor, framing, RateCard.read(raw["price"], window))

    @property
    def key(self):
        return self.provider, self.model

    def body(self, prompt):
        return request_body(*self.key, prompt, self.max_output_tokens, self.effort)

    def token_estimate(self, size):
        return math.ceil(size * self.input_tokens_per_byte) + self.framing_tokens


@dataclass(frozen=True)
class ReaderConfig:
    raw: dict
    file_sha256: str
    seed: int
    repeats: int
    in_context: tuple[int, ...]
    tolerance: float
    budget: Decimal
    credits: dict
    readers: tuple[Reader, ...]
    timeout: int


def load_config(path):
    raw_bytes = Path(path).read_bytes()
    raw = fields(loads(raw_bytes), (
        "schema_version", "seed", "repeats", "in_context", "agreement_tolerance", "budget_usd", "readers",
    ), ("confirmed_credits_usd", "timeout_seconds"))
    integer(raw["schema_version"], 1, 1, "config schema_version")
    seed = integer(raw["seed"], 0, 2**32 - 1, "seed")
    repeats = integer(raw["repeats"], 2, 100, "repeats")
    ks = raw["in_context"]
    if (not isinstance(ks, list) or not ks or any(type(k) is not int or k not in (0, 8, 32) for k in ks)
            or len(set(ks)) != len(ks)):
        raise ValueError("in_context must be a nonempty list of unique values from 0, 8 and 32")
    tolerance = finite_range(raw["agreement_tolerance"], 0, 1, "agreement_tolerance")
    budget = dollars(raw["budget_usd"], "budget_usd")
    if not isinstance(raw["readers"], list) or not raw["readers"]:
        raise ValueError("readers must be a nonempty list")
    readers = tuple(Reader.read(record) for record in raw["readers"])
    if len({reader.key for reader in readers}) != len(readers):
        raise ValueError("reader provider and model pairs must be unique")
    credits = fields(raw.get("confirmed_credits_usd", {}), (), {reader.provider for reader in readers})
    credits = {name: dollars(amount, "confirmed credits") for name, amount in credits.items()}
    timeout = integer(raw.get("timeout_seconds", 180), 1, 3600, "timeout_seconds")
    return ReaderConfig(raw, sha256(raw_bytes).hexdigest(), seed, repeats, tuple(sorted(ks)),
                        tolerance, budget, credits, readers, timeout)


@dataclass(frozen=True)
class PreparedPlan:
    metadata: dict
    config: ReaderConfig
    variants: dict
    prefixes: dict

    def prompt(self, call):
        return self.prefixes[call["k"]] + self.variants[call["network_sha256"]][call["condition"]].text


def _funding_blockers(config, totals):
    blockers = set()
    if config.budget <= 0 or config.budget < sum(totals.values()):
        blockers.add("budget_usd must be positive and cover the estimated reservation for every call")
    for provider, total in totals.items():
        if config.credits.get(provider, Decimal(0)) < total:
            blockers.add(f"{provider}: confirmed_credits_usd must cover that provider's projected reservation")
    return blockers


def prepare_plan(config, panel_path):
    """Build every request offline. The metadata omits prompt text and local paths."""
    panel = load_panel(panel_path)
    train = sorted(panel.split("train"), key=lambda row: json_digest([config.seed, row.network_sha256]))
    if len(train) < max(config.in_context):
        raise ValueError(f"k={max(config.in_context)} requires that many distinct training networks, found {len(train)}")
    prefixes = {k: example_prefix(train[:k], panel.architecture) for k in config.in_context}
    variants = {}
    tensor_identities = {row.tensors_sha256: row.network_sha256 for row in panel.checkpoints}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for row in panel.split("test"):
            row_variants = variants_for(row, panel.architecture, config.seed)
            for variant in row_variants.values():
                digest = variant.provenance["tensors_sha256"]
                previous = tensor_identities.setdefault(digest, row.network_sha256)
                if previous != row.network_sha256:
                    raise ValueError("generated variant overlaps a different source training run")
            variants[row.network_sha256] = row_variants
    finally:
        torch.set_num_threads(previous_threads)

    requests = []
    totals = dict.fromkeys((reader.provider for reader in config.readers), Decimal(0))
    blockers = set()
    for row in panel.split("test"):
        for k, prefix in prefixes.items():
            for condition, variant in variants[row.network_sha256].items():
                prompt = prefix + variant.text
                prompt_bytes = prompt.encode("utf-8")
                prompt_hash = sha256(prompt_bytes).hexdigest()
                for reader in config.readers:
                    reserved_input = reader.token_estimate(len(prompt_bytes))
                    if reserved_input + reader.max_output_tokens > reader.context_window:
                        blockers.add(f"{reader.provider}/{reader.model}: k={k} exceeds the quoted context window")
                    reserved = reader.rates.reservation(reserved_input, reader.max_output_tokens)
                    totals[reader.provider] += config.repeats * reserved
                    endpoint = ENDPOINTS[reader.provider]
                    details = {
                        "provider": reader.provider, "requested_model": reader.model, "k": k,
                        "network_sha256": row.network_sha256, "condition": condition,
                        "prompt_sha256": prompt_hash, "prompt_bytes": len(prompt_bytes),
                        "request_sha256": sha256(reader.body(prompt)).hexdigest(),
                        "endpoint": endpoint.url, "api_version": endpoint.version,
                        "tensors_sha256": variant.provenance["tensors_sha256"],
                        "transform_sha256": variant.provenance["transform_sha256"],
                        "input_tokens_reserved_estimate": reserved_input,
                        "max_output_tokens": reader.max_output_tokens,
                        "reserved_usd": str(reserved),
                    }
                    for repetition in range(config.repeats):
                        identity = {**details, "repetition": repetition}
                        requests.append({**identity, "call_id": json_digest(identity)})
    Random(config.seed).shuffle(requests)
    projected = sum(totals.values())
    blockers.update(_funding_blockers(config, totals))

    metadata = {
        "schema_version": 1, "protocol": "cnn_accuracy_reading", "prompt_version": PROMPT_VERSION,
        "config": config.raw, "config_sha256": json_digest(config.raw), "config_file_sha256": config.file_sha256,
        "panel": panel.provenance(), "package": _source_provenance(),
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
            "platform": platform.platform(),
        },
        "examples": {str(k): [row.network_sha256 for row in train[:k]] for k in prefixes},
        "conditions": list(CONDITIONS),
        "variants": {network: {condition: variant.provenance for condition, variant in items.items()}
                     for network, items in variants.items()},
        "requests": requests, "planned_calls": len(requests), "live_preflight_blockers": sorted(blockers),
        "projected_reservation_usd": str(projected),
        "approval_threshold_usd": str(JOB_APPROVAL_THRESHOLD_USD),
        "requires_over_threshold_approval": projected > JOB_APPROVAL_THRESHOLD_USD,
        "projected_reservation_by_provider_usd": {key: str(value) for key, value in totals.items()},
        "estimate_notes": (
            "Input token reservation is a configured bytes-to-tokens estimate plus framing allowance. "
            "Output token limits are sent to the provider and include billed reasoning where applicable. "
            "Quotes use the maximum applicable input-category list price. Provider invoices may differ "
            "from these estimates. Reservations are never refunded within a run."
        ),
    }
    reject_embedded_credentials(metadata)
    return PreparedPlan(metadata, config, variants, prefixes)


def _load_keys(readers):
    keys = _environment_keys()
    for reader in readers:
        variable = ENDPOINTS[reader.provider].key_variable
        if reader.provider not in keys:
            raise ValueError(f"missing {variable}")
        key = keys[reader.provider]
        if (key.startswith(("sk-ant-oat", "sess-"))
                or re.fullmatch(r"eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", key)):
            raise ValueError(f"{variable} must contain a first-party API key")
        if any(not 33 <= ord(character) <= 126 for character in key):
            raise ValueError(f"{variable} must contain a printable API key without whitespace")
    return keys


def _service_drift(provider, response, usage):
    if provider == "openai":
        return response.get("service_tier") not in (None, "default")
    return (usage.get("service_tier") not in (None, "standard")
            or usage.get("inference_geo") not in (None, "global")
            or usage.get("speed") not in (None, "standard"))


def _tool_telemetry_present(telemetry):
    if isinstance(telemetry, dict):
        return any(_tool_telemetry_present(value) for value in telemetry.values())
    if isinstance(telemetry, list):
        return any(_tool_telemetry_present(value) for value in telemetry)
    return bool(telemetry)


def _provider_observation(reader, call, wire, credentials):
    response, parsed = credentials.response(wire.body)
    stored_bytes = response.encode("utf-8")
    record = {
        "http_status": wire.status, "provider_request_id": credentials.clean(wire.request_id),
        "raw_response": response, "response_truncated": wire.truncated,
        "response_sha256": sha256(stored_bytes).hexdigest(),
        "wire_response_sha256": sha256(wire.body).hexdigest() if stored_bytes == wire.body else None,
        "wire_response_bytes_received": len(wire.body),
        "raw_response_changed": stored_bytes != wire.body,
        "reported_model": None, "provider_response_id": None, "usage": None, "usage_counts": None,
        "usage_error": None, "quoted_usage_usd": None, "actual_charge_usd": None, "prediction": None,
    }
    if isinstance(parsed, dict):
        record.update({
            "reported_model": parsed.get("model"), "provider_response_id": parsed.get("id"),
            "usage": parsed.get("usage"),
        })
    stop = None
    if not isinstance(parsed, dict):
        record["usage_error"] = "provider response did not include usage"
    else:
        raw_usage = parsed.get("usage")
        try:
            counts = normalize_usage(reader.provider, raw_usage)
        except ValueError as error:
            record["usage_error"] = str(error)
        else:
            record["usage_counts"] = counts
            record["quoted_usage_usd"] = str(reader.rates.estimate(counts))
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        counts = record["usage_counts"] or {"input_total": usage.get("input_tokens"), "output": usage.get("output_tokens")}
        limits = {"input_total": call["input_tokens_reserved_estimate"], "output": call["max_output_tokens"]}
        if any(type(counts[name]) is int and counts[name] > limit for name, limit in limits.items()):
            stop = "reported usage exceeded the per-call token reservation"
        drift = None
        if parsed.get("model") not in (None, reader.model):
            drift = "provider reported a different model"
        elif _service_drift(reader.provider, parsed, usage):
            drift = "provider reported a different service tier, speed or region"
        elif any(_tool_telemetry_present(value) for value in (
            parsed.get("tool_usage"), usage.get("tool_usage"), usage.get("server_tool_use"),
        )):
            drift = "provider reported tool usage for a tool-free request"
        if drift is not None:
            return record | {"status": "failed", "error": drift, "usage_error": drift, "quoted_usage_usd": None}, drift
    if wire.status in (401, 402, 403, 429):
        stop = "provider rejected credentials, access, billing or rate limits"
    if wire.truncated:
        return record | {"status": "failed", "error": "provider response was truncated"}, stop
    if not 200 <= wire.status < 300:
        return record | {"status": "failed", "error": "provider returned an HTTP error"}, stop
    try:
        prediction = parse_response(reader.provider, parsed, reader.model)
    except (TypeError, ValueError) as error:
        return record | {"status": "failed", "error": str(error)}, stop
    return record | {"status": "completed", "error": None, "prediction": prediction}, stop


def run_plan(plan, output, *, allow_api=False, confirm_credits=False, approve_over_40=False, transport=None):
    """Execute exactly one attempt per scheduled cell, with an append-only journal."""
    if allow_api is not True or confirm_credits is not True:
        raise ValueError("reading run requires --allow-api and --confirm-credits")
    if plan.metadata["live_preflight_blockers"]:
        raise ValueError("reading plan is blocked. " + ". ".join(plan.metadata["live_preflight_blockers"]))
    if plan.metadata["requires_over_threshold_approval"] and approve_over_40 is not True:
        raise ValueError("job estimates above $40 require explicit approval with --approve-over-40")
    keys = _load_keys(plan.config.readers)
    private = CredentialFilter(keys.values())
    reject_embedded_credentials(plan.metadata, keys.values())
    readers = {reader.key: reader for reader in plan.config.readers}

    def request_bytes(call):
        prompt = plan.prompt(call)
        reader = readers[call["provider"], call["requested_model"]]
        return reader.body(prompt)

    # Check the request bodies before creating the journal or spending anything.
    checked = {}
    totals = dict.fromkeys((reader.provider for reader in plan.config.readers), Decimal(0))
    for call in plan.metadata["requests"]:
        digest = call["request_sha256"]
        reader = readers[call["provider"], call["requested_model"]]
        if digest not in checked:
            prompt = plan.prompt(call)
            body = reader.body(prompt)
            if sha256(body).hexdigest() != call["request_sha256"]:
                raise ValueError("planned request hash disagrees with its payload")
            reject_embedded_credentials(body, keys.values())
            checked[digest] = len(prompt.encode("utf-8"))
        input_tokens = reader.token_estimate(checked[digest])
        cap = reader.max_output_tokens
        reserve = reader.rates.reservation(input_tokens, cap)
        if (input_tokens != call["input_tokens_reserved_estimate"] or cap != call["max_output_tokens"]
                or reserve != Decimal(call["reserved_usd"]) or input_tokens + cap > reader.context_window):
            raise ValueError("planned reservations disagree with the configured request limits")
        totals[reader.provider] += reserve
    funding_blockers = _funding_blockers(plan.config, totals)
    if funding_blockers:
        raise ValueError("reading plan is blocked. " + ". ".join(sorted(funding_blockers)))
    if sum(totals.values()) > JOB_APPROVAL_THRESHOLD_USD and approve_over_40 is not True:
        raise ValueError("job estimates above $40 require explicit approval with --approve-over-40")
    output = Path(output)
    journal = RunJournal.create(
        output.parent, plan.metadata | {"approved_over_threshold": approve_over_40 is True}, output.name,
    )
    transport = send_request if transport is None else transport
    totals = {"attempted": 0, "completed": 0, "failed": 0, "unknown_usage": 0}
    charged_reservation, quoted_estimate = Decimal(0), Decimal(0)
    stop = None
    for call in plan.metadata["requests"]:
        provider, model = call["provider"], call["requested_model"]
        endpoint = ENDPOINTS[provider]
        attempt = {"call_id": call["call_id"], "request_sha256": call["request_sha256"]}
        journal.append("call_started", **attempt, reserved_usd=call["reserved_usd"])
        charged_reservation += Decimal(call["reserved_usd"])
        totals["attempted"] += 1
        try:
            wire = transport(endpoint, request_bytes(call), endpoint.headers(keys[provider]), plan.config.timeout)
        except Exception as error:
            result = {
                "status": "failed", "error": private.clean(str(error)), "error_type": private.clean(type(error).__name__),
                "raw_response": None, "http_status": None, "usage": None, "usage_counts": None,
                "response_sha256": None, "wire_response_sha256": None, "wire_response_bytes_received": 0,
                "quoted_usage_usd": None, "actual_charge_usd": None, "prediction": None,
            }
        else:
            result, stop = _provider_observation(readers[provider, model], call, wire, private)
        totals[result["status"]] += 1
        if result["quoted_usage_usd"] is None:
            totals["unknown_usage"] += 1
        else:
            quoted_estimate += Decimal(result["quoted_usage_usd"])
        journal.append("call_finished", **attempt, **result)
        if stop is not None:
            break
    report = {
        **totals, "unattempted": plan.metadata["planned_calls"] - totals["attempted"],
        "attempted_reservation_usd": str(charged_reservation), "known_quoted_usage_usd": str(quoted_estimate),
        "actual_charge_usd": None, "stop_reason": stop, "run_directory": str(output),
    }
    journal.append("run_finished", **{key: value for key, value in report.items() if key != "run_directory"})
    return report
