"""Score accuracy and estimate representation sensitivity at the network level."""

from collections import Counter, defaultdict
from decimal import Decimal
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path
from statistics import mean

import numpy as np

from ..artifacts import json_digest
from .data import finite_range, integer, loads
from .prompts import CONDITIONS, EQUIVALENT
from .providers import parse_response


MIN_INTERVAL_NETWORKS = 10


def kendall_tau_b(labels, predictions):
    """Return Kendall's tau-b, or None when the denominator vanishes."""
    labels, predictions = np.asarray(labels, dtype=float), np.asarray(predictions, dtype=float)
    if (labels.ndim != 1 or labels.shape != predictions.shape
            or not np.isfinite(labels).all() or not np.isfinite(predictions).all()):
        raise ValueError("Kendall tau-b requires finite matched one-dimensional arrays")
    numerator = usable_label_pairs = usable_prediction_pairs = 0
    for i in range(len(labels) - 1):
        ld = np.sign(labels[i] - labels[i + 1:])
        pd = np.sign(predictions[i] - predictions[i + 1:])
        numerator += int(np.dot(ld, pd))
        usable_label_pairs += int(np.count_nonzero(ld))
        usable_prediction_pairs += int(np.count_nonzero(pd))
    if usable_label_pairs and usable_prediction_pairs:
        return numerator / (usable_label_pairs * usable_prediction_pairs) ** 0.5
    return None


def accuracy_metrics(labels, predictions):
    return {
        "n_networks": len(labels), "kendall_tau_b": kendall_tau_b(labels, predictions),
        "mae": mean(abs(label - pred) for label, pred in zip(labels, predictions, strict=True)) if labels else None,
    }


def _read_journal(directory):
    manifest = loads((directory / "manifest.json").read_bytes())
    metadata = manifest["metadata"]
    if (manifest["schema_version"] != 1 or metadata.get("protocol") != "cnn_accuracy_reading"
            or metadata.get("schema_version") != 1 or manifest["metadata_sha256"] != json_digest(metadata)
            or metadata["config_sha256"] != json_digest(metadata["config"])):
        raise ValueError("reader journal manifest or digest is invalid")
    requests = metadata["requests"]
    if len(requests) != metadata["planned_calls"] or len({call["call_id"] for call in requests}) != len(requests):
        raise ValueError("reader journal has inconsistent planned call IDs")
    for call in requests:
        if json_digest({key: value for key, value in call.items() if key != "call_id"}) != call["call_id"]:
            raise ValueError("reader call ID disagrees with its planned request")

    started, finished = {}, {}
    active = None
    terminal = None
    with (directory / "events.jsonl").open(encoding="utf-8") as handle:
        for sequence, line in enumerate(handle):
            event = loads(line)
            if event["sequence"] != sequence or event["run_id"] != manifest["run_id"] or terminal is not None:
                raise ValueError("reader journal events are out of order")
            match event["event"]:
                case "call_started":
                    if active is not None or len(started) >= len(requests):
                        raise ValueError("unexpected call_started event")
                    call = requests[len(started)]
                    if any(event.get(key) != call[key] for key in ("call_id", "request_sha256", "reserved_usd")):
                        raise ValueError("attempted call does not match the ordered plan")
                    active = call
                    started[call["call_id"]] = event
                case "call_finished":
                    if active is None or any(event.get(key) != active[key] for key in ("call_id", "request_sha256")):
                        raise ValueError("finished call does not match an active attempt")
                    raw_response = event["raw_response"]
                    digest = sha256(raw_response.encode("utf-8")).hexdigest() if raw_response is not None else None
                    if event.get("response_sha256") != digest or event.get("wire_response_sha256") not in (None, digest):
                        raise ValueError("reader stored response digest disagrees with the raw response")
                    if event["status"] == "completed":
                        if (event.get("response_truncated") or not 200 <= event["http_status"] < 300
                                or parse_response(active["provider"], loads(event["raw_response"]),
                                                  active["requested_model"]) != event["prediction"]):
                            raise ValueError("recorded prediction disagrees with the successful response")
                    elif event["status"] != "failed" or event["prediction"] is not None:
                        raise ValueError("invalid call outcome")
                    finished[active["call_id"]] = event
                    active = None
                case "run_finished":
                    if active is not None:
                        raise ValueError("terminal event followed an unfinished attempt")
                    terminal = event
                case _:
                    raise ValueError("unknown reader journal event")
    return metadata, started, finished, terminal


def _network_intervals(values, *, replicates, seed, fixtures):
    if not values:
        return {"n_networks": 0, "estimate": None, "ci95": None}
    result = {"n_networks": len(values), "estimate": mean(values), "ci95": None}
    if len(values) < MIN_INTERVAL_NETWORKS or fixtures:
        return result
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for i in range(0, replicates, 256):
        draws = rng.integers(len(array), size=(min(256, replicates - i), len(array)))
        bootstrap.extend(array[draws].mean(axis=1))
    result["ci95"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return result


def _agreement(pairs, tolerance):
    limit = Decimal(str(tolerance))
    differences = [abs(Decimal(str(left)) - Decimal(str(right))) for left, right in pairs]
    return {
        "agreement": mean(difference <= limit for difference in differences),
        "absolute_difference": float(mean(differences)),
    }


def _table(metadata, started, finished):
    table = defaultdict(dict)
    availability = []
    for call in metadata["requests"]:
        call_id = call["call_id"]
        event = finished.get(call_id)
        if event is None:
            status = "interrupted" if call_id in started else "unattempted"
            error = None
        else:
            status, error = event["status"], event["error"]
        if status == "completed":
            cell = tuple(call[key] for key in ("provider", "requested_model", "k", "network_sha256", "condition"))
            table[cell][call["repetition"]] = event["prediction"]
        availability.append({
            **{key: call[key] for key in (
                "call_id", "provider", "requested_model", "k", "network_sha256", "condition", "repetition",
            )},
            "status": status, "error": error,
            "usage_error": event.get("usage_error") if event is not None else None,
            "usage_measured": event["quoted_usage_usd"] is not None if event is not None else None,
        })
    return table, availability


def _complete(table, key, repeats):
    values = table.get(key, {})
    if set(values) == set(range(repeats)):
        return [values[i] for i in range(repeats)]
    return None


def _condition_group(rows, table, key, repeats, tolerance, intervals):
    predictions = {
        row["network_sha256"]: {
            condition: _complete(table, (*key, row["network_sha256"], condition), repeats)
            for condition in CONDITIONS
        }
        for row in rows
    }
    scores = {}
    for condition in ("original", *EQUIVALENT):
        available = [(row["network_sha256"], row["accuracy"], mean(values)) for row in rows
                     if (values := predictions[row["network_sha256"]][condition]) is not None]
        labels, estimates = [item[1] for item in available], [item[2] for item in available]
        scores[condition] = accuracy_metrics(labels, estimates) | {"network_sha256": [item[0] for item in available]}

    repeat = {}
    paired = {condition: {} for condition in (*EQUIVALENT, "all_equivalent", "shuffle")}
    for network, conditions in predictions.items():
        original = conditions["original"]
        if original is None:
            continue
        baseline = _agreement(combinations(original, 2), tolerance)
        repeat[network] = baseline
        for condition, values in conditions.items():
            if condition != "original" and values is not None:
                paired[condition][network] = _agreement(product(original, values), tolerance)
        if all(network in paired[condition] for condition in EQUIVALENT):
            paired["all_equivalent"][network] = {
                metric: mean(paired[condition][network][metric] for condition in EQUIVALENT)
                for metric in ("agreement", "absolute_difference")
            }

    def compare(net_scores):
        ids = list(net_scores)
        return {
            "paired_network_sha256": ids,
            "original_repeat_agreement": mean(repeat[network]["agreement"] for network in ids) if ids else None,
            "condition_agreement": mean(net_scores[network]["agreement"] for network in ids) if ids else None,
            "agreement_drop": intervals([
                repeat[network]["agreement"] - net_scores[network]["agreement"] for network in ids
            ]),
            "original_repeat_mean_absolute_difference": (
                mean(repeat[network]["absolute_difference"] for network in ids) if ids else None
            ),
            "condition_mean_absolute_difference": (
                mean(net_scores[network]["absolute_difference"] for network in ids) if ids else None
            ),
        }

    symmetry = {condition: compare(paired[condition]) for condition in (*EQUIVALENT, "all_equivalent")}
    for result in symmetry.values():
        result["symmetry_gap"] = result.pop("agreement_drop")
    return {
        "planned_networks": len(rows),
        "accuracy": scores,
        "original_repeat_agreement": intervals([item["agreement"] for item in repeat.values()]),
        "symmetry": symmetry,
        "shuffle_response_control": compare(paired["shuffle"]),
    }


def summarize(directory, *, bootstrap_replicates=2000, bootstrap_seed=20261119):
    integer(bootstrap_replicates, 1, 100_000, "bootstrap_replicates")
    integer(bootstrap_seed, 0, 2**32 - 1, "bootstrap_seed")
    metadata, started, finished, terminal = _read_journal(Path(directory))
    config = metadata["config"]
    tolerance = finite_range(config["agreement_tolerance"], 0, 1, "agreement tolerance")
    rows = [row for row in metadata["panel"]["networks"] if row["split"] == "test"]
    table, availability = _table(metadata, started, finished)
    counts = dict.fromkeys(("completed", "failed", "interrupted", "unattempted"), 0)
    counts.update(Counter(item["status"] for item in availability))
    attempted_reservation = sum((Decimal(event["reserved_usd"]) for event in started.values()), Decimal(0))
    quoted = [Decimal(event["quoted_usage_usd"]) for event in finished.values()
              if event["quoted_usage_usd"] is not None]
    known_quote = sum(quoted, Decimal(0))
    billing = {
        "attempted_reservation_usd": str(attempted_reservation), "known_quoted_usage_usd": str(known_quote),
        "unknown_usage_attempts": len(started) - len(quoted), "actual_charge_usd": None,
    }
    if terminal is not None:
        if (
            any(terminal[key] != counts[key] for key in ("completed", "failed", "unattempted"))
            or terminal["attempted"] != len(started) or terminal["unknown_usage"] != billing["unknown_usage_attempts"]
            or Decimal(terminal["attempted_reservation_usd"]) != attempted_reservation
            or Decimal(terminal["known_quoted_usage_usd"]) != known_quote
        ):
            raise ValueError("reader journal terminal totals disagree with its recorded attempts")
    fixtures = metadata["panel"]["dataset"]["sampling"] == "fixtures"

    def intervals(values):
        return _network_intervals(values, replicates=bootstrap_replicates, seed=bootstrap_seed, fixtures=fixtures)

    groups = []
    for reader in config["readers"]:
        for k in sorted(config["in_context"]):
            key = reader["provider"], reader["model"], k
            for activation in ("all", "relu", "tanh"):
                subset = [row for row in rows if activation == "all" or row["activation"] == activation]
                if subset:
                    groups.append({
                        "provider": key[0], "requested_model": key[1], "k": k, "activation": activation,
                        **_condition_group(subset, table, key, config["repeats"], tolerance, intervals),
                    })
    return {
        "schema_version": 1, "config_sha256": metadata["config_sha256"],
        "panel_sha256": metadata["panel"]["manifest_sha256"], "sample": metadata["panel"]["dataset"]["sampling"],
        "counts": counts, "billing": billing, "terminal_recorded": terminal is not None,
        "stop_reason": terminal["stop_reason"] if terminal is not None else None,
        "availability": availability, "agreement_tolerance": tolerance, "groups": groups,
        "bootstrap": {
            "replicates": bootstrap_replicates, "seed": bootstrap_seed, "unit": "source training network",
            "minimum_networks_for_ci95": MIN_INTERVAL_NETWORKS, "method": "network-cluster percentile",
        },
        "interpretation": (
            "Estimates describe the explicit sampled panel and completed repeats. "
            "Intervals assume independent source training networks and do not establish generality. "
            "Intervals are omitted for fixtures and groups with fewer than 10 paired networks. "
            "Shuffle accuracy is unknown and excluded from accuracy scoring."
        ),
    }
