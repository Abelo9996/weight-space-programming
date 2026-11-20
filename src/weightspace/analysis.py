"""Summarize finite-panel policy outcomes and paired-seed uncertainty."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from .artifacts import json_digest
from .codex_editor import MODEL_EVIDENCE, catalog_provenance
from .edits import EditLimits, apply_edits
from .experiment import EDITOR_IMPLEMENTATIONS, ExperimentConfig, planned_cells
from .policies import load_policy, policy_digest
from .search import EDITOR_METHODS, SEARCH_METHODS


METRICS = ("objective", "failure_success", "normal_success")
COMPARISON_FIELDS = ("dev_sha256", "heldout_sha256", "evaluator_source")
BUDGET_FIELDS = ("candidate_cap", "row_budget", "rows_per_candidate")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _one(events, kind):
    matches = [event for event in events if event["event"] == kind]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {kind} event")
    return matches[0]


def _read_run(input_dir, cell, config_hash, limits, editor_options, editor_catalog):
    directory = input_dir / "runs" / cell["run_id"]
    if not directory.is_dir():
        return "missing", "planned run directory is absent", None
    try:
        manifest = _read_json(directory / "manifest.json")
        if (
            not isinstance(manifest, dict) or not isinstance(manifest.get("metadata"), dict)
            or manifest.get("run_id") != cell["run_id"]
            or manifest.get("metadata_sha256") != json_digest(manifest["metadata"])
            or manifest["metadata"].get("config_sha256") != config_hash
            or any(manifest["metadata"].get(key) != value for key, value in cell.items())
        ):
            raise ValueError("manifest does not match the planned run")
        with (directory / "events.jsonl").open(encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle]
        if any(
            not isinstance(event, dict) or not isinstance(event.get("event"), str)
            or event["sequence"] != i or event["run_id"] != cell["run_id"]
            for i, event in enumerate(events)
        ):
            raise ValueError("event sequence or run ID does not match")
        if events and events[-1]["event"] == "run_failed":
            return "failed", events[-1]["reason"], None
        if not events or events[-1]["event"] != "run_completed":
            return "incomplete", "run has no terminal completion event", None
        completed = _one(events, "run_completed")
        checkpoint = _one(events, "checkpoint")
        selected = _one(events, "final_selected")
        heldout = _one(events, "heldout_result")
        if not checkpoint["sequence"] < selected["sequence"] < heldout["sequence"] < completed["sequence"]:
            raise ValueError("checkpoint, selection, and held-out evaluation are out of order")
        if (
            not all(completed[field] == heldout[field] for field in (
                "candidate_id", "ops", "dev", "heldout", "budget", "heldout_rows_reserved", "heldout_rows_used",
            ))
            or completed["selected_policy_sha256"] != heldout["policy_sha256"]
            or completed["starting_policy_sha256"] != checkpoint["starting_policy_sha256"]
            or selected["policy_sha256"] != heldout["policy_sha256"]
            or selected["heldout_sha256"] != checkpoint["provenance"]["heldout_sha256"]
        ):
            raise ValueError("completion does not match the evaluator records")
        provenance = checkpoint["provenance"]
        if any(not isinstance(provenance.get(field), str) for field in (*COMPARISON_FIELDS, "policy_sha256")):
            raise ValueError("checkpoint lacks split or evaluator provenance")
        if any(
            type(completed["budget"].get(field)) is not int or completed["budget"][field] < 0
            for field in (*BUDGET_FIELDS, "candidates_used", "rows_reserved", "rows_used")
        ):
            raise ValueError("completion lacks valid budget accounting")
        starting_policy = load_policy(directory / "starting_policy.npz")
        selected_policy = load_policy(directory / "selected_policy.npz")
        if (
            policy_digest(starting_policy) != completed["starting_policy_sha256"]
            or policy_digest(selected_policy) != completed["selected_policy_sha256"]
            or policy_digest(apply_edits(starting_policy, completed["ops"], limits).policy)
            != completed["selected_policy_sha256"]
        ):
            raise ValueError("checkpoint digest or replayed selection does not match")
        for source in ("dev", "heldout"):
            for field in METRICS:
                if type(completed[source][field]) not in (float, int) or not 0 <= completed[source][field] <= 1:
                    raise ValueError("score outside [0, 1]")
        attempts = defaultdict(int)
        for event in events:
            if event["event"] == "candidate_result":
                if event["sequence"] > selected["sequence"]:
                    raise ValueError("candidate evaluated after selection")
                attempts[event["status"]] += 1
        editor_events = [event for event in events if event["event"] == "editor_turn"]
        if any(event["sequence"] > selected["sequence"] for event in editor_events):
            raise ValueError("editor ran after selection")
        model_identity = None
        method = cell["method"]
        if method in EDITOR_METHODS:
            preflight = _one(events, "editor_preflight")
            model_identity = (
                preflight["access"], preflight["cli_version"], preflight["requested_model"], preflight["effort"],
            )
            editor_type, _ = EDITOR_IMPLEMENTATIONS[method]
            if (
                model_identity[0] != editor_type.access or model_identity[2] != editor_options.model
                or model_identity[3] != editor_options.effort
                or not 1 <= len(editor_events) <= editor_options.max_calls
                or any(
                    event["access"] != model_identity[0] or event["cli_version"] != model_identity[1]
                    or event["requested_model"] != model_identity[2] for event in editor_events
                )
            ):
                raise ValueError("editor run has inconsistent provenance")
            if method == "codex":
                if (
                    model_identity[1] != editor_options.cli_version
                    or preflight["model_evidence"] != MODEL_EVIDENCE
                    or preflight.get("model_catalog") != editor_catalog
                    or any(
                        event["reported_models"] != [] or event["model_evidence"] != MODEL_EVIDENCE
                        or event.get("model_catalog") != editor_catalog
                        or event["stream_error"] is not None for event in editor_events
                    )
                ):
                    raise ValueError("Codex run has inconsistent CLI or model provenance")
            elif any(event["reported_models"] != [model_identity[2]] for event in editor_events):
                raise ValueError("Claude run has inconsistent reported models")
        return "completed", None, {
            **completed, "provenance": checkpoint["provenance"], "attempts": dict(attempts),
            "editor_calls": len(editor_events), "editor_identity": model_identity,
        }
    except (AttributeError, OSError, TypeError, ValueError, KeyError) as error:
        return "invalid", str(error), None


def _stats(values):
    if not values:
        return {"n": 0, "mean": None, "sd": None, "minimum": None, "maximum": None}
    return {
        "n": len(values), "mean": mean(values), "sd": stdev(values) if len(values) > 1 else None,
        "minimum": min(values), "maximum": max(values),
    }


def _bootstrap(matrix, statistic, replicates, seed):
    """Resample whole rows, keeping all methods or conditions for a seed together."""
    if not len(matrix):
        return {"n_pairs": 0, "estimate": None, "ci95": None}
    estimate = float(statistic(matrix[None, :, :])[0])
    if len(matrix) < 2:
        return {"n_pairs": len(matrix), "estimate": estimate, "ci95": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(matrix), size=(replicates, len(matrix)))
    boot = statistic(matrix[draws])
    interval = np.quantile(boot, [0.025, 0.975]).tolist()
    return {"n_pairs": len(matrix), "estimate": estimate, "ci95": interval}


def _matched(rows, include_starting):
    def key(row):
        return (
            tuple(row["provenance"][field] for field in COMPARISON_FIELDS),
            tuple(row["budget"][field] for field in BUDGET_FIELDS),
            row["starting_policy_sha256"] if include_starting else row["provenance"]["policy_sha256"],
        )

    if len({key(row) for row in rows}) != 1:
        return False
    return len({row["editor_identity"] for row in rows if row["editor_identity"] is not None}) <= 1


def _compare(table, config, task, cells, *, compare_conditions, best_search, replicates, bootstrap_seed):
    available = []
    seeds = []
    excluded_seeds = []
    for seed in config.seeds:
        rows = [table.get((task, condition, method, seed)) for condition, method in cells]
        if all(row is not None for row in rows) and _matched(rows, not compare_conditions):
            available.append(rows)
            seeds.append(seed)
        else:
            excluded_seeds.append({
                "seed": seed, "reason": "missing or unsuccessful run" if any(row is None for row in rows)
                else "mismatched checkpoint, split, budget, or editor provenance",
            })
    model_identities = {
        row["editor_identity"] for rows in available for row in rows if row["editor_identity"] is not None
    }
    if len(model_identities) > 1:
        excluded_seeds.extend({"seed": seed, "reason": "editor provenance differs across seeds"} for seed in seeds)
        available, seeds = [], []

    def difference(batch):
        averages = batch.mean(axis=1)
        return averages[:, 0] - averages[:, 1:].max(axis=1)

    results = {}
    winners = {}
    for metric in METRICS:
        matrix = np.asarray([[row["heldout"][metric] for row in rows] for rows in available], dtype=float)
        results[metric] = _bootstrap(matrix, difference, replicates, bootstrap_seed)
        if best_search and len(matrix):
            winners[metric] = cells[int(np.argmax(matrix[:, 1:].mean(axis=0))) + 1][1]
    return {
        "cells": [{"condition": condition, "method": method} for condition, method in cells],
        "paired_seeds": seeds, "excluded_seeds": excluded_seeds,
        "metrics": results, **({"best_search_by_metric": winners} if best_search else {}),
    }


def summarize(input_dir, *, bootstrap_replicates=2000, bootstrap_seed=20261119):
    """Read the planned comparison and summarize successful, matched runs."""
    if type(bootstrap_replicates) is not int or not 1 <= bootstrap_replicates <= 100_000:
        raise ValueError("bootstrap replicates must be in [1, 100000]")
    if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed < 2**32:
        raise ValueError("bootstrap seed must be in [0, 2**32)")
    input_dir = Path(input_dir)
    manifest = _read_json(input_dir / "experiment.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
        raise ValueError("experiment manifest must include a config object")
    config_hash = json_digest(manifest["config"])
    config = ExperimentConfig(**manifest["config"])
    cells = planned_cells(config)
    if (
        manifest.get("schema_version") != 1 or manifest.get("config_sha256") != config_hash
        or manifest.get("cells") != cells
    ):
        raise ValueError("experiment manifest does not match its config and planned cells")
    table, availability = {}, []
    counts = dict.fromkeys(("completed", "failed", "incomplete", "missing", "invalid"), 0)
    limits = EditLimits(**config.edit_limits)
    editor_options = config.editor_options() if config.editor_method else None
    editor_catalog = None
    if config.editor_method == "codex":
        editor_catalog = catalog_provenance(editor_options, input_dir)
        launch_bounds = manifest.get("live_editor_launch_bounds")
        if not isinstance(launch_bounds, dict) or launch_bounds.get("model_catalog") != editor_catalog:
            raise ValueError("experiment manifest has inconsistent Codex model catalog provenance")
    for cell in cells:
        status, reason, data = _read_run(input_dir, cell, config_hash, limits, editor_options, editor_catalog)
        counts[status] += 1
        availability.append({**cell, "status": status, "reason": reason})
        if data is not None:
            table[cell["task"], cell["condition"], cell["method"], cell["seed"]] = data
    groups, comparisons = [], []
    for task in config.tasks:
        for condition in config.conditions:
            for method in config.methods:
                runs = {
                    seed: table[task, condition, method, seed]
                    for seed in config.seeds if (task, condition, method, seed) in table
                }
                score_summary = {}
                for metric in METRICS:
                    dev = [row["dev"][metric] for row in runs.values()]
                    heldout = [row["heldout"][metric] for row in runs.values()]
                    paired = np.array(list(zip(dev, heldout)), dtype=float)
                    score_summary[metric] = {
                        "dev": _stats(dev), "heldout": _stats(heldout),
                        "dev_minus_heldout": _stats([before - after for before, after in zip(dev, heldout)]),
                        "dev_minus_heldout_interval": _bootstrap(
                            paired, lambda batch: (batch[:, :, 0] - batch[:, :, 1]).mean(axis=1),
                            bootstrap_replicates, bootstrap_seed,
                        ),
                    }
                groups.append({
                    "task": task, "condition": condition, "method": method, "completed_seeds": list(runs),
                    "planned_seed_count": len(config.seeds), "scores": score_summary,
                    "budget": {key: _stats([row["budget"][key] for row in runs.values()]) for key in (
                        "candidate_cap", "candidates_used", "row_budget", "rows_reserved", "rows_used",
                    )},
                    "editor_calls": _stats([row["editor_calls"] for row in runs.values()]),
                    "per_run": [
                        {"seed": seed, "attempts": row["attempts"], "editor_identity": row["editor_identity"]}
                        for seed, row in runs.items()
                    ],
                })
                targets = []
                if condition != "original":
                    targets.append(("original_minus_transformed", [("original", method), (condition, method)], True, False))
                if method != "no_edit" and "no_edit" in config.methods:
                    targets.append(("method_minus_no_edit", [(condition, method), (condition, "no_edit")], False, False))
                searches = [search for search in SEARCH_METHODS if search in config.methods]
                if method in EDITOR_METHODS and searches:
                    for search in searches:
                        targets.append(("editor_minus_search", [(condition, method), (condition, search)], False, False))
                    if len(searches) > 1:
                        targets.append(("editor_minus_best_search", [(condition, method)] + [
                            (condition, search) for search in searches
                        ], False, True))
                for name, target, compare_conditions, best_search in targets:
                    comparisons.append({
                        "task": task, "comparison": name,
                        **_compare(
                            table, config, task, target, compare_conditions=compare_conditions,
                            best_search=best_search, replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed,
                        ),
                    })
    return {
        "schema_version": 1, "phase": config.phase, "fixture_source": config.fixture_source,
        "config_sha256": config_hash, "counts": counts, "availability": availability,
        "bootstrap": {
            "replicates": bootstrap_replicates, "seed": bootstrap_seed, "unit": "paired experiment seed",
            "method": "percentile intervals conditional on these tasks and the fixed held-out starts",
            "minimum_pairs_for_interval": 2,
        },
        "interpretation": (
            "Smoke and pilot results describe these measured runs. "
            "Intervals with few seeds are unstable. Results do not quantify transfer to new tasks or checkpoints."
        ),
        "groups": groups, "heldout_comparisons": comparisons,
    }
