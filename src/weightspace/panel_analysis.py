"""Read the frozen panel journals and calculate the predeclared contrasts."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from .analysis import _read_run
from .artifacts import json_digest
from .codex_editor import USAGE_FIELDS, catalog_provenance
from .edits import EditLimits
from .experiment import ExperimentConfig, planned_cells
from .panel import ARMS, COMPOUND_METHODS, FORMS, _configs, _digest, _frozen_plan, _read_json


def _events(path, run_id):
    """Preserve readable records before a damaged or interrupted journal tail."""
    if not path.is_file():
        return [], None
    events = []
    with path.open("rb") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
                if (
                    not isinstance(entry, dict) or entry.get("sequence") != len(events)
                    or not isinstance(entry.get("event"), str) or entry.get("run_id") != run_id
                ):
                    raise ValueError("nonconsecutive event")
                events.append(entry)
            except ValueError:
                return events, f"unreadable journal event at sequence {len(events)}"
    return events, None


def _experiment_context(root, job, expected, frozen):
    """Validate that the experiment recorded the corresponding frozen job."""
    experiment = root / "experiments" / job
    config_path = root / "configs" / f"{job}.json"
    if _digest(config_path) != frozen["files_sha256"][f"configs/{job}.json"]:
        raise ValueError("configuration differs from the frozen input manifest")
    config_data = _read_json(config_path)
    # Use the original absolute path from staging so a private evidence copy can be analyzed elsewhere.
    normalized = asdict(expected)
    normalized["editor"]["model_catalog_source"] = config_data["editor"]["model_catalog_source"]
    if json_digest(normalized) != json_digest(config_data):
        raise ValueError("configuration differs from the prospective plan")
    if not (experiment / "experiment.json").is_file():
        raise ValueError("experiment manifest is absent")
    config = ExperimentConfig(**config_data)
    manifest = _read_json(experiment / "experiment.json")
    digest = json_digest(config_data)
    catalog = catalog_provenance(config.editor_options(), experiment)
    if (
        manifest.get("schema_version") != 1 or manifest.get("source") != frozen["source"]["package"]
        or manifest["config"] != config_data or manifest["config_sha256"] != digest
        or manifest["cells"] != planned_cells(config)
        or manifest["live_editor_launch_bounds"]["editor_cli_launches_upper_bound"] != 4
        or manifest["live_editor_launch_bounds"]["model_catalog"] != catalog
        or catalog["catalog_sha256"] != frozen["files_sha256"][catalog["catalog_file"]]
    ):
        raise ValueError("experiment does not match the frozen source, plan, or catalog")
    return digest, EditLimits(**config.edit_limits), config.editor_options(), catalog


def _cell_evidence(events):
    results = [event for event in events if event["event"] == "candidate_result"]
    editor = [event for event in events if event["event"] in ("editor_turn", "editor_process_error")]
    completion = next((event for event in reversed(events) if event["event"] in (
        "run_completed", "heldout_result", "final_selected",
    )), {})
    checkpoint = next((event for event in events if event["event"] == "checkpoint"), {})
    scores = {key: completion.get(key) for key in (
        "candidate_id", "ops", "dev", "heldout", "starting_policy_sha256", "selected_policy_sha256",
        "heldout_rows_reserved", "heldout_rows_used",
    )}
    scores["starting_policy_sha256"] = checkpoint.get("starting_policy_sha256")
    scores["selected_policy_sha256"] = completion.get("selected_policy_sha256", completion.get("policy_sha256"))
    budget = next((event["budget"] for event in reversed(events) if "budget" in event), None)
    return {
        **scores, "budget": budget,
        "dev_minus_heldout": (
            scores["dev"]["objective"] - scores["heldout"]["objective"]
            if scores["dev"] is not None and scores["heldout"] is not None else None
        ),
        "candidate_result_counts": dict(Counter(event.get("status", "unknown") for event in results)),
        "candidate_errors": [
            {"candidate_id": event.get("candidate_id"), "status": event.get("status"), "reason": event.get("reason")}
            for event in results if event.get("status") != "scored"
        ],
        "editor_attempt_records": [
            {key: event.get(key) for key in (
                "event", "sequence", "call", "attempt", "usage", "usage_excesses", "error_type", "interruption",
                "stream_error", "proposal_error",
            )} | {"model_command_recorded": "command" in event}
            for event in editor
        ],
        "editor_preflight_events": [
            {key: event.get(key) for key in ("event", "sequence", "check", "error_type", "returncode", "interruption")}
            for event in events if event["event"] in ("editor_preflight", "editor_preflight_failed")
        ],
    }


def _cell_rows(root, job, config, frozen, executions):
    experiment = root / "experiments" / job
    cells = planned_cells(config)
    expected_journals = max(
        (index + 1 for index, cell in enumerate(cells) if (experiment / "runs" / cell["run_id"]).is_dir()),
        default=0,
    )
    if job in executions:
        expected_journals = max(expected_journals, executions[job]["completed"] + executions[job]["failed"])
    context, problem = None, None
    if experiment.is_dir():
        try:
            context = _experiment_context(root, job, config, frozen)
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            problem = str(error)
    pair = frozen["fixtures"][job]
    rows = []
    for index, cell in enumerate(cells):
        run = experiment / "runs" / cell["run_id"]
        journal = run / "events.jsonl"
        events, damage = _events(journal, cell["run_id"])
        try:
            evidence = _cell_evidence(events)
        except (KeyError, TypeError, ValueError):
            evidence = _cell_evidence([event for event in events if event["event"].startswith("editor_")])
            damage = damage or "malformed cell evidence"
        if not run.is_dir():
            status, reason = (
                ("missing", "expected run journal is absent") if index < expected_journals
                else ("not_started", "no run directory was created")
            )
        elif damage or context is None:
            status, reason = "invalid", damage or problem
        else:
            status, reason, completed = _read_run(experiment, cell, *context)
            checkpoint = next((event for event in events if event["event"] == "checkpoint"), {})
            if completed is not None and (
                completed["provenance"] != pair["provenance"]
                or completed["starting_policy_sha256"] != pair["starting_policy_sha256"][cell["condition"]]
                or checkpoint.get("transform") != (pair["transform"] if cell["condition"] == FORMS[1] else None)
                or any(completed["budget"][key] != value for key, value in (
                    ("candidate_cap", 2), ("row_budget", 6400), ("rows_per_candidate", 3200),
                ))
                or completed["heldout_rows_reserved"] != 40000
                or completed["attempts"].get("failed", 0)
            ):
                status = "invalid"
                reason = "candidate evaluator failure" if completed["attempts"].get("failed") else (
                    "completed cell differs from the frozen splits, starting checkpoint, or budget"
                )
        rows.append({
            **cell, "job": job, "fixture_seed": config.fixture_seed, "status": status, "reason": reason,
            "journal_sha256": _digest(journal) if journal.is_file() else None, **evidence,
        })
    return rows


def _distribution(values):
    return {
        "n": len(values), "mean": mean(values) if values else None, "sd": stdev(values) if len(values) > 1 else None,
        "minimum": min(values) if values else None, "maximum": max(values) if values else None,
    }


def _contrast_output(jobs, labels, differences, *, draws=None, bootstrap_values=None):
    per_fixture = []
    for job, values in zip(jobs, differences):
        pooled = mean(values) if all(value is not None for value in values) else None
        per_fixture.append({"job": job, "differences": dict(zip(labels, values)), "pooled": pooled})
    eligible = sum(fixture["pooled"] is not None for fixture in per_fixture)
    output = {
        "paired_fixtures": eligible, "planned_fixtures": len(jobs), "per_fixture": per_fixture,
        "pooled": None, "by_form": None,
    }
    if eligible == len(jobs):
        matrix = np.asarray(differences, dtype=float)
        samples = matrix[draws].mean(axis=1) if bootstrap_values is None else bootstrap_values

        def estimate(value, replicates):
            return {"estimate": float(value), "ci95": np.quantile(replicates, [0.025, 0.975]).tolist()}

        output["pooled"] = estimate(matrix.mean(), samples.mean(axis=1))
        output["by_form"] = {
            name: estimate(matrix[:, index].mean(), samples[:, index]) for index, name in enumerate(labels)
        }
    return output


def _contrasts(jobs, rows, bootstrap):
    table = {
        (row["job"], row["condition"], row["method"]): row
        for row in rows if row["status"] == "completed"
    }
    draws = np.random.default_rng(bootstrap["seed"]).integers(len(jobs), size=(bootstrap["replicates"], len(jobs)))

    def difference(left, right):
        first, second = table.get(left), table.get(right)
        return None if first is None or second is None else first["heldout"]["objective"] - second["heldout"]["objective"]

    contrasts = {}
    for arm in ("velocity_controller", *COMPOUND_METHODS, "no_edit"):
        differences = [[difference((job, form, "codex"), (job, form, arm)) for form in FORMS] for job in jobs]
        contrasts[f"codex_minus_{arm}"] = _contrast_output(jobs, FORMS, differences, draws=draws)
    for arm in ARMS:
        differences = [[difference((job, FORMS[0], arm), (job, FORMS[1], arm))] for job in jobs]
        contrasts[f"original_minus_transformed_{arm}"] = _contrast_output(
            jobs, ("original_minus_transformed",), differences, draws=draws,
        )

    keys = [(job, form, method) for job in jobs for form in FORMS for method in COMPOUND_METHODS]
    winner = None
    selected_bootstrap = None
    selected_differences = [[None] * len(FORMS) for _ in jobs]
    if all(key in table for key in keys):
        shape = (len(jobs), len(FORMS), len(COMPOUND_METHODS))
        dev = np.asarray([table[key]["dev"]["objective"] for key in keys]).reshape(shape)
        winner = COMPOUND_METHODS[int(np.argmax(dev.mean(axis=(0, 1))))]
        selected_differences = [[difference((job, form, "codex"), (job, form, winner)) for form in FORMS] for job in jobs]
        if all(value is not None for pair in selected_differences for value in pair):
            search_heldout = np.asarray([table[key]["heldout"]["objective"] for key in keys]).reshape(shape)
            codex_heldout = np.asarray([
                [table[job, form, "codex"]["heldout"]["objective"] for form in FORMS] for job in jobs
            ])
            bootstrap_winners = np.argmax(dev[draws].mean(axis=(1, 2)), axis=1)
            boot_search = np.take_along_axis(search_heldout[draws], bootstrap_winners[:, None, None, None], axis=3)
            selected_bootstrap = (codex_heldout[draws] - boot_search[..., 0]).mean(axis=1)
    contrasts["codex_minus_dev_selected_compound"] = {
        "development_winner": winner, "development_tie_order": list(COMPOUND_METHODS),
        **_contrast_output(jobs, FORMS, selected_differences, bootstrap_values=selected_bootstrap),
    }
    return contrasts


def summarize_panel(root):
    """Account for every planned cell, including interrupted and unstarted work."""
    root = Path(root).resolve()
    frozen, plan = _frozen_plan(root)
    jobs = _configs(plan, root)
    ledger, ledger_problem = _events(root / "execution" / "events.jsonl", "execution")
    executions = {event["job"]: event["result"] for event in ledger if event["event"] == "fixture_terminal"}
    rows = [row for job, config in jobs.items() for row in _cell_rows(root, job, config, frozen, executions)]
    counts = dict.fromkeys(("completed", "failed", "incomplete", "invalid", "missing", "not_started"), 0)
    counts.update(Counter(row["status"] for row in rows))
    attempts = [attempt for row in rows for attempt in row["editor_attempt_records"]]
    usage = [attempt["usage"] for attempt in attempts if attempt["usage"] is not None]
    groups = []
    for form in FORMS:
        for arm in ARMS:
            cells = [row for row in rows if row["condition"] == form and row["method"] == arm and row["status"] == "completed"]
            groups.append({
                "condition": form, "method": arm,
                **{split: {
                    field: _distribution([row[split][field] for row in cells])
                    for field in ("objective", "failure_success", "normal_success")
                } for split in ("dev", "heldout")},
                "dev_minus_heldout": _distribution([row["dev_minus_heldout"] for row in cells]),
            })
    return {
        "schema_version": 1, "frozen_manifest_sha256": _digest(root / "frozen-manifest.json"),
        "counts": counts, "planned_cells": len(rows), "cells": rows, "groups": groups,
        "bootstrap": {**plan["bootstrap"], "unit": "fixture with both paired forms", "method": "percentile"},
        "primary_contrast": "codex_minus_velocity_controller",
        "contrasts": _contrasts(list(jobs), rows, plan["bootstrap"]),
        "execution": {"last_event": ledger[-1] if ledger else None, "journal_problem": ledger_problem},
        "native": {
            "planned_attempt_cap": plan["max_native_launches"],
            "journaled_model_command_attempts": sum(attempt["model_command_recorded"] for attempt in attempts),
            "journaled_editor_turns": sum(attempt["event"] == "editor_turn" for attempt in attempts),
            "journaled_attempts_with_usage": len(usage),
            "sum_of_reported_usage": {field: sum(record.get(field, 0) for record in usage) for field in sorted(USAGE_FIELDS)},
            "actual_charge_usd": None,
        },
    }
