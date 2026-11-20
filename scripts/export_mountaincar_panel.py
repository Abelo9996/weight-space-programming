"""Export a terminal MountainCar panel or replay its fixed public recipes offline."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import gzip
import io
import json
from math import isfinite
import os
from pathlib import Path
import re
import tarfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

if __package__:
    from .export_baseline_results import (
        BUDGET, EDIT_LIMITS, FORBIDDEN_TEXT, GENERATION, OPS_FIELDS, SCORE,
        canonical, open_directory, require, safe_read, sha256, take, write_new,
    )
else:
    from export_baseline_results import (
        BUDGET, EDIT_LIMITS, FORBIDDEN_TEXT, GENERATION, OPS_FIELDS, SCORE,
        canonical, open_directory, require, safe_read, sha256, take, write_new,
    )

from weightspace.budget import RowMeter
from weightspace.codex_editor import MODEL_EVIDENCE, USAGE_FIELDS
from weightspace.edits import EditError, EditLimits, apply_edits
from weightspace.experiment import _source_provenance, _transform_policy, planned_cells
from weightspace.panel import ARMS, FIXTURE_FILES, FORMS, TASK, _configs, _frozen_plan
from weightspace.panel_analysis import _contrasts, _distribution, _events, summarize_panel
from weightspace.policies import policy_digest
from weightspace.tasks import TASKS, evaluate_split, load_task


ARCHIVE_NAME = "mountaincar-panel-2026-11-20-fixtures.tar.gz"
CELL = ("job", "fixture_seed", "task", "seed", "condition", "method", "run_id")
STATUS = ("completed", "failed", "incomplete", "invalid", "missing", "not_started")
CANDIDATE_STATUS = ("scored", "rejected", "failed", "budget_exhausted")
METRICS = ("objective", "failure_success", "normal_success")
LICENSES = {
    "LICENSE": "LICENSE",
    "LICENSE-WeightBench": "src/weightspace/_vendor/WEIGHTBENCH_LICENSE",
    "LICENSE-Gymnasium": "src/weightspace/_vendor/GYMNASIUM_LICENSE",
}
NPZ_KEYS = {
    "public/policy.npz": {f"fc{layer}.{part}" for layer in range(3) for part in ("weight", "bias")},
    "public/failure_trajectory.npz": {"obs", "action", "reward", "success", "start"},
    "private/eval_sets.npz": {"fail_train", "norm_train", "fail_eval", "norm_eval", "traj_start"},
}


def _hash(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA-256 value")
    return value


def _number(value):
    require(type(value) in (float, int) and isfinite(value), "expected a finite number")
    return value


def _nullable(value, project):
    return None if value is None else project(value)


def _numeric_fields(record, fields):
    return {key: _nullable(record[key], _number) for key in fields}


def _projection(value, project, *, tolerate_damage):
    try:
        return _nullable(value, project)
    except (KeyError, OverflowError, TypeError, ValueError):
        if not tolerate_damage:
            raise
        return None


def _version(value):
    require(isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+){1,2}(?:\+cpu)?", value),
            "unexpected software version")
    return value


def _transform(record):
    operations = []
    for operation, (kind, layer) in zip(record["operations"], (
        ("permutation", "fc0"), ("relu_scale", "fc0"), ("permutation", "fc1"), ("relu_scale", "fc1"),
    ), strict=True):
        require((operation["kind"], operation["layer"]) == (kind, layer), "unexpected frozen transform")
        field = "order" if kind == "permutation" else "scales"
        operations.append({"kind": kind, "layer": layer, field: [_number(value) for value in operation[field]]})
    return {
        "operations": operations, "torch_version": _version(record["torch_version"]),
        **_numeric_fields(record, ("absolute_tolerance", "fixed_inputs", "fixed_inputs_seed_stream", "max_abs_logit_error")),
    }


def _ops(value):
    """Copy only recognized operation names, fields, layers, and numeric data."""
    if not isinstance(value, list):
        return None
    output = []
    try:
        for entry in value:
            fields = OPS_FIELDS[entry["op"]]
            require(set(entry) == {"op", "layer", *fields} and entry["layer"] in ("fc0", "fc1", "fc2"),
                    "unexpected edit fields or layer")
            projected = take(entry, ("op", "layer"))
            for key in fields:
                projected[key] = [_number(item) for item in entry[key]] if key in ("u", "v") else _number(entry[key])
            output.append(projected)
    except (KeyError, OverflowError, TypeError, ValueError):
        return None
    return output


def _reason(status, message, events):
    """Translate raw journal prose to stable public failure categories."""
    if status == "completed":
        return None
    reasons = {
        "not_started": "not_started",
        "missing": "missing_journal",
        "incomplete": "missing_terminal_completion",
        "failed": "run_failed",
        "invalid": "integrity_check_failed",
    }
    if status == "invalid" and message == "candidate evaluator failure":
        return "candidate_evaluator_failure"
    if status == "invalid" and isinstance(message, str) and message.startswith("unreadable journal event at sequence "):
        return "damaged_journal"
    if status == "failed" and events and events[-1]["event"] == "run_failed":
        if events[-1].get("error_type") == "InvalidProposal":
            return "editor_invalid_proposal"
        if events[-1].get("error_type") == "EditorError":
            return "native_editor_error"
    return reasons[status]


def _native_attempt(event, stdout_sha256):
    usage = event["usage"]
    excesses = event["usage_excesses"] or {}
    return {
        "event": event["event"], **_numeric_fields(event, ("sequence", "call", "attempt")),
        "model_command_recorded": event["model_command_recorded"] is True,
        "stdout_sha256": stdout_sha256,
        "usage": None if usage is None else _numeric_fields(usage, sorted(USAGE_FIELDS & usage.keys())),
        "usage_excesses": {
            field: _numeric_fields(excesses[field], ("observed", "limit"))
            for field in ("input_tokens", "output_tokens") if field in excesses
        },
        "process_error": event["error_type"] is not None,
        "interrupted": event["interruption"] is not None,
        "stream_error": event["stream_error"] is not None,
        "proposal_error": event["proposal_error"] is not None,
    }


def _candidate_records(events, run_id, *, tolerate_damage):
    requests = {}
    for event in events:
        kind = event["event"]
        if kind not in ("candidate_request", "candidate_reserved", "candidate_result"):
            continue
        identifier = event.get("candidate_id")
        if type(identifier) is not int or identifier < 0:
            continue
        if kind == "candidate_request":
            requests.setdefault(identifier, {
                "run_id": run_id, "candidate_id": identifier, "request_sequence": event["sequence"],
                "request_ops": _ops(event.get("ops")), "reservation": None, "result": None,
            })
        elif identifier in requests:
            row = requests[identifier]
            if kind == "candidate_reserved" and row["reservation"] is None:
                row["reservation"] = {
                    "sequence": event["sequence"], "ops": _ops(event.get("ops")),
                    **{key: _projection(event.get(key), _hash, tolerate_damage=tolerate_damage)
                       for key in ("edit_sha256", "policy_sha256")},
                    "budget": _projection(event.get("budget"), lambda record: _numeric_fields(record, BUDGET),
                                          tolerate_damage=tolerate_damage),
                }
            elif kind == "candidate_result" and row["result"] is None:
                status = event.get("status")
                if status not in CANDIDATE_STATUS:
                    continue
                row["result"] = {
                    "sequence": event["sequence"], "status": status,
                    "reason_code": {
                        "scored": None, "rejected": "illegal_recipe", "failed": "evaluator_error",
                        "budget_exhausted": "development_budget_exhausted",
                    }[status],
                    **{key: _projection(event.get(key), _number, tolerate_damage=tolerate_damage)
                       for key in ("changed_entries", "touched_entries")},
                    "score": _projection(event.get("score"), lambda record: _numeric_fields(record, SCORE),
                                         tolerate_damage=tolerate_damage),
                    "budget": _projection(event.get("budget"), lambda record: _numeric_fields(record, BUDGET),
                                          tolerate_damage=tolerate_damage),
                }
    return list(requests.values())


def _project_rows(root, jobs, raw):
    expected = [dict(cell, job=job, fixture_seed=config.fixture_seed)
                for job, config in jobs.items() for cell in planned_cells(config)]
    require(len(expected) == len(raw) == 60, "panel does not account for all 60 planned cells")
    rows, candidates = [], []
    for cell, private in zip(expected, raw, strict=True):
        require(cell == take(private, CELL) and private["status"] in STATUS, "cell differs from the frozen plan")
        damaged = private["status"] == "invalid"
        journal = root / "experiments" / cell["job"] / "runs" / cell["run_id"] / "events.jsonl"
        journal_hash = _nullable(private["journal_sha256"], _hash)
        events, _ = _events(journal, cell["run_id"])
        if journal_hash is not None:
            require(sha256(safe_read(root, journal.relative_to(root).as_posix())) == journal_hash,
                    "cell journal changed during export")
        current = _candidate_records(events, cell["run_id"], tolerate_damage=damaged)
        native_hashes = {
            event["sequence"]: _projection(event.get("raw_stdout_sha256"), _hash, tolerate_damage=damaged)
            for event in events if event["event"] == "editor_turn"
        }
        attempt_counts = Counter(item["result"]["status"] for item in current if item["result"])
        if private["status"] == "completed":
            require(attempt_counts == private["candidate_result_counts"], "candidate counts differ from cell evidence")
        rows.append(cell | {
            "status": private["status"], "reason_code": _reason(private["status"], private["reason"], events),
            "journal_sha256": journal_hash, "candidate_result_counts": dict(attempt_counts),
            "candidate_id": _projection(private["candidate_id"], _number, tolerate_damage=damaged),
            "selected_ops": _ops(private["ops"]),
            **{field: _projection(private[field], _hash, tolerate_damage=damaged)
               for field in ("starting_policy_sha256", "selected_policy_sha256")},
            **{split: _projection(private[split], lambda record: _numeric_fields(record, SCORE), tolerate_damage=damaged)
               for split in ("dev", "heldout")},
            "budget": _projection(private["budget"], lambda record: _numeric_fields(record, BUDGET), tolerate_damage=damaged),
            **{field: _projection(private[field], _number, tolerate_damage=damaged)
               for field in ("dev_minus_heldout", "heldout_rows_reserved", "heldout_rows_used")},
            "editor_attempt_records": [_native_attempt(event, native_hashes.get(event["sequence"]))
                                       for event in private["editor_attempt_records"]
                                       if event["event"] in ("editor_turn", "editor_process_error")],
        })
        candidates.extend(current)
    return rows, candidates


def _groups(rows):
    output = []
    for form in FORMS:
        for arm in ARMS:
            cells = [row for row in rows if row["condition"] == form and row["method"] == arm and row["status"] == "completed"]
            output.append({
                "condition": form, "method": arm,
                **{split: {field: _distribution([row[split][field] for row in cells]) for field in METRICS}
                   for split in ("dev", "heldout")},
                "dev_minus_heldout": _distribution([row["dev_minus_heldout"] for row in cells]),
            })
    return output


def _csv(rows, jobs):
    buffer = io.StringIO(newline="")
    fields = (*CELL, "status", "reason_code", "candidate_id", "starting_policy_sha256", "selected_policy_sha256",
              "dev_sha256", "heldout_sha256", "dev_minus_heldout", "heldout_rows_used", "heldout_rows_reserved",
              *[f"{split}_{key}" for split in ("dev", "heldout") for key in SCORE],
              *[f"budget_{key}" for key in BUDGET], *[f"candidate_{key}" for key in CANDIDATE_STATUS],
              "model_command_attempts")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(take(row, (*CELL, "status", "reason_code", "candidate_id", "starting_policy_sha256",
                                   "selected_policy_sha256", "dev_minus_heldout", "heldout_rows_used",
                                   "heldout_rows_reserved")) |
                        take(jobs[row["job"]], ("dev_sha256", "heldout_sha256")) |
                        {f"{split}_{key}": value for split in ("dev", "heldout", "budget")
                         for key, value in (row[split] or {}).items()} |
                        {f"candidate_{key}": row["candidate_result_counts"].get(key, 0) for key in CANDIDATE_STATUS} |
                        {"model_command_attempts": sum(a["model_command_recorded"] for a in row["editor_attempt_records"])})
    return buffer.getvalue().encode()


def _verified_input(root, frozen, path):
    content = safe_read(root, path)
    require(sha256(content) == _hash(frozen["files_sha256"][path]), "frozen input changed")
    return content


def _archive(members):
    checksums = {name: sha256(content) for name, content in sorted(members.items())}
    members = members | {"SHA256SUMS": "".join(f"{digest}  {name}\n" for name, digest in checksums.items()).encode()}
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, content in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.mode, info.size = 0o644, len(content)
                archive.addfile(info, io.BytesIO(content))
    return output.getvalue(), checksums


def _fixture_files(root, frozen, jobs):
    members, rows = {}, {}
    for job, config in jobs.items():
        prefix = f"fixtures/{job}/{TASK}"
        for name in FIXTURE_FILES:
            key = f"{prefix}/{name}"
            content = _verified_input(root, frozen, key)
            if name in NPZ_KEYS:
                with ZipFile(io.BytesIO(content)) as arrays:
                    require(sorted(arrays.namelist()) == sorted(f"{field}.npy" for field in NPZ_KEYS[name])
                            and not arrays.comment and not any(item.comment for item in arrays.infolist()),
                            "unexpected fixture archive members or comments")
                with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
                    require(all(arrays[field].dtype.kind in "bfiu" for field in NPZ_KEYS[name]),
                            "fixture archive arrays must be numeric or boolean")
            else:
                generation = json.loads(content)
                require(generation == take(generation, GENERATION) and generation["task_id"] == TASK
                        and generation["seed"] == config.fixture_seed and generation["fixture_format"] == "weightspace-v1"
                        and not FORBIDDEN_TEXT.search(content.decode()), "unexpected generated fixture manifest")
                for field in ("demo_sha256", "policy_sha256", "starts_sha256"):
                    _hash(generation[field])
                require(generation["evaluator_source"] == TASKS[TASK].evaluator_source, "unexpected fixture evaluator")
                _version(generation["numpy_version"])
                _version(generation["torch_version"])
            members[key] = content
        pair = frozen["fixtures"][job]
        private_config = _verified_input(root, frozen, f"configs/{job}.json")
        public_config = asdict(config)
        raw_config = json.loads(private_config)
        public_config["editor"]["model_catalog_source"] = raw_config["editor"]["model_catalog_source"]
        require(canonical(public_config) == canonical(raw_config), "saved job config differs from the prospective plan")
        rows[job] = {
            "fixture_seed": config.fixture_seed, "run_seed": config.seeds[0],
            "config_file_sha256": sha256(private_config), "config_canonical_sha256": sha256(canonical(raw_config)),
            **{key: _hash(pair["provenance"][key]) for key in ("dev_sha256", "heldout_sha256")},
            "starting_policy_sha256": {form: _hash(pair["starting_policy_sha256"][form]) for form in FORMS},
            "transform": _transform(pair["transform"]),
        }
    return members, rows


def _readme(commit):
    return f"""# MountainCar policy repair comparison

This artifact records every planned fixture, parameterization, and method in the five-fixture MountainCar study. See the [frozen protocol](protocol.md) for objectives, budgets, and prespecified contrasts. The primary contrast compares Codex with the task-aware analytic velocity controller. Candidate selection precedes held-out evaluation. The two-slot development budget permits one scored proposal after the starting checkpoint.

- [cells.csv](cells.csv) records the 60 planned cells, including failures, invalid evidence, and unstarted work. A blank metric means the journal lacks that result. Only completed cells enter the statistical summaries.
- [summary.json](summary.json) records the paired contrasts, all outcomes and their public reason codes, and reported native attempt usage. An aggregate contrast requires every prespecified fixture pair.
- [candidates.jsonl](candidates.jsonl) records each readable proposal, reservation, and development result. Candidate IDs are local to a cell. A null recipe means the available record did not contain a releasable operation list.
- [config.json](config.json) contains replay parameters and hashes of the original frozen plan and job configs. [manifest.json](manifest.json) contains source and output hashes and the fixture archive checksum.

## Replay fixed recipes

Download `{ARCHIVE_NAME}` from the companion release. Install the frozen evaluator from commit `{commit}` into a clean Python environment with NumPy and PyTorch. From a checkout containing the export script, run the following command with that environment's interpreter.

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
  python scripts/export_mountaincar_panel.py replay \\
  --results results/mountaincar-panel-2026-11-20 \\
  --fixture-archive {ARCHIVE_NAME}
```

Replay checks release hashes, rebuilds the paired checkpoints, validates released edit recipes, and compares recorded successful development and held-out evaluations. It separately counts absent or invalid cell evidence and nonnumeric candidate outcomes. The replay runs offline with the released fixtures and uses zero model calls. It requires no native login or descriptor. Its results describe the released checkpoints and start panels. Future comparisons should use fresh held-out starts.
""".encode()


def export(root, summary_path, *, readme=None):
    """Cross-check a terminal private summary and return staged public file bytes."""
    root = Path(root).resolve()
    ledger, problem = _events(root / "execution" / "events.jsonl", "execution")
    require(problem is None and ledger and ledger[-1]["event"] in ("panel_terminal", "panel_interrupted"),
            "panel ledger has no validated terminal state")
    frozen, plan = _frozen_plan(root)
    jobs = _configs(plan, root)
    raw = summarize_panel(root)
    require(raw == json.loads(summary_path.read_bytes()) and raw["execution"]["last_event"] == ledger[-1],
            "provided private summary differs from terminal evidence")
    require(_source_provenance() == frozen["source"]["package"]
            and sha256(safe_read(root, "frozen-manifest.json")) == raw["frozen_manifest_sha256"],
            "loaded source or frozen input manifest changed")
    rows, candidates = _project_rows(root, jobs, raw["cells"])
    counts = {key: sum(row["status"] == key for row in rows) for key in STATUS}
    bootstrap = _numeric_fields(plan["bootstrap"], ("replicates", "seed"))
    groups, contrasts = _groups(rows), _contrasts(list(jobs), rows, bootstrap)
    require(counts == raw["counts"] and groups == raw["groups"] and contrasts == raw["contrasts"],
            "public outcomes differ from the frozen analysis")
    attempts = [record for row in rows for record in row["editor_attempt_records"]]
    usage = [record["usage"] for record in attempts if record["usage"] is not None]
    native = {
        "planned_attempt_cap": _number(plan["max_native_launches"]),
        "journaled_model_command_attempts": sum(record["model_command_recorded"] for record in attempts),
        "journaled_editor_turns": sum(record["event"] == "editor_turn" for record in attempts),
        "journaled_attempts_with_usage": len(usage),
        "sum_of_reported_usage": {key: sum(record.get(key, 0) for record in usage) for key in sorted(USAGE_FIELDS)},
        "actual_charge_usd": None,
    }
    require(native == raw["native"], "public native usage differs from the frozen analysis")
    editor = next(iter(jobs.values())).editor_options()
    require(editor.model == "gpt-6.2-sol" and editor.cli_version == "codex-cli 0.169.0" and editor.effort == "low"
            and editor.billing_assumption == "subscription_only", "unexpected native editor identity")
    members, job_rows = _fixture_files(root, frozen, jobs)
    source = _verified_input(root, frozen, "source.tar")
    wheel = _verified_input(root, frozen, "package.whl")
    protocol = _verified_input(root, frozen, "protocol.md")
    with tarfile.open(fileobj=io.BytesIO(source), mode="r:") as archive:
        commit = archive.pax_headers.get("comment", "")
        require(re.fullmatch(r"[0-9a-f]{40}", commit) and commit == frozen["source"]["git_commit"],
                "source commit differs from the frozen manifest")
        for name, path in LICENSES.items():
            require(archive.getmember(path).isfile(), "license member is not a regular file")
            with archive.extractfile(path) as handle:
                members[name] = handle.read()
    members["ATTRIBUTION.md"] = (
        "# MountainCar panel fixtures\n\n"
        "These generated policies, start panels, and standalone demonstrations use the adapted WeightBench "
        "classic-control builder at commit `2acd60172fc9758eb9030b1cab4a2ff29be5c993` and Gymnasium dynamics. "
        "The accompanying MIT notices preserve attribution.\n\n"
        "[Weight-Space Programming](https://github.com/Abelo9996/weight-space-programming) · "
        "[WeightBench](https://github.com/Aayush-Marasini/WeightBench) · "
        "[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)\n"
    ).encode()
    require(all(not FORBIDDEN_TEXT.search(members[name].decode()) for name in (*LICENSES, "ATTRIBUTION.md")),
            "unexpected license or attribution text")
    archive, member_hashes = _archive(members)
    experiment = plan["experiment"]
    config = {
        "schema_version": 1, "task": TASK, "conditions": FORMS, "methods": ARMS,
        **_numeric_fields(experiment, ("candidate_cap", "dev_per_region", "heldout_per_region", "es_offspring")),
        "edit_limits": _numeric_fields(experiment["edit_limits"], EDIT_LIMITS), "jobs": job_rows, "job_order": list(jobs),
        "bootstrap": bootstrap, "frozen_plan_sha256": _hash(frozen["files_sha256"]["plan.json"]),
        "editor": {
            "tool": "Codex CLI", "requested_model": editor.model, "cli_version": editor.cli_version,
            "reasoning_effort": editor.effort, "access": "native ChatGPT subscription", "model_evidence": MODEL_EVIDENCE,
            **{field: _number(getattr(editor, field)) for field in (
                "max_calls", "retries", "timeout_seconds", "max_output_bytes",
                "max_observed_input_tokens", "max_observed_output_tokens",
            )},
        },
    }
    terminal = ledger[-1]
    summary = {
        "schema_version": 1, "planned_cells": len(rows), "counts": counts, "cells": rows,
        "groups": groups, "contrasts": contrasts, "bootstrap": bootstrap,
        "primary_contrast": "codex_minus_velocity_controller", "native": native,
        "execution": {
            "terminal_event": terminal["event"],
            "stopped_on_hard_native_failure": (terminal.get("stop") or {}).get("kind") == "hard_native_failure",
        },
    }
    files = {
        "README.md": _readme(commit) if readme is None else readme.read_bytes(),
        "protocol.md": protocol, "cells.csv": _csv(rows, job_rows),
        "summary.json": canonical(summary) + b"\n", "config.json": canonical(config) + b"\n",
        "candidates.jsonl": b"".join(canonical(row) + b"\n" for row in candidates),
    }
    manifest = {
        "schema_version": 1, "git_commit": commit,
        "source_sha256": _hash(frozen["source"]["package"]["source_sha256"]),
        "source_archive_sha256": sha256(source), "wheel_sha256": sha256(wheel),
        "frozen_manifest_sha256": _hash(raw["frozen_manifest_sha256"]),
        "data_file_sha256": {name: sha256(content) for name, content in sorted(files.items())},
        "fixture_archive": {"name": ARCHIVE_NAME, "sha256": sha256(archive), "size_bytes": len(archive),
                            "member_sha256": member_hashes},
    }
    files["manifest.json"] = canonical(manifest) + b"\n"
    require(all(not FORBIDDEN_TEXT.search(content.decode()) for content in files.values()),
            "unexpected private text in release files")
    return files, archive


def _public_fixture_archive(path, manifest, jobs):
    data = path.read_bytes()
    require(len(data) == manifest["size_bytes"] and sha256(data) == _hash(manifest["sha256"]),
            "fixture archive checksum mismatch")
    hashes = manifest["member_sha256"]
    expected = {f"fixtures/{job}/{TASK}/{name}" for job in jobs for name in FIXTURE_FILES}
    require(set(hashes) == expected | set(LICENSES) | {"ATTRIBUTION.md"}, "unexpected release archive index")
    members = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for info in archive:
            require(info.isfile() and info.name not in members and info.name in hashes.keys() | {"SHA256SUMS"},
                    "unexpected archived member")
            with archive.extractfile(info) as handle:
                members[info.name] = handle.read()
    require(set(members) == set(hashes) | {"SHA256SUMS"}
            and all(sha256(members[name]) == _hash(digest) for name, digest in hashes.items()),
            "fixture archive member checksum mismatch")
    checksum_bytes = "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())).encode()
    require(checksum_bytes == members["SHA256SUMS"], "embedded fixture checksum index differs")
    return {path: members[path] for path in expected}


def _score_replay(task, policy, split, recorded):
    meter = RowMeter(split.max_rows(task.spec.horizon))
    result = evaluate_split(task.spec, policy, split, meter.charge)
    require(asdict(result) == recorded and meter.used == result.rows, "recorded rollout score differs from replay")
    return result


def _replay_candidate(task, base, limits, record, counts):
    reservation, result = record["reservation"], record["result"]
    if reservation is not None:
        applied = apply_edits(base, reservation["ops"], limits)
        require(sha256(canonical(applied.ops)) == reservation["edit_sha256"]
                and policy_digest(applied.policy) == reservation["policy_sha256"], "candidate recipe digest mismatch")
        if result is None:
            counts["reserved_without_terminal_result"] += 1
            return None
        require(result["status"] in ("scored", "failed") and applied.changed_entries == result["changed_entries"]
                and applied.touched_entries == result["touched_entries"], "candidate edit accounting differs")
        if result["status"] == "scored":
            _score_replay(task, applied.policy, task.dev, result["score"])
            counts["development_scores_replayed"] += 1
            return result["score"], applied.ops, reservation["policy_sha256"]
        counts["candidate_evaluator_failures_unscored"] += 1
    elif result is None:
        counts["unreserved_without_terminal_result"] += 1
    elif result["status"] == "rejected":
        ops = record["request_ops"]
        if ops is None:
            counts["structural_rejections_without_safe_recipe"] += 1
        else:
            try:
                apply_edits(base, ops, limits)
            except EditError:
                counts["structural_rejections_replayed"] += 1
            else:
                raise ValueError("recorded structural rejection accepted by the frozen edit validator")
    else:
        require(result["status"] == "budget_exhausted", "candidate result requires a reservation")
        counts["budget_exhausted_requests_unscored"] += 1
    return None


def replay(results, fixture_archive):
    """Recompute numeric results for intact evidence without querying an editor."""
    manifest = json.loads(safe_read(results, "manifest.json"))
    file_names = {"README.md", "protocol.md", "cells.csv", "summary.json", "config.json", "candidates.jsonl"}
    require(manifest["schema_version"] == 1 and set(manifest["data_file_sha256"]) == file_names,
            "unexpected public file manifest")
    require(_source_provenance()["source_sha256"] == _hash(manifest["source_sha256"]),
            "installed evaluator differs from the frozen source")
    files = {name: safe_read(results, name) for name in file_names}
    require(all(sha256(data) == _hash(manifest["data_file_sha256"][name]) for name, data in files.items()),
            "public result file checksum mismatch")
    config, summary = (json.loads(files[f"{name}.json"]) for name in ("config", "summary"))
    jobs = config["job_order"]
    require(config["schema_version"] == 1 and config["task"] == TASK and tuple(config["methods"]) == ARMS
            and tuple(config["conditions"]) == FORMS and len(jobs) == len(set(jobs)) == 5
            and set(config["jobs"]) == set(jobs), "invalid public replay configuration")
    expected = []
    for job in jobs:
        pair = config["jobs"][job]
        seed = pair["run_seed"]
        require(type(pair["fixture_seed"]) is int and type(seed) is int and 0 <= pair["fixture_seed"] < 2**32
                and 0 <= seed < 2**32 and job == f"fixture-{pair['fixture_seed']}", "unexpected replay job identity")
        expected.extend({
            "job": job, "fixture_seed": pair["fixture_seed"], "task": TASK, "seed": seed,
            "condition": form, "method": arm, "run_id": f"{TASK}.s{seed}.{form}.{arm}",
        } for form in FORMS for arm in ARMS)
    rows = summary["cells"]
    require([take(row, CELL) for row in rows] == expected and summary["planned_cells"] == len(expected) == 60,
            "summary deviates from the public panel plan")
    require(all(row["status"] in STATUS for row in rows)
            and summary["counts"] == {key: sum(row["status"] == key for row in rows) for key in STATUS},
            "public cell status counts differ")
    require(files["cells.csv"] == _csv(rows, config["jobs"]) and summary["groups"] == _groups(rows)
            and summary["contrasts"] == _contrasts(jobs, rows, config["bootstrap"]),
            "released statistical evidence is inconsistent")
    by_run = {row["run_id"]: {} for row in rows}
    for line in files["candidates.jsonl"].splitlines():
        candidate = json.loads(line)
        group = by_run[candidate["run_id"]]
        require(candidate["candidate_id"] not in group, "duplicate public candidate evidence")
        group[candidate["candidate_id"]] = candidate
    limits = EditLimits(**config["edit_limits"])
    members = _public_fixture_archive(fixture_archive, manifest["fixture_archive"], jobs)
    counts = dict.fromkeys((
        "development_scores_replayed", "heldout_scores_replayed", "structural_rejections_replayed",
        "structural_rejections_without_safe_recipe", "candidate_evaluator_failures_unscored",
        "reserved_without_terminal_result", "unreserved_without_terminal_result", "budget_exhausted_requests_unscored",
        "invalid_cell_candidate_records_skipped", "invalid_cell_recorded_heldouts_skipped", "cells_without_heldout_score",
    ), 0)
    with TemporaryDirectory(prefix="weightspace-fixed-replay-") as work:
        root = Path(work)
        for name, data in members.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for job in jobs:
            pair = config["jobs"][job]
            task = load_task(
                root / "fixtures" / job, TASK, seed=pair["fixture_seed"], split_seed=pair["run_seed"],
                dev_per_region=config["dev_per_region"], heldout_per_region=config["heldout_per_region"],
            )
            require(task.dev.digest() == pair["dev_sha256"] and task.heldout.digest() == pair["heldout_sha256"],
                    "replayed split identity differs from the release")
            transformed, transform = _transform_policy(task.policy, task.spec.sizes, FORMS[1], pair["run_seed"])
            require(transform["operations"] == pair["transform"]["operations"], "replayed transform differs")
            policies = dict(zip(FORMS, (task.policy, transformed), strict=True))
            require({form: policy_digest(policy) for form, policy in policies.items()} == pair["starting_policy_sha256"],
                    "replayed starting checkpoint differs")
            for row in (row for row in rows if row["job"] == job):
                records = by_run[row["run_id"]]
                require(Counter(record["result"]["status"] for record in records.values() if record["result"])
                        == row["candidate_result_counts"], "public candidate statuses differ")
                if row["heldout"] is None:
                    counts["cells_without_heldout_score"] += 1
                if row["status"] == "invalid":
                    counts["invalid_cell_candidate_records_skipped"] += len(records)
                    counts["invalid_cell_recorded_heldouts_skipped"] += row["heldout"] is not None
                    continue
                form = row["condition"]
                if row["starting_policy_sha256"] is not None:
                    require(row["starting_policy_sha256"] == pair["starting_policy_sha256"][form], "cell starting digest differs")
                base = policies[form]
                scored = {identifier: _replay_candidate(task, base, limits, record, counts)
                          for identifier, record in records.items()}
                if row["heldout"] is None:
                    require(row["status"] != "completed", "completed cell lacks recorded held-out score")
                    continue
                require(scored.get(row["candidate_id"]) == (row["dev"], row["selected_ops"], row["selected_policy_sha256"]),
                        "selected candidate differs from its development evidence")
                policy = apply_edits(base, row["selected_ops"], limits).policy
                score = _score_replay(task, policy, task.heldout, row["heldout"])
                require(score.rows == row["heldout_rows_used"]
                        and task.heldout.max_rows(task.spec.horizon) == row["heldout_rows_reserved"],
                        "held-out replay row accounting differs")
                counts["heldout_scores_replayed"] += 1
    return {"planned_cells": len(rows), "cell_status_counts": summary["counts"], **counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("export", help="stage public files from audited terminal private evidence")
    for flag in ("input", "private-summary", "output", "fixture-archive"):
        stage.add_argument(f"--{flag}", type=Path, required=True)
    stage.add_argument("--readme", type=Path, help="reviewed results notes to include in the checksummed release")
    offline = commands.add_parser("replay", help="verify the release and evaluate its fixed recipes offline")
    offline.add_argument("--results", type=Path, required=True)
    offline.add_argument("--fixture-archive", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "replay":
        print(json.dumps(replay(args.results, args.fixture_archive), indent=2))
    else:
        require(not os.path.lexists(args.output) and not os.path.lexists(args.fixture_archive),
                "output directory and fixture archive must be new paths")
        files, archive = export(args.input, args.private_summary, readme=args.readme)
        parent = open_directory(args.output.parent)
        try:
            os.mkdir(args.output.name, dir_fd=parent)
        finally:
            os.close(parent)
        destination = open_directory(args.output)
        try:
            for name, content in files.items():
                write_new(destination, name, content)
        finally:
            os.close(destination)
        destination = open_directory(args.fixture_archive.parent)
        try:
            write_new(destination, args.fixture_archive.name, archive)
        finally:
            os.close(destination)


if __name__ == "__main__":
    main()
