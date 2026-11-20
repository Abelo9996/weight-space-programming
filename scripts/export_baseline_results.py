"""Project the frozen baseline pilot into public records and a separate fixture archive."""

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile


TASKS = ("TR02-CTRL-002", "TR02-CTRL-003", "TR02-CTRL-004")
CONDITIONS = ("original", "permutation_relu_scale")
METHODS = ("no_edit", "random", "greedy", "es")
FIXTURE_FILES = ("manifest.json", "public/policy.npz", "public/failure_trajectory.npz", "private/eval_sets.npz")
CELL = ("task", "condition", "method", "seed", "run_id")
BUDGET = ("candidate_cap", "candidates_remaining", "candidates_used", "row_budget",
          "rows_per_candidate", "rows_reserved", "rows_used")
SCORE = ("failure_count", "failure_success", "failure_wins", "normal_count",
         "normal_success", "normal_wins", "objective", "rows")
METRICS = ("failure_success", "normal_success", "objective")
EDIT_LIMITS = ("max_bias_shift", "max_changed_fraction", "max_operations", "max_rank_alpha",
               "max_rank_vector", "max_scale", "min_scale")
GENERATION = ("demo_sha256", "evaluator_source", "fixture_format", "numpy_version", "policy_sha256",
              "seed", "starts_sha256", "task_id", "torch_version")
OPS_FIELDS = {
    "scale_neuron": ("neuron", "factor"),
    "zero_neuron": ("neuron",),
    "shift_bias": ("neuron", "shift"),
    "rank_one": ("alpha", "u", "v"),
}
EVENT_FIELDS = {
    "checkpoint": ("starting_policy_sha256", "provenance", "transform"),
    "broker_started": ("limits", "task_id", "dev_sha256", "budget", "starting_policy_sha256", "evaluator_source"),
    "candidate_request": ("candidate_id", "ops"),
    "candidate_reserved": ("candidate_id", "budget", "ops", "edit_sha256", "policy_sha256"),
    "candidate_result": ("candidate_id", "budget", "changed_entries", "reason", "score", "status", "touched_entries"),
    "final_selected": ("candidate_id", "ops", "policy_sha256", "dev", "budget",
                       "heldout_rows_reserved", "heldout_sha256", "edit_sha256"),
    "heldout_result": ("candidate_id", "ops", "policy_sha256", "dev", "budget",
                        "heldout_rows_reserved", "heldout_rows_used", "heldout"),
    "selected_transport": ("condition", "fixed_input_check", "selected_policy_sha256", "transported_policy_sha256"),
    "run_completed": ("candidate_id", "ops", "selected_policy_sha256", "dev", "budget",
                      "heldout_rows_reserved", "heldout_rows_used", "heldout", "starting_policy_sha256"),
}
CONFIG = ("phase", "fixtures", "fixture_source", "fixture_seed", "tasks", "seeds", "conditions", "methods",
          "candidate_cap", "dev_per_region", "heldout_per_region")
COUNTS = ("completed", "failed", "incomplete", "invalid", "missing")
STATS = ("n", "mean", "sd", "minimum", "maximum")
INTERVAL = ("n_pairs", "estimate", "ci95")
FORBIDDEN_TEXT = re.compile(
    r"(?:[/](?:Users|home|private/var)[/]|file://|[A-Za-z]:\\Users\\|[Cc]o-authored-by|[Gg]enerated by|"
    r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|BEGIN [A-Z ]*PRIVATE KEY|\u2014)",
    re.IGNORECASE,
)


def take(record, fields):
    return {field: record[field] for field in fields}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(record):
    return json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def open_directory(path):
    path = Path(path).absolute()
    require(".." not in path.parts, "invalid directory path")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
    fd = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
    except OSError:
        os.close(fd)
        raise
    return fd


def safe_read(root, relative):
    """Read a regular file beneath root without following symbolic links."""
    parts = PurePosixPath(relative).parts
    require(parts and all(part not in {"/", ".."} for part in parts), "invalid relative path")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = open_directory(root)
    try:
        for part in parts[:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, "rb") as stream:
            require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "input must be a regular file")
            return stream.read()
    finally:
        os.close(fd)


def transform(record):
    fields = ("absolute_tolerance", "fixed_inputs", "fixed_inputs_seed_stream", "max_abs_logit_error", "torch_version")
    output = take(record, fields)
    output["operations"] = [
        take(op, ("kind", "layer", {"permutation": "order", "relu_scale": "scales"}[op["kind"]]))
        for op in record["operations"]
    ]
    return output


def project_event(record):
    kind = record["event"]
    out = take(record, ("event", "run_id", "sequence", "timestamp", *EVENT_FIELDS[kind]))
    for field in ("score", "dev", "heldout"):
        if field in out:
            out[field] = take(out[field], SCORE)
    if "budget" in out:
        out["budget"] = take(out["budget"], BUDGET)
    if "limits" in out:
        out["limits"] = take(out["limits"], EDIT_LIMITS)
    if "ops" in out:
        out["ops"] = [take(op, ("op", "layer", *OPS_FIELDS[op["op"]])) for op in out["ops"]]
    if "provenance" in out:
        data = out["provenance"]
        fields = ("task_id", "seed", "split_seed", "evaluator_source", "fixture_format", "policy_sha256",
                  "dev_sha256", "heldout_sha256", "demo_sha256", "numpy_version")
        out["provenance"] = take(data, fields) | {"generation": take(data["generation"], GENERATION)}
    for field in ("transform", "fixed_input_check"):
        if out.get(field) is not None:
            out[field] = transform(out[field])
    return out


def project_summary(raw):
    result = take(raw, ("config_sha256", "fixture_source", "phase", "schema_version", "interpretation"))
    result["bootstrap"] = take(raw["bootstrap"], ("method", "minimum_pairs_for_interval", "replicates", "seed", "unit"))
    result["counts"] = take(raw["counts"], COUNTS)
    result["availability"] = [take(cell, (*CELL, "status", "reason")) for cell in raw["availability"]]
    result["groups"] = []
    for group in raw["groups"]:
        out = take(group, ("task", "condition", "method", "planned_seed_count", "completed_seeds"))
        out["scores"] = {
            metric: {field: take(group["scores"][metric][field], STATS)
                     for field in ("dev", "heldout", "dev_minus_heldout")} |
            {"dev_minus_heldout_interval": take(group["scores"][metric]["dev_minus_heldout_interval"], INTERVAL)}
            for metric in METRICS
        }
        out["budget"] = {key: take(group["budget"][key], STATS) for key in
                         ("candidate_cap", "candidates_used", "row_budget", "rows_reserved", "rows_used")}
        out["editor_calls"] = take(group["editor_calls"], STATS)
        out["per_run"] = [
            take(row, ("seed", "editor_identity")) | {"attempts": take(row["attempts"], ("scored",))}
            for row in group["per_run"]
        ]
        result["groups"].append(out)
    result["heldout_comparisons"] = [
        take(row, ("task", "comparison", "paired_seeds")) |
        {"cells": [take(cell, ("condition", "method")) for cell in row["cells"]],
         "excluded_seeds": [take(seed, ("seed", "reason")) for seed in row["excluded_seeds"]],
         "metrics": {metric: take(row["metrics"][metric], INTERVAL) for metric in METRICS}}
        for row in raw["heldout_comparisons"]
    ]
    return result


def planned_cells(config):
    require(config["tasks"] == list(TASKS) and config["conditions"] == list(CONDITIONS)
            and config["methods"] == list(METHODS), "unsupported baseline pilot panel")
    require(config["fixtures"] == "fixtures" and config["fixture_source"] == "generated",
            "expected an experiment with local generated fixtures")
    seeds = config["seeds"]
    require(len(seeds) == len(set(seeds)) == 5 and all(type(seed) is int and seed >= 0 for seed in seeds),
            "expected five nonnegative paired seeds")
    return [
        dict(task=task, seed=seed, condition=condition, method=method, run_id=f"{task}.s{seed}.{condition}.{method}")
        for task, seed, condition, method in itertools.product(TASKS, seeds, CONDITIONS, METHODS)
    ]


def read_cells(root, plan, summary, fixtures):
    by_run = summary["availability"]
    require(len(by_run) == len(plan), "availability does not cover all planned cells")
    all_events, rows = [], []
    for cell, availability in zip(plan, by_run, strict=True):
        require(take(availability, CELL) == cell, "summary run ID is inconsistent")
        require(availability["status"] == "completed" and availability["reason"] is None,
                "unexpected non-completed cell in the frozen pilot")
        events = [project_event(json.loads(line)) for line in
                  safe_read(root, f"experiment/runs/{cell['run_id']}/events.jsonl").splitlines()]
        for sequence, event in enumerate(events):
            require(event["sequence"] == sequence and event["run_id"] == cell["run_id"],
                    "event sequence or run ID is inconsistent")
        checkpoint, final = events[0], events[-1]
        require(checkpoint["event"] == "checkpoint" and final["event"] == "run_completed",
                "cell is missing its checkpoint or completion record")
        require(checkpoint["provenance"]["generation"] == fixtures[cell["task"]]["generation"],
                "cell fixture does not match archived bytes")
        all_events.extend(events)
        rows.append(cell | {"status": availability["status"], "selected_candidate_id": final["candidate_id"]} |
                    take(checkpoint["provenance"], ("dev_sha256", "heldout_sha256")) |
                    take(final, ("starting_policy_sha256", "selected_policy_sha256",
                                 "heldout_rows_reserved", "heldout_rows_used")) |
                    {"dev_" + key: value for key, value in final["budget"].items()} |
                    {part + "_" + key: value for part in ("dev", "heldout") for key, value in final[part].items()
                     if key != "rows"})
    return all_events, rows


def fixture_archive(root):
    members, fixtures = {}, {}
    for task in TASKS:
        prefix = f"fixtures/{task}"
        files = {name: safe_read(root, f"{prefix}/{name}") for name in FIXTURE_FILES}
        record = json.loads(files["manifest.json"])
        require(record == take(record, GENERATION) and record["task_id"] == task
                and record["fixture_format"] == "weightspace-v1", "unexpected task fixture manifest")
        fixtures[task] = {"generation": record, "file_sha256": {key: sha256(data) for key, data in files.items()}}
        members.update({f"{prefix}/{name}": data for name, data in files.items()})
    licenses = {
        "LICENSE": "source/LICENSE",
        "LICENSE-WeightBench": "source/src/weightspace/_vendor/WEIGHTBENCH_LICENSE",
        "LICENSE-Gymnasium": "source/src/weightspace/_vendor/GYMNASIUM_LICENSE",
    }
    members.update({name: safe_read(root, path) for name, path in licenses.items()})
    members["ATTRIBUTION.md"] = (
        "# Baseline pilot fixtures\n\n"
        "These generated policies, start panels and standalone demonstration trajectories reproduce the "
        "offline baseline pilot in Weight-Space Programming by Abel Yagubyan. The preparer uses adapted "
        "WeightBench classic-control tasks at commit `2acd60172fc9758eb9030b1cab4a2ff29be5c993` "
        "and Gymnasium dynamics. The accompanying MIT notices preserve source attribution.\n\n"
        "[Weight-Space Programming](https://github.com/Abelo9996/weight-space-programming) · "
        "[WeightBench](https://github.com/Aayush-Marasini/WeightBench) · "
        "[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)\n"
    ).encode()
    members["SHA256SUMS"] = "".join(
        f"{sha256(data)}  {name}\n" for name, data in sorted(members.items())
    ).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, data in sorted(members.items()):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue(), fixtures


def export(root):
    config_bytes = safe_read(root, "config.json")
    config = json.loads(config_bytes)
    require(config == take(config, CONFIG), "unexpected input config fields")
    plan = planned_cells(config)
    frozen = json.loads(safe_read(root, "frozen-manifest.json"))
    experiment = json.loads(safe_read(root, "experiment/experiment.json"))
    config_hash = sha256(canonical(experiment["config"]))
    require(
        sha256(config_bytes) == frozen["config_sha256"]
        and sha256(canonical(config)) == frozen["config_canonical_input_sha256"]
        and all(experiment["config"][key] == value for key, value in config.items())
        and config_hash == experiment["config_sha256"]
        and plan == experiment["cells"] and len(plan) == frozen["expected_cells"]
        and experiment["live_editor_launch_bounds"] is None,
        "experiment does not match the frozen baseline plan",
    )
    require(experiment["source"]["modules_sha256"] == frozen["source_modules_sha256"],
            "run module digests differ from frozen source")
    summary = project_summary(json.loads(safe_read(root, "summary.json")))
    require(summary["config_sha256"] == config_hash
            and summary["counts"] == {key: len(plan) if key == "completed" else 0 for key in COUNTS},
            "summary does not match the completed pilot")
    archive, fixtures = fixture_archive(root)
    events, rows = read_cells(root, plan, summary, fixtures)
    reserve = [event for event in events if event["event"] == "candidate_reserved"]
    scores = [event for event in events if event["event"] == "candidate_result"]
    require(len(reserve) == len(scores) == sum(row["dev_candidates_used"] for row in rows)
            and all(event["status"] == "scored" and event["reason"] is None for event in scores),
            "candidate totals or statuses differ from the frozen pilot")
    runtime = safe_read(root, "runtime.txt").decode().splitlines()
    wheel_path = "distribution/weightspace-0.1.0-py3-none-any.whl"
    wheel_digest = sha256(safe_read(root, wheel_path))
    require(wheel_digest == runtime[6].split()[0], "wheel digest mismatch")
    require(sha256(safe_read(root, "source.tar")) == frozen["source_archive_sha256"], "source archive digest mismatch")
    dependencies = dict(line.split("==") for line in runtime[7:]
                        if re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", line))
    require(runtime[1] == experiment["python_version"] and dependencies["numpy"] == experiment["numpy_version"]
            and all(item["generation"]["torch_version"] == dependencies["torch"] for item in fixtures.values()),
            "runtime versions do not match the experiment or fixtures")
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    files = {
        "config.json": config_bytes,
        "summary.json": (json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
        "cells.csv": csv_buffer.getvalue().encode(),
        "events.jsonl": b"".join(canonical(event) + b"\n" for event in events),
    }
    manifest = {
        "schema_version": 1,
        "source": take(frozen, ("source_commit", "source_tree", "source_archive_sha256", "source_modules_sha256",
                                "frozen_at_utc")) |
        take(experiment["source"], ("source_sha256", "distribution_version")) | {"wheel_sha256": wheel_digest},
        "config": take(frozen, ("config_sha256", "config_canonical_input_sha256")) |
        {"experiment_config_sha256": config_hash},
        "runtime": {"platform": runtime[0], "python": runtime[1], "build_tool": runtime[5],
                    "dependencies": dependencies,
                    "thread_limits": dict.fromkeys(("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"), 1)},
        "fixtures": fixtures,
        "fixture_archive": {"name": "baseline-pilot-2026-11-19-fixtures.tar.gz", "size_bytes": len(archive),
                            "sha256": sha256(archive)},
        "data_file_sha256": {name: sha256(data) for name, data in files.items()},
        "counts": summary["counts"] | {"development_candidates_scored": len(scores), "editor_calls": 0},
        "budget_totals": {key: sum(row[key] for row in rows) for key in
                          ("dev_rows_used", "dev_rows_reserved", "heldout_rows_used", "heldout_rows_reserved")},
        "max_fixed_input_logit_errors": {
            "starting": max(event["transform"]["max_abs_logit_error"] for event in events
                            if event["event"] == "checkpoint" and event["transform"] is not None),
            "selected_transport": max(event["fixed_input_check"]["max_abs_logit_error"] for event in events
                                      if event["event"] == "selected_transport"),
        },
    }
    files["manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    for name, data in files.items():
        require(not FORBIDDEN_TEXT.search(data.decode()), f"unexpected private text in {name}")
    return files, archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="directory containing the frozen pilot")
    parser.add_argument("--output", type=Path, required=True, help="new directory for public text files")
    parser.add_argument("--fixture-archive", type=Path, required=True, help="new archive in private staging")
    args = parser.parse_args()
    files, archive = export(args.run)
    parent = open_directory(args.output.parent)
    try:
        os.mkdir(args.output.name, dir_fd=parent)
    finally:
        os.close(parent)
    destination = open_directory(args.output)
    try:
        for name, data in files.items():
            write_new(destination, name, data)
    finally:
        os.close(destination)
    staging = open_directory(args.fixture_archive.parent)
    try:
        write_new(staging, args.fixture_archive.name, archive)
    finally:
        os.close(staging)


def write_new(directory_fd, name, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(name, flags, 0o644, dir_fd=directory_fd)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


if __name__ == "__main__":
    main()
