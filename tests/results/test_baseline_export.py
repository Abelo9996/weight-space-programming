"""Checks of the public export using only its text files."""

import csv
import importlib.util
import io
import json
from collections import defaultdict
from pathlib import Path
import re
from statistics import mean, stdev
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/baseline-pilot-2026-11-19"
SPEC = importlib.util.spec_from_file_location("baseline_export", ROOT / "scripts/export_baseline_results.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


@pytest.fixture(scope="module")
def records():
    with (RESULTS / "cells.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows, json.loads((RESULTS / "summary.json").read_text()), json.loads((RESULTS / "manifest.json").read_text())


def test_export_drops_unreviewed_event_fields():
    record = {
        "event": "candidate_request", "run_id": "sample", "sequence": 2, "candidate_id": 0,
        "timestamp": "2026-11-19T21:48:32+00:00", "private_file": "omitted",
        "ops": [{"op": "shift_bias", "layer": "fc0", "neuron": 1, "shift": 0.5, "other": "omitted"}],
    }
    projected = EXPORT.project_event(record)
    assert "private_file" not in projected
    assert projected["ops"] == [{"op": "shift_bias", "layer": "fc0", "neuron": 1, "shift": 0.5}]


def test_input_rejects_symlinks_and_traversal(tmp_path):
    directory = tmp_path.resolve()
    source = directory / "source"
    source.mkdir()
    file = source / "allowed"
    file.write_bytes(b"content")
    (source / "link").symlink_to(file)
    (directory / "alias").symlink_to(source, target_is_directory=True)
    assert EXPORT.safe_read(source, "allowed") == b"content"
    for root, path in ((source, "link"), (directory, "alias/allowed"), (directory / "alias", "allowed")):
        with pytest.raises(OSError):
            EXPORT.safe_read(root, path)
    for path in ("../allowed", file.as_posix()):
        with pytest.raises(ValueError, match="relative path"):
            EXPORT.safe_read(source, path)


def test_fixture_archive_identity_and_allowlist(tmp_path):
    root = tmp_path.resolve()
    expected = {}
    for task in EXPORT.TASKS:
        folder = root / "fixtures" / task
        for name in EXPORT.FIXTURE_FILES:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            generation = dict.fromkeys(EXPORT.GENERATION, 1) | {
                "task_id": task, "fixture_format": "weightspace-v1",
            }
            data = EXPORT.canonical(generation) if name == "manifest.json" else name.encode()
            path.write_bytes(data)
            expected[path.relative_to(root).as_posix()] = data
        (folder / "private" / "reference_delta.npz").write_bytes(b"excluded")
    licenses = {
        "LICENSE": "source/LICENSE",
        "LICENSE-WeightBench": "source/src/weightspace/_vendor/WEIGHTBENCH_LICENSE",
        "LICENSE-Gymnasium": "source/src/weightspace/_vendor/GYMNASIUM_LICENSE",
    }
    for name, file in licenses.items():
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        expected[name] = name.encode()

    payload, fixtures = EXPORT.fixture_archive(root)
    assert payload == EXPORT.fixture_archive(root)[0]
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert set(archive.getnames()) == set(expected) | {"ATTRIBUTION.md", "SHA256SUMS"}
        assert all(member.isfile() and member.mtime == 0 and member.uid == member.gid == 0 for member in archive)
        for name, data in expected.items():
            assert archive.extractfile(name).read() == data
        for line in archive.extractfile("SHA256SUMS").read().decode().splitlines():
            digest, name = line.split("  ", 1)
            assert EXPORT.sha256(archive.extractfile(name).read()) == digest
    for task, spec in fixtures.items():
        for name, digest in spec["file_sha256"].items():
            assert digest == EXPORT.sha256(expected[f"fixtures/{task}/{name}"])


def test_per_cell_outcomes_and_budget_identity(records):
    rows, summary, manifest = records
    for name, digest in manifest["data_file_sha256"].items():
        data = (RESULTS / name).read_bytes()
        assert EXPORT.sha256(data) == digest
        assert EXPORT.FORBIDDEN_TEXT.search(data.decode()) is None
    assert len(rows) == len({row["run_id"] for row in rows}) == summary["counts"]["completed"] == 120
    assert {key: sum(int(row[key]) for row in rows) for key in manifest["budget_totals"]} == manifest["budget_totals"]
    dev_rows, candidate_counts, completions, checkpoints = defaultdict(int), defaultdict(int), {}, {}
    with (RESULTS / "events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            key = event["run_id"]
            kind = event["event"]
            if kind == "checkpoint":
                checkpoints[key] = event
            elif kind == "candidate_result":
                assert event["status"] == "scored"
                candidate_counts[key] += 1
                dev_rows[key] += event["score"]["rows"]
            elif kind == "run_completed":
                completions[key] = event
    assert sum(candidate_counts.values()) == manifest["counts"]["development_candidates_scored"] == 2280
    for row, availability in zip(rows, summary["availability"], strict=True):
        run_id = row["run_id"]
        for field in ("run_id", "task", "method", "condition", "status"):
            assert row[field] == availability[field]
        completion = completions[run_id]
        assert int(row["seed"]) == availability["seed"]
        assert int(row["selected_candidate_id"]) == completion["candidate_id"]
        assert candidate_counts[run_id] == int(row["dev_candidates_used"])
        assert dev_rows[run_id] == int(row["dev_rows_used"])
        assert completion["heldout_rows_used"] == int(row["heldout_rows_used"])
        for field in ("dev_sha256", "heldout_sha256"):
            assert row[field] == checkpoints[run_id]["provenance"][field]
        for field in ("starting_policy_sha256", "selected_policy_sha256"):
            assert row[field] == completion[field]
        for split in ("dev", "heldout"):
            score = completion[split]
            for region in ("failure", "normal"):
                wins, count = (int(row[f"{split}_{region}_{item}"]) for item in ("wins", "count"))
                assert wins / count == score[f"{region}_success"] == float(row[f"{split}_{region}_success"])
            assert mean([score["failure_success"], score["normal_success"]]) == float(row[f"{split}_objective"])


def test_rederive_group_summaries_and_paired_intervals(records):
    rows, summary, _ = records
    cells = {(row["task"], row["condition"], row["method"], int(row["seed"])): row for row in rows}
    seeds = [41, 43, 47, 53, 59]
    bootstrap = summary["bootstrap"]
    rng = np.random.default_rng(bootstrap["seed"])
    draws = rng.integers(len(seeds), size=(bootstrap["replicates"], len(seeds)))

    def scores(task, condition, method, split, metric):
        return [float(cells[task, condition, method, seed][f"{split}_{metric}"]) for seed in seeds]

    def check_stats(output, values):
        assert output == pytest.approx({
            "n": len(values), "mean": mean(values), "sd": stdev(values), "minimum": min(values), "maximum": max(values),
        })

    def check_interval(output, differences):
        assert output["n_pairs"] == len(seeds)
        assert output["estimate"] == pytest.approx(mean(differences), abs=1e-14)
        bounds = np.quantile(np.asarray(differences)[draws].mean(axis=1), [0.025, 0.975])
        assert output["ci95"] == pytest.approx(bounds, abs=1e-14)

    assert len(summary["groups"]) == 24
    for group in summary["groups"]:
        identity = [group[field] for field in ("task", "condition", "method")]
        assert group["completed_seeds"] == seeds
        for metric, published in group["scores"].items():
            dev = scores(*identity, "dev", metric)
            heldout = scores(*identity, "heldout", metric)
            drop = np.subtract(dev, heldout)
            check_stats(published["dev"], dev)
            check_stats(published["heldout"], heldout)
            check_stats(published["dev_minus_heldout"], drop)
            check_interval(published["dev_minus_heldout_interval"], drop)

    assert len(summary["heldout_comparisons"]) == 30
    for paired in summary["heldout_comparisons"]:
        assert paired["paired_seeds"] == seeds
        assert paired["excluded_seeds"] == []
        one, two = paired["cells"]
        for metric, published in paired["metrics"].items():
            first = scores(paired["task"], one["condition"], one["method"], "heldout", metric)
            second = scores(paired["task"], two["condition"], two["method"], "heldout", metric)
            check_interval(published, np.subtract(first, second))


def test_all_arm_figure_contains_all_seed_markers(records):
    _, summary, _ = records
    root = ET.parse(RESULTS / "heldout-objective.svg").getroot()
    groups = {node.get("id"): node for node in root.iter("{http://www.w3.org/2000/svg}g")}
    for group in summary["groups"]:
        identity = "-".join(group[field] for field in ("task", "method", "condition"))
        assert f"{identity}-mean" in groups
        assert len(list(groups[f"{identity}-seeds"].iter("{http://www.w3.org/2000/svg}use"))) == 5


def test_readme_table_matches_summary_at_display_precision(records):
    _, summary, _ = records
    tasks = dict(zip(("MountainCar", "Pendulum", "CartPole"), EXPORT.TASKS, strict=True))
    methods = dict(zip(("No edit", "Random", "Greedy", "Evolution"), EXPORT.METHODS, strict=True))
    groups = {(row["task"], row["method"], row["condition"]): row["scores"]["objective"] for row in summary["groups"]}
    contrasts = {(row["task"], row["cells"][0]["method"]): row["metrics"]["objective"]
                 for row in summary["heldout_comparisons"] if row["comparison"] == "original_minus_transformed"}
    covered = set()
    for line in (RESULTS / "README.md").read_text().splitlines():
        parts = [value.strip() for value in line.split("|")[1:-1]]
        if len(parts) != 7 or parts[0] not in tasks:
            continue
        task, method = tasks[parts[0]], methods[parts[1]]
        covered.add((task, method))
        original, transformed = (groups[task, method, condition] for condition in EXPORT.CONDITIONS)
        paired = contrasts[task, method]
        observed = list(map(float, re.findall(r"[+-]?[0-9]+\.[0-9]+", " ".join(parts[2:]))))
        expected = [original["heldout"]["mean"], transformed["heldout"]["mean"], paired["estimate"],
                    *paired["ci95"], original["dev_minus_heldout"]["mean"], transformed["dev_minus_heldout"]["mean"]]
        assert observed == pytest.approx(expected, abs=0.000500000001)
    assert len(covered) == 12
