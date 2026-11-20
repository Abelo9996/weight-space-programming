from collections import Counter
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from scripts import export_mountaincar_panel as release
from weightspace.artifacts import RunJournal, canonical_json
from weightspace.broker import BudgetBroker
from weightspace.edits import EditLimits
from weightspace.experiment import planned_cells
from weightspace.panel import TASK, _configs, _describe_pair
from weightspace.panel_analysis import _cell_evidence, _contrasts, _events
from weightspace.policies import array_digest, policy_digest
from weightspace.tasks import FIXTURE_FORMAT, TASKS


SECRET = "ACCOUNT_TAG_never_export_synthetic_229"


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_json(path, record):
    _write(path, canonical_json(record) + b"\n")


def _fixture(root, seed):
    spec = TASKS[TASK]
    policy = {}
    for layer, (inputs, outputs) in enumerate(zip(spec.sizes, spec.sizes[1:])):
        policy[f"fc{layer}.weight"] = np.zeros((outputs, inputs))
        policy[f"fc{layer}.bias"] = np.zeros(outputs)
    policy["fc2.bias"][2] = 1.0
    sets = {
        key: np.column_stack((np.linspace(-0.8, 0.2, 100), np.full(100, 0.003 * index)))
        for index, key in enumerate(("fail_train", "norm_train", "fail_eval", "norm_eval"))
    }
    sets["traj_start"] = np.array([[-0.9, -0.01]])
    demo = {"start": sets["traj_start"][0], "obs": np.zeros((1, 2)), "action": np.array([2]),
            "reward": np.array([-1.0]), "success": np.array(False)}
    directory = root / TASK
    (directory / "public").mkdir(parents=True)
    (directory / "private").mkdir()
    np.savez(directory / "public" / "policy.npz", **policy)
    np.savez(directory / "public" / "failure_trajectory.npz", **demo)
    np.savez(directory / "private" / "eval_sets.npz", **sets)
    _write_json(directory / "manifest.json", {
        "fixture_format": FIXTURE_FORMAT, "task_id": TASK, "seed": seed, "evaluator_source": spec.evaluator_source,
        "numpy_version": "2.2.6", "torch_version": "2.7.1", "policy_sha256": policy_digest(policy),
        "starts_sha256": array_digest(sets), "demo_sha256": array_digest(demo),
    })


@pytest.fixture
def mock_terminal_panel(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    repository = Path(__file__).resolve().parents[2]
    plan = json.loads((repository / "examples" / "mountaincar-panel.json").read_bytes())
    plan["fixtures"] = [{"fixture_seed": seed, "run_seed": seed + 7} for seed in range(401, 406)]
    root = tmp_path / "private"
    root.mkdir()
    _write_json(root / "plan.json", plan)
    _write(root / "protocol.md", b"# Mock frozen protocol\n")
    _write(root / "package.whl", b"mock wheel")
    with tarfile.open(root / "source.tar", mode="w", format=tarfile.PAX_FORMAT,
                      pax_headers={"comment": "a" * 40}) as archive:
        for file in release.LICENSES.values():
            data = (repository / file).read_bytes()
            member = tarfile.TarInfo(file)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    jobs = _configs(plan, root)
    pairs = {}
    for job, config in jobs.items():
        _fixture(root / config.fixtures, config.fixture_seed)
        _write_json(root / "configs" / f"{job}.json", asdict(config))
        pairs[job] = _describe_pair(root, config)
    frozen = {
        "source": {"git_commit": "a" * 40, "package": release._source_provenance()}, "fixtures": pairs,
        "files_sha256": {path.relative_to(root).as_posix(): release.sha256(path.read_bytes())
                         for path in root.rglob("*") if path.is_file()},
    }
    _write_json(root / "frozen-manifest.json", frozen)
    ledger = RunJournal.create(root, {}, run_id="execution")
    ledger.append("panel_terminal", stop={"kind": "hard_native_failure", "raw_error": SECRET})
    last_event = _events(ledger.path / "events.jsonl", "execution")[0][-1]
    rows = []
    states = ("completed", "incomplete", "invalid", "missing", "not_started", "failed")
    limits = EditLimits()
    for index, (job, config) in enumerate(jobs.items()):
        task = release.load_task(root / config.fixtures, TASK, seed=config.fixture_seed, split_seed=config.seeds[0],
                                 dev_per_region=config.dev_per_region, heldout_per_region=config.heldout_per_region)
        for ordinal, cell in enumerate(planned_cells(config)):
            events, journal_hash = [], None
            status = states[ordinal] if index == 0 and ordinal < len(states) else "not_started"
            if status in ("completed", "incomplete", "invalid", "failed"):
                directory = root / "experiments" / job / "runs"
                journal = RunJournal.create(directory, {}, run_id=cell["run_id"])
                journal.append("checkpoint", starting_policy_sha256=policy_digest(task.policy))
                broker = BudgetBroker(task, candidate_cap=2, limits=limits, journal=journal)
                initial = broker.dev.probe([])
                if status == "completed":
                    result = broker.finalize(initial.candidate_id)
                    journal.append("run_completed", **asdict(result), selected_policy_sha256=result.policy_sha256)
                elif status == "incomplete":
                    broker.dev.probe([{"op": "shift_bias", "layer": "fc2", "neuron": 0, "shift": 99.0}])
                    broker.dev.probe([{"op": "shift_bias", "layer": SECRET, "neuron": 0, "shift": 1.0}])
                    journal.append("candidate_request", candidate_id=3, ops=[])
                elif status == "invalid":
                    journal.append("candidate_request", candidate_id=1, ops=[])
                    journal.append("candidate_result", candidate_id=1, status="failed", reason=SECRET, budget=SECRET)
                else:
                    journal.append("editor_turn", call=1, attempt=1, usage={"input_tokens": 121000, "output_tokens": 21},
                                   usage_excesses={"input_tokens": {"observed": 121000, "limit": 120000}},
                                   command=["/Users/private/bin/codex", SECRET], thread_id=SECRET, prompt=SECRET,
                                   model_catalog={"descriptor": SECRET}, raw_stderr=SECRET, stream_error=SECRET,
                                   raw_stdout_sha256=release.sha256(SECRET.encode()))
                    journal.append("run_failed", reason=SECRET)
                path = journal.path / "events.jsonl"
                events, _ = _events(path, cell["run_id"])
                journal_hash = release.sha256(path.read_bytes())
            rows.append(cell | {"job": job, "fixture_seed": config.fixture_seed, "status": status,
                                "reason": "candidate evaluator failure" if status == "invalid" else SECRET,
                                "journal_sha256": journal_hash, **_cell_evidence(events)})
    raw = {
        "frozen_manifest_sha256": release.sha256((root / "frozen-manifest.json").read_bytes()),
        "execution": {"last_event": last_event}, "cells": rows, "counts": dict(Counter(row["status"] for row in rows)),
        "groups": release._groups(rows), "contrasts": _contrasts(list(jobs), rows, plan["bootstrap"]),
        "native": {
            "planned_attempt_cap": 20, "journaled_model_command_attempts": 1, "journaled_editor_turns": 1,
            "journaled_attempts_with_usage": 1,
            "sum_of_reported_usage": dict.fromkeys(release.USAGE_FIELDS, 0) | {"input_tokens": 121000, "output_tokens": 21},
            "actual_charge_usd": None,
        },
    }
    private_summary = tmp_path / "private-summary.json"
    _write_json(private_summary, raw)
    monkeypatch.setattr(release, "summarize_panel", lambda path: raw)
    return root, private_summary


def test_exports_all_states_scrubs_private_fields_and_replays_only_verified_scores(mock_terminal_panel, tmp_path):
    root, private_summary = mock_terminal_panel
    files, archive = release.export(root, private_summary)
    assert release.export(root, private_summary) == (files, archive)
    assert all(SECRET.encode() not in payload and b"/Users/private/" not in payload for payload in files.values())
    rows = list(csv.DictReader(io.StringIO(files["cells.csv"].decode())))
    summary = json.loads(files["summary.json"])
    assert len(rows) == 60
    assert rows[2]["reason_code"] == "candidate_evaluator_failure"
    assert rows[5]["reason_code"] == "run_failed"
    assert summary["counts"] == {
        "completed": 1, "incomplete": 1, "invalid": 1, "missing": 1, "failed": 1, "not_started": 55,
    }
    assert summary["contrasts"]["codex_minus_velocity_controller"]["pooled"] is None
    assert summary["native"]["sum_of_reported_usage"]["input_tokens"] == 121000
    assert summary["cells"][5]["editor_attempt_records"][0]["stdout_sha256"] == release.sha256(SECRET.encode())
    records = [json.loads(line) for line in files["candidates.jsonl"].splitlines()]
    rejected = [record for record in records if record["result"] and record["result"]["status"] == "rejected"]
    assert [row["request_ops"] for row in rejected] == [
        [{"op": "shift_bias", "layer": "fc2", "neuron": 0, "shift": 99.0}], None,
    ]
    public = tmp_path / "public"
    for name, data in files.items():
        _write(public / name, data)
    archive_path = tmp_path / release.ARCHIVE_NAME
    archive_path.write_bytes(archive)
    output = release.replay(public, archive_path)
    assert output == {
        "planned_cells": 60, "cell_status_counts": summary["counts"], "development_scores_replayed": 3,
        "heldout_scores_replayed": 1, "structural_rejections_replayed": 1,
        "structural_rejections_without_safe_recipe": 1, "candidate_evaluator_failures_unscored": 0,
        "reserved_without_terminal_result": 0, "unreserved_without_terminal_result": 1,
        "budget_exhausted_requests_unscored": 0, "invalid_cell_candidate_records_skipped": 2,
        "invalid_cell_recorded_heldouts_skipped": 0, "cells_without_heldout_score": 59,
    }
    archive_path.write_bytes(archive + b"changed")
    with pytest.raises(ValueError, match="archive checksum"):
        release.replay(public, archive_path)


def test_export_rejects_a_different_summary_or_nonterminal_ledger(mock_terminal_panel):
    root, private_summary = mock_terminal_panel
    private_summary.write_bytes(b"{}")
    with pytest.raises(ValueError, match="provided private summary"):
        release.export(root, private_summary)
    (root / "execution" / "events.jsonl").write_bytes(b"")
    with pytest.raises(ValueError, match="terminal state"):
        release.export(root, private_summary)
