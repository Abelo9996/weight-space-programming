from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest

from weightspace import broker, codex_editor, experiment
from weightspace.analysis import _bootstrap, summarize
from weightspace.editors import ProcessResult
from weightspace.policies import policy_digest
from weightspace.tasks import Score, StartSplit, TaskBundle, TaskSpec


TASK_ID = "TR02-CTRL-002"


def stub_task(fixtures, task_id, *, seed, split_seed, **kwargs):
    spec = TaskSpec(task_id, "Toy", (2, 4, 2), 7, "Repair", "identity", "argmax", "survive")
    rng = np.random.default_rng(seed)
    weights = {
        "fc0.weight": rng.normal(size=(4, 2)), "fc0.bias": rng.normal(size=4),
        "fc1.weight": rng.normal(size=(2, 4)), "fc1.bias": rng.normal(size=2),
    }
    dev = StartSplit(np.array([[split_seed, 0]]), np.array([[split_seed, 1]]))
    heldout = StartSplit(np.array([[500, 0]]), np.array([[500, 1]]))
    return TaskBundle(spec, weights, dev, heldout, {
        "fixture_format": "weightspace-v1", "policy_sha256": policy_digest(weights),
        "dev_sha256": dev.digest(), "heldout_sha256": heldout.digest(), "evaluator_source": "test",
    })


def stub_score(spec, policy, split, charge):
    charge(7)
    failure = int(split.failure[0, 0] < 500)
    return Score(failure, 1, 1, 1, 7, float(failure), 1, (failure + 1) / 2)


def config(**changes):
    base = experiment.ExperimentConfig(
        phase="smoke", fixtures=".", fixture_source="generated", fixture_seed=0,
        tasks=(TASK_ID,), seeds=(32, 91), conditions=("original",),
        methods=("no_edit", "random", "greedy", "es"), candidate_cap=5,
        dev_per_region=1, heldout_per_region=1, edit_limits={"max_changed_fraction": 1},
    )
    return replace(base, **changes)


def test_paired_experiment_keeps_artifacts_and_reports_missing_run(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(experiment, "load_task", stub_task)
    monkeypatch.setattr(broker, "evaluate_split", stub_score)
    settings = config(conditions=("original", "permutation_relu_scale"))
    output = tmp_path / "paired"
    result = experiment.run_experiment(settings, tmp_path, output)
    assert result["completed"] == result["expected"] == 16

    for run in (output / "runs").iterdir():
        rows = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
        names = [row["event"] for row in rows]
        assert names.index("final_selected") < names.index("heldout_result")
        assert names.count("heldout_result") == 1
        assert names.index("final_selected") > max(i for i, name in enumerate(names) if name == "candidate_result")
        assert rows[-1]["budget"]["candidate_cap"] == 5
        assert rows[-1]["budget"]["row_budget"] == 70
        if run.name.endswith("greedy") or run.name.endswith("es") or run.name.endswith("random"):
            assert rows[-1]["budget"]["rows_reserved"] == 70
        if ".original." in run.name:
            assert "selected_transport" in names
        else:
            assert rows[0]["transform"]["max_abs_logit_error"] < 1e-5

    summary = summarize(output, bootstrap_replicates=50)
    assert summary["counts"]["completed"] == 16
    assert summary["groups"][0]["scores"]["objective"]["dev_minus_heldout"]["mean"] == 0.5
    assert summary["groups"][0]["scores"]["objective"]["dev_minus_heldout_interval"]["ci95"] == [0.5, 0.5]
    assert all(entry["paired_seeds"] == [32, 91] for entry in summary["heldout_comparisons"])

    (output / "runs" / f"{TASK_ID}.s32.original.random").rename(tmp_path / "moved")
    corrupted = output / "runs" / f"{TASK_ID}.s91.original.es" / "selected_policy.npz"
    corrupted.unlink()
    summary = summarize(output, bootstrap_replicates=50)
    assert summary["counts"] == {"completed": 14, "failed": 0, "incomplete": 0, "missing": 1, "invalid": 1}
    comparison = next(
        entry for entry in summary["heldout_comparisons"]
        if entry["comparison"] == "original_minus_transformed" and entry["cells"][0]["method"] == "random"
    )
    assert comparison["metrics"]["objective"]["ci95"] is None
    assert comparison["paired_seeds"] == [91]
    bad_manifest = output / "runs" / f"{TASK_ID}.s91.original.greedy" / "manifest.json"
    bad_manifest.write_text("[]", encoding="utf-8")
    assert summarize(output)["counts"]["invalid"] == 2


def test_mislabeled_fixture_is_recorded_as_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "load_task", stub_task)
    settings = config(methods=("no_edit",), seeds=(7,), fixture_source="public_fixture")
    output = tmp_path / "failed"
    assert experiment.run_experiment(settings, tmp_path, output)["failed"] == 1
    summary = summarize(output)
    assert summary["counts"]["failed"] == 1
    assert summary["availability"][0]["reason"] == "public_fixture source requires fixture format weightbench"


def test_editor_config_requires_live_opt_in_before_output(tmp_path):
    settings = config(methods=("claude",), editor={
        "model": "claude-haiku-5-5", "billing_assumption": "extra_usage_disabled",
    })
    with pytest.raises(ValueError, match="allow-live-editor"):
        experiment.run_experiment(settings, tmp_path, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_editor_bound_counts_cli_launches_and_reports_thresholds(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "load_task", stub_task)
    monkeypatch.setattr(broker, "evaluate_split", stub_score)
    monkeypatch.setattr(experiment, "run_editor", lambda dev, initial, editor: initial.candidate_id)
    settings = config(methods=("claude", "no_edit"), editor={
        "model": "claude-haiku-5-5", "billing_assumption": "extra_usage_disabled",
        "max_calls": 3, "max_api_equivalent_usd_per_call": 0.2,
    })
    output = tmp_path / "run"
    assert experiment.run_experiment(settings, tmp_path, output, allow_live=True)["completed"] == 4
    data = json.loads((output / "experiment.json").read_text())
    bounds = data["live_editor_launch_bounds"]
    assert bounds["editor_cli_launches_upper_bound"] == 6
    assert bounds["sum_of_cli_api_equivalent_thresholds_usd"] == pytest.approx(1.2)
    assert bounds["actual_charge_usd"] is None


@pytest.mark.parametrize("hard_failure", [False, True])
def test_codex_edit_experiment_keeps_budget_and_stops_on_hard_cli_failure(tmp_path, monkeypatch, hard_failure):
    monkeypatch.setattr(experiment, "load_task", stub_task)
    monkeypatch.setattr(broker, "evaluate_split", stub_score)
    launches = []
    launch_catalogs = []
    operation = {"op": "shift_bias", "layer": "fc0", "neuron": 0, "shift": 0.5}
    model = {"slug": "gpt-6.2-sol", "tool_mode": "code_mode_only", "other_field": {"setting": [7, "kept"]}}
    model_source = tmp_path / "local_model_descriptor.json"
    model_source.write_text(json.dumps(model))
    source_hash = sha256(model_source.read_bytes()).hexdigest()

    def invoke(command, *, prompt, **kwargs):
        if command[1] == "--version":
            return ProcessResult("codex-cli 0.169.0", "", 0, None)
        if command[1] == "login":
            return ProcessResult("", "Logged in using ChatGPT", 0, None)
        catalog_flag = next(flag for flag in command if flag.startswith("model_catalog_json="))
        launch_catalogs.append(json.loads(catalog_flag.split("=", 1)[1]))
        data = json.loads(prompt)
        launches.append(data)
        if hard_failure:
            return ProcessResult('{"type":"turn.failed"}\n', "rate limit", 1, None)
        probe = len(data["history"]) == 1
        response = {"action": "probe", "ops": [operation]} if probe else {"action": "select", "candidate_id": 1}
        events = [
            {"type": "thread.started", "thread_id": "019ac000-0000-7000-8000-000000000001"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": json.dumps(response)}},
            {"type": "turn.completed", "usage": {
                "input_tokens": 1200, "cached_input_tokens": 0, "output_tokens": 10,
                "reasoning_output_tokens": 0,
            }},
        ]
        return ProcessResult("\n".join(map(json.dumps, events)), "", 0, None)

    monkeypatch.setattr(codex_editor, "run_process", invoke)
    settings = config(methods=("codex", "random"), candidate_cap=2, editor={
        "model": "gpt-6.2-sol", "cli_version": "codex-cli 0.169.0",
        "billing_assumption": "subscription_only", "max_calls": 2,
        "model_catalog_source": str(model_source), "model_catalog_source_sha256": source_hash,
    })
    output = tmp_path / "codex"
    with pytest.raises(ValueError, match="allow-live-editor"):
        experiment.run_experiment(settings, tmp_path, output)
    result = experiment.run_experiment(settings, tmp_path, output, allow_live=True)
    manifest = json.loads((output / "experiment.json").read_text())
    bounds = manifest["live_editor_launch_bounds"]
    catalog = bounds["model_catalog"]
    archive = output / catalog["catalog_file"]
    assert sha256(archive.read_bytes()).hexdigest() == catalog["catalog_sha256"]
    assert json.loads(archive.read_text()) == {"models": [{**model, "tool_mode": "direct"}]}
    assert (output / catalog["source_file"]).read_bytes() == model_source.read_bytes()
    assert catalog["source_sha256"] == source_hash
    assert launch_catalogs and set(launch_catalogs) == {str(archive.resolve())}
    model_source.unlink()
    assert bounds["editor_cli_launches_upper_bound"] == 4
    assert "sum_of_cli_api_equivalent_thresholds_usd" not in bounds
    rows = [
        json.loads(line) for line in
        (output / "runs" / f"{TASK_ID}.s32.original.codex" / "events.jsonl").read_text().splitlines()
    ]
    summary = summarize(output)
    if hard_failure:
        assert result == {"phase": "smoke", "expected": 4, "completed": 0, "failed": 1, "not_started": 3}
        assert len(launches) == 1
        assert rows[-1]["event"] == "run_failed"
        assert rows[-2]["event"] == "broker_aborted"
        assert rows[-2]["budget"]["candidates_used"] == 1
        assert all(row["event"] != "heldout_result" for row in rows)
        assert summary["counts"]["failed"] == 1 and summary["counts"]["missing"] == 3
    else:
        assert result["completed"] == 4 and len(launches) == 4
        assert rows[-1]["ops"] == [operation]
        assert rows[-1]["budget"]["candidates_used"] == 2
        assert rows[-1]["budget"]["rows_used"] == 14
        assert rows[-1]["budget"]["rows_reserved"] == 28
        assert rows[-1]["heldout_rows_used"] == 7
        assert [data["candidate_slots_remaining"] for data in launches] == [1, 0, 1, 0]
        assert all("heldout" not in json.dumps(data) for data in launches)
        assert summary["counts"]["completed"] == 4
        assert any(entry["comparison"] == "editor_minus_search" for entry in summary["heldout_comparisons"])
    archive.write_text('{"models":[]}')
    with pytest.raises(ValueError, match="pinned Direct-mode derivative"):
        summarize(output)


def test_bootstrap_samples_seed_pairs_instead_of_independent_arms():
    matrix = np.array([[0.1, 0.0], [0.8, 0.7], [1.0, 0.9]])
    result = _bootstrap(matrix, lambda rows: (rows[:, :, 0] - rows[:, :, 1]).mean(axis=1), 200, 4)
    assert result["n_pairs"] == 3
    assert result["estimate"] == pytest.approx(0.1)
    assert result["ci95"] == pytest.approx([0.1, 0.1])
