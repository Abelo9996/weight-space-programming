from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from weightspace import panel, panel_analysis


@pytest.fixture
def frozen_panel(tmp_path):
    plan = json.loads((Path(__file__).resolve().parents[2] / panel.PLAN_PATH).read_bytes())
    jobs = panel._configs(plan, tmp_path)
    panel._write_json(tmp_path / "plan.json", plan)
    panel._write_json(tmp_path / "frozen-manifest.json", {
        "schema_version": 1,
        "files_sha256": {"plan.json": panel._digest(tmp_path / "plan.json")},
        "fixtures": dict.fromkeys(jobs, {}),
    })
    return tmp_path, plan, list(jobs)


def test_panel_plans_twenty_calls_and_claims_the_run_once(frozen_panel, monkeypatch):
    root, plan, job_ids = frozen_panel
    jobs = panel._configs(plan, root)
    assert sum(len(panel.planned_cells(config)) for config in jobs.values()) == 60
    assert all(config.editor_options().max_observed_output_tokens == 8192 for config in jobs.values())
    assert all(config.editor_options().retries == 0 for config in jobs.values())
    monkeypatch.setattr(panel, "_verify_for_launch", lambda *args: None)
    executed = []

    def run(config, fixtures, output, *, allow_live):
        assert allow_live
        assert fixtures == root / config.fixtures
        assert output == root / "experiments" / output.name
        assert (root / "execution" / "manifest.json").is_file()
        executed.append(output.name)
        if len(executed) == 2:
            return {"expected": 12, "completed": 11, "failed": 1, "not_started": 0}
        return {"expected": 12, "completed": 12, "failed": 0}

    monkeypatch.setattr(panel, "run_experiment", run)
    with pytest.raises(ValueError, match="allow-live-editor"):
        panel.run_panel(root)
    assert executed == [] and not (root / "execution").exists()
    result = panel.run_panel(root, allow_live=True)
    assert executed == job_ids[:2]
    assert result == {
        "expected": 60, "completed": 23, "failed": 1, "not_started": 36,
        "stop": {"kind": "hard_native_failure", "job": job_ids[1]},
    }
    with pytest.raises(FileExistsError, match="launch claim"):
        panel.run_panel(root, allow_live=True)
    assert executed == job_ids[:2]


def test_panel_counts_interrupted_and_unstarted_cells(frozen_panel):
    root, plan, job_ids = frozen_panel
    config = panel._configs(plan, root)[job_ids[0]]
    (root / "configs").mkdir()
    panel._write_json(root / "configs" / f"{job_ids[0]}.json", asdict(config))
    execution = panel.RunJournal.create(root, {}, run_id="execution")
    execution.append("fixture_started", job=job_ids[0])
    experiment = root / "experiments" / job_ids[0]
    cells = panel.planned_cells(config)
    cell = cells[5]
    journal = panel.RunJournal.create(experiment / "runs", {}, run_id=cell["run_id"])
    journal.append("editor_turn", command=["fake"], usage={"input_tokens": 11})
    with (journal.path / "events.jsonl").open("ab") as handle:
        handle.write(b'{"sequence":1}\n')
    execution.append("panel_interrupted", job=job_ids[0])
    summary = panel_analysis.summarize_panel(root)
    assert summary["counts"] == {
        "completed": 0, "failed": 0, "incomplete": 0, "invalid": 1, "missing": 5, "not_started": 54,
    }
    assert len(summary["cells"]) == summary["planned_cells"] == 60
    assert summary["cells"][5]["status"] == "invalid"
    assert summary["cells"][5]["journal_sha256"]
    assert summary["native"]["journaled_model_command_attempts"] == 1
    assert summary["native"]["sum_of_reported_usage"]["input_tokens"] == 11
    assert summary["execution"]["last_event"]["event"] == "panel_interrupted"


def test_paired_analysis_reselects_only_by_dev_and_withholds_incomplete_contrasts(frozen_panel):
    _, plan, jobs = frozen_panel
    available = []
    for index, job in enumerate(jobs):
        for form in panel.FORMS:
            scores = {
                "codex": (0.75, 0.8), "velocity_controller": (0.75, 0.6 if form == "original" else 0.8),
                "compound_random": (1 if index < 3 else 0.5, 0.5),
                "compound_greedy": (0.75, 0.9), "compound_es": (0.5, 1.0),
            }
            for arm, (dev, heldout) in scores.items():
                available.append({
                    "job": job, "condition": form, "method": arm, "status": "completed",
                    "dev": {"objective": dev}, "heldout": {"objective": heldout},
                })
    contrasts = panel_analysis._contrasts(jobs, available, plan["bootstrap"])
    primary = contrasts["codex_minus_velocity_controller"]
    assert primary["pooled"]["estimate"] == pytest.approx(0.1)
    assert primary["by_form"]["original"]["estimate"] == pytest.approx(0.2)
    assert primary["by_form"][panel.FORMS[1]]["estimate"] == 0
    search = contrasts["codex_minus_dev_selected_compound"]
    assert search["development_winner"] == "compound_random"
    assert search["pooled"]["estimate"] == pytest.approx(0.3)
    assert search["pooled"]["ci95"] == pytest.approx([-0.1, 0.3])
    assert np.allclose([fixture["pooled"] for fixture in search["per_fixture"]], 0.3)

    available.pop(0)
    partial = panel_analysis._contrasts(jobs, available, plan["bootstrap"])["codex_minus_velocity_controller"]
    assert partial["paired_fixtures"] == 4
    assert partial["pooled"] is None and partial["by_form"] is None
    assert partial["per_fixture"][0]["differences"] == {"original": None, panel.FORMS[1]: 0.0}
