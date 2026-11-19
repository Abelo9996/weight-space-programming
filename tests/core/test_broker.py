import json

import numpy as np
import pytest

import weightspace.broker as broker_module
import weightspace.tasks as tasks_module
from weightspace.artifacts import RunJournal
from weightspace.broker import BudgetBroker
from weightspace.tasks import TASKS, StartSplit, TaskBundle


def toy_task():
    spec = TASKS["TR02-CTRL-002"]
    policy = {}
    for i, (inputs, outputs) in enumerate(zip(spec.sizes, spec.sizes[1:])):
        policy[f"fc{i}.weight"] = np.zeros((outputs, inputs))
        policy[f"fc{i}.bias"] = np.zeros(outputs)
    policy["fc2.bias"][2] = 1.0
    return TaskBundle(
        spec, policy, StartSplit([[-0.52, 0.0], [-0.53, 0.0]], [[0.23, 0.013]]),
        StartSplit([[-0.54, 0.0]], [[0.231, 0.018]]), {},
    )


def shift(value):
    return [{"op": "shift_bias", "layer": "fc2", "neuron": 2, "shift": value}]


def test_reservation_and_active_row_counts():
    broker = BudgetBroker(toy_task(), candidate_cap=1, row_budget=1200)
    score = broker.dev.probe([])
    assert score.status == "scored"
    assert score.score.failure_wins == 0
    assert score.score.normal_wins == 1
    assert score.score.rows == 462
    assert score.budget.rows_reserved == 600
    assert score.budget.rows_used == 462
    assert score.budget.candidates_remaining == 0
    assert broker.dev.probe([]).status == "budget_exhausted"
    row_limited = BudgetBroker(toy_task(), candidate_cap=3, row_budget=1199)
    assert row_limited.dev.probe([]).budget.candidates_remaining == 0
    assert row_limited.dev.probe([]).status == "budget_exhausted"


def test_validation_and_failed_forward_accounting(monkeypatch, tmp_path):
    run = RunJournal.create(tmp_path, {"model": None}, run_id="failed-test")
    broker = BudgetBroker(toy_task(), candidate_cap=2, journal=run)
    rejected = broker.dev.probe(shift("bad"))
    assert rejected.status == "rejected"
    assert rejected.budget.rows_reserved == rejected.budget.rows_used == 0
    observed_batches = []

    def explode(policy, obs):
        observed_batches.append(len(obs))
        raise FloatingPointError("inference failed")

    with monkeypatch.context() as patch:
        patch.setattr(tasks_module, "forward", explode)
        failed = broker.dev.probe([])
    assert failed.status == "failed"
    assert observed_batches == [2]
    assert failed.budget.rows_reserved == 600
    assert failed.budget.rows_used == 2
    with pytest.raises(ValueError, match="scored"):
        broker.finalize(failed.candidate_id)
    good = broker.dev.probe([])
    assert good.budget.rows_reserved == 1200
    assert good.budget.rows_used == 464
    assert broker.dev.probe([]).status == "budget_exhausted"
    events = [json.loads(line) for line in (run.path / "events.jsonl").read_text().splitlines()]
    assert [e["sequence"] for e in events] == list(range(len(events)))
    results = [event for event in events if event["event"] == "candidate_result"]
    assert [event["status"] for event in results] == ["rejected", "failed", "scored", "budget_exhausted"]


def test_fixed_start_and_sealed_lifecycle(monkeypatch, tmp_path):
    task = toy_task()
    calls = []
    evaluate = broker_module.evaluate_split

    def trace(spec, policy, split, charge):
        calls.append((split.digest(), policy["fc2.bias"][2]))
        return evaluate(spec, policy, split, charge)

    monkeypatch.setattr(broker_module, "evaluate_split", trace)
    run = RunJournal.create(tmp_path, {}, run_id="final-test")
    broker = BudgetBroker(task, candidate_cap=3, journal=run)
    with pytest.raises(ValueError):
        task.dev.failure.setflags(write=True)
    assert not hasattr(broker.dev, "heldout")
    checkpoint = broker.dev.policy()
    checkpoint["fc2.bias"][2] = -100
    task.policy["fc2.bias"][2] = -100
    assert broker.dev.policy()["fc2.bias"][2] == 1
    first = broker.dev.probe(shift(1))
    second = broker.dev.probe(shift(2))
    assert calls == [(task.dev.digest(), 2), (task.dev.digest(), 3)]
    before = broker.dev.budget
    final = broker.finalize(first.candidate_id)
    assert calls[-1] == (task.heldout.digest(), 2)
    assert final.dev == first.score
    assert final.budget == broker.dev.budget == before
    assert final.heldout_rows_reserved == 400
    assert final.heldout_rows_used == final.heldout.rows
    assert broker.selected_policy()["fc2.bias"][2] == 2
    for proceed in (lambda: broker.dev.probe([]), lambda: broker.finalize(second.candidate_id)):
        with pytest.raises(RuntimeError, match="sealed"):
            proceed()
    assert len(calls) == 3
    events = [json.loads(line) for line in (run.path / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events[-2:]] == ["final_selected", "heldout_result"]


def test_heldout_failure_is_terminal(monkeypatch, tmp_path):
    run = RunJournal.create(tmp_path, {})
    broker = BudgetBroker(toy_task(), candidate_cap=2, journal=run)
    candidate = broker.dev.probe([])
    calls = 0

    def fail(spec, policy, split, charge):
        nonlocal calls
        calls += 1
        charge(1)
        raise RuntimeError("held-out evaluation failed")

    monkeypatch.setattr(broker_module, "evaluate_split", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        broker.finalize(candidate.candidate_id)
    with pytest.raises(RuntimeError, match="sealed"):
        broker.finalize(candidate.candidate_id)
    assert calls == 1
    record = json.loads((run.path / "events.jsonl").read_text().splitlines()[-1])
    assert record["event"] == "heldout_failed"
    assert record["heldout_rows_used"] == 1


def test_abort_and_unique_artifacts(tmp_path):
    run = RunJournal.create(tmp_path, {"config": {"seed": 123}}, run_id="named-run")
    with pytest.raises(FileExistsError):
        RunJournal.create(tmp_path, {}, run_id=run.run_id)
    with pytest.raises(ValueError, match="run ID"):
        RunJournal.create(tmp_path, {}, run_id="../escape")
    broker = BudgetBroker(toy_task(), candidate_cap=1, journal=run)
    candidate = broker.dev.probe([])
    broker.abort("method failed")
    with pytest.raises(RuntimeError, match="sealed"):
        broker.finalize(candidate.candidate_id)
    events = [json.loads(line) for line in (run.path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "broker_aborted"
    assert not any(e["event"].startswith("heldout") for e in events)
