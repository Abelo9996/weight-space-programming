import json
from types import SimpleNamespace

import numpy as np
import pytest

from weightspace import broker
from weightspace.artifacts import RunJournal
from weightspace.edits import EditLimits, apply_edits
from weightspace.search import COMPOUND_SEARCH_METHODS, SEARCH_METHODS, EditSpace, run_search
from weightspace.tasks import Score, StartSplit, TaskBundle, TaskSpec


class Dev:
    task = SimpleNamespace(sizes=(2, 12, 12, 3))
    limits = EditLimits(max_changed_fraction=1)
    policy = {
        f"fc{index}.{name}": np.full(shape, 0.25)
        for index, (source, target) in enumerate(zip(task.sizes, task.sizes[1:]))
        for name, shape in (("weight", (target, source)), ("bias", (target,)))
    }

    def __init__(self, candidate_cap=25, *, score=len):
        self.proposals = []
        self.budget = SimpleNamespace(candidates_remaining=candidate_cap)
        self.score = score

    def probe(self, ops):
        applied = apply_edits(self.policy, ops, self.limits)
        self.proposals.append(ops)
        self.budget.candidates_remaining -= 1
        return SimpleNamespace(
            candidate_id=len(self.proposals) - 1, status="scored", changed_entries=applied.changed_entries,
            score=SimpleNamespace(objective=self.score(ops)),
        )


@pytest.mark.parametrize("method", ("no_edit", *SEARCH_METHODS))
def test_baselines_are_seeded_relative_to_the_same_checkpoint(method):
    left, right = Dev(), Dev()
    result = run_search(method, left, left.probe([]), 3, offspring=3)
    assert result == run_search(method, right, right.probe([]), 3, offspring=3)
    assert left.proposals == right.proposals
    if method == "no_edit":
        assert result == 0 and left.proposals == [[]]
        return
    assert len(left.proposals) == 25
    assert len(left.proposals[result]) == max(map(len, left.proposals))
    if method in ("greedy", "es"):
        assert len(left.proposals[result]) > 1
        assert any(
            len(ancestor) and len(descendant) > len(ancestor) and descendant[:len(ancestor)] == ancestor
            for index, ancestor in enumerate(left.proposals) for descendant in left.proposals[index + 1:]
        )


def test_two_slots_score_one_multiedit_recipe_per_compound_method():
    proposals = []
    for method in COMPOUND_SEARCH_METHODS:
        first, second = Dev(2), Dev(2)
        assert run_search(method, first, first.probe([]), 41) == 1
        assert run_search(method, second, second.probe([]), 41) == 1
        assert first.proposals == second.proposals
        assert len(first.proposals) == 2
        proposal = first.proposals[1]
        assert 4 <= len(proposal) <= EditLimits().max_operations
        assert proposal not in proposals
        proposals.append(proposal)


def test_compound_greedy_compares_reflection_before_extending_winner(monkeypatch):
    parents = []
    coordinate_blocks = EditSpace.coordinate_blocks

    def record_blocks(self, parent):
        parents.append(parent)
        return coordinate_blocks(self, parent)

    monkeypatch.setattr(EditSpace, "coordinate_blocks", record_blocks)
    scores = iter((0, 1, 2, 3))
    dev = Dev(4, score=lambda ops: next(scores))
    assert run_search("compound_greedy", dev, dev.probe([]), 41) == 3
    first, reflected = dev.proposals[1:3]
    assert len(first) == len(reflected)
    assert first != reflected
    for edit, opposite in zip(first, reflected, strict=True):
        key = {"scale_neuron": "factor", "rank_one": "alpha", "shift_bias": "shift"}.get(edit["op"])
        if key is None:
            assert edit == opposite
        else:
            assert {**edit, key: opposite[key]} == opposite
            expected = 1 / edit[key] if key == "factor" else -edit[key]
            assert opposite[key] == pytest.approx(expected)
    assert parents == [[], reflected]


def test_compound_es_scores_siblings_before_replacement_and_handles_partial_batch(monkeypatch):
    parents = []

    def mutant(self, parent):
        parents.append(parent)
        return [{"op": "shift_bias", "layer": "fc0", "neuron": len(parents) - 1, "shift": 0.5}]

    monkeypatch.setattr(EditSpace, "compound_mutate", mutant)
    scores = iter((0, 1, 4, 2, 3, 5, 2))
    dev = Dev(7, score=lambda ops: next(scores))
    assert run_search("compound_es", dev, dev.probe([]), 41, offspring=4) == 5
    assert len(dev.proposals) == 7
    assert parents == [[], [], [], [], *([dev.proposals[2]] * 4)]


@pytest.mark.parametrize("method", COMPOUND_SEARCH_METHODS)
def test_compound_attempt_limit_on_unscored_rejections(method, tmp_path, monkeypatch):
    spec = TaskSpec("toy", "Toy", Dev.task.sizes, 7, "Repair", "identity", "argmax", "survive")
    dev = StartSplit([[0, 0]], [[0, 1]])
    heldout = StartSplit([[1, 0]], [[1, 1]])
    task = TaskBundle(spec, Dev.policy, dev, heldout, {})
    journal = RunJournal.create(tmp_path, {}, run_id=method)
    session = broker.BudgetBroker(
        task, candidate_cap=2, limits=EditLimits(max_changed_fraction=0, max_operations=1), journal=journal,
    ).dev
    evaluated = []

    def evaluate(spec, policy, split, charge):
        evaluated.append(split.digest())
        charge(2)
        return Score(0, 1, 1, 1, 2, 0.0, 1.0, 0.5)

    monkeypatch.setattr(broker, "evaluate_split", evaluate)
    initial = session.probe([])
    assert run_search(method, session, initial, 41) == 0
    assert session.budget == initial.budget
    assert evaluated == [dev.digest()]
    events = [json.loads(line) for line in (journal.path / "events.jsonl").read_text().splitlines()]
    assert [event["status"] for event in events if event["event"] == "candidate_result"] == [
        "scored", *(["rejected"] * 12),
    ]
