from types import SimpleNamespace

import numpy as np
import pytest

from weightspace.edits import EditLimits, apply_edits
from weightspace.search import METHODS, run_search


class Dev:
    task = SimpleNamespace(sizes=(2, 12, 12, 3))
    limits = EditLimits(max_changed_fraction=1)
    policy = {
        f"fc{index}.{name}": np.full(shape, 0.25)
        for index, (source, target) in enumerate(zip(task.sizes, task.sizes[1:]))
        for name, shape in (("weight", (target, source)), ("bias", (target,)))
    }

    def __init__(self):
        self.proposals = []
        self.budget = SimpleNamespace(candidates_remaining=25)

    def probe(self, ops):
        applied = apply_edits(self.policy, ops, self.limits)
        self.proposals.append(ops)
        self.budget.candidates_remaining -= 1
        return SimpleNamespace(
            candidate_id=len(self.proposals) - 1, status="scored", changed_entries=applied.changed_entries,
            score=SimpleNamespace(objective=len(ops)),
        )


@pytest.mark.parametrize("method", METHODS[:-1])
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
