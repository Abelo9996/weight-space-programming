"""Seeded searches over the broker's four policy edit primitives."""

from dataclasses import dataclass

import numpy as np


SEARCH_METHODS = ("random", "greedy", "es")
EDITOR_METHODS = ("claude", "codex")
METHODS = ("no_edit", *SEARCH_METHODS, *EDITOR_METHODS)


def better(candidate, incumbent):
    """Prefer dev objective, then a smaller edit, then the earlier proposal."""
    if candidate.status != "scored":
        return False
    return (
        candidate.score.objective, -candidate.changed_entries, -candidate.candidate_id
    ) > (
        incumbent.score.objective, -incumbent.changed_entries, -incumbent.candidate_id
    )


@dataclass
class EditSpace:
    sizes: tuple[int, ...]
    limits: object
    rng: np.random.Generator

    def atomic(self, family, *, coordinate=False):
        hidden = family in ("scale_neuron", "zero_neuron")
        layer = int(self.rng.integers(len(self.sizes) - (2 if hidden else 1)))
        source, target = self.sizes[layer:layer + 2]
        edit = {"op": str(family), "layer": f"fc{layer}"}
        neuron = int(self.rng.integers(target))
        if family == "zero_neuron":
            return [{**edit, "neuron": neuron}]
        if family == "scale_neuron":
            return [
                {**edit, "neuron": neuron, "factor": factor}
                for factor in self.magnitudes(
                    (0.5, 0.8, 1.25, 2), coordinate,
                    minimum=self.limits.min_scale, maximum=self.limits.max_scale,
                )
            ]
        if family == "shift_bias":
            magnitudes = self.magnitudes(
                (-1.0, -0.25, 0.25, 1.0), coordinate,
                minimum=-self.limits.max_bias_shift, maximum=self.limits.max_bias_shift,
            )
            return [{**edit, "neuron": neuron, "shift": shift} for shift in magnitudes]
        u = self.vector(target, coordinate)
        v = self.vector(source, coordinate)
        magnitudes = self.magnitudes(
            (-1.0, -0.25, 0.25, 1.0), coordinate,
            minimum=-self.limits.max_rank_alpha, maximum=self.limits.max_rank_alpha,
        )
        return [{**edit, "alpha": alpha, "u": u, "v": v} for alpha in magnitudes]

    def magnitudes(self, values, coordinate, *, minimum, maximum):
        values = np.clip(values, minimum, maximum).tolist()
        if coordinate:
            index = int(self.rng.integers(len(values) // 2))
            return [values[index], values[-(index + 1)]]
        return [values[int(self.rng.integers(len(values)))]]

    def vector(self, width, coordinate):
        count = 1 if coordinate else int(self.rng.integers(1, min(width, 4) + 1))
        vector = np.zeros(width)
        positions = self.rng.choice(width, count, replace=False)
        vector[positions] = min(1.0, self.limits.max_rank_vector)
        return vector.tolist()

    def sample(self):
        return self.atomic(self.rng.choice(FAMILIES))[0]

    def mutate(self, parent):
        if not parent:
            return [self.sample()]
        operation = self.rng.choice(("add", "replace", "delete"))
        index = int(self.rng.integers(len(parent)))
        if operation == "delete":
            return parent[:index] + parent[index + 1:]
        if operation == "add" and len(parent) < self.limits.max_operations:
            return parent + [self.sample()]
        return parent[:index] + [self.sample()] + parent[index + 1:]


FAMILIES = ("scale_neuron", "zero_neuron", "shift_bias", "rank_one")


def run_search(method, dev, initial, seed, *, offspring=4):
    """Return a scored candidate ID without access to evaluator data."""
    if method == "no_edit":
        return initial.candidate_id
    if method not in SEARCH_METHODS:
        raise ValueError(f"unknown search method {method!r}")
    space = EditSpace(tuple(dev.task.sizes), dev.limits, np.random.default_rng(seed))
    candidate_count = dev.budget.candidates_remaining
    attempts_remaining = 12 * candidate_count
    incumbent = initial
    incumbent_ops = []
    while dev.budget.candidates_remaining and attempts_remaining:
        if method == "random":
            count = int(space.rng.integers(1, min(space.limits.max_operations, 4) + 1))
            proposals = [[space.sample() for _ in range(count)]]
        elif method == "greedy":
            coordinate = space.atomic(space.rng.choice(FAMILIES), coordinate=True)
            prefix = incumbent_ops
            if len(prefix) == space.limits.max_operations:
                index = int(space.rng.integers(len(prefix)))
                prefix = prefix[:index] + prefix[index + 1:]
            proposals = [prefix + [edit] for edit in coordinate]
        else:
            proposals = [space.mutate(incumbent_ops) for _ in range(offspring)]
        for proposal in proposals:
            if not attempts_remaining or not dev.budget.candidates_remaining:
                break
            attempts_remaining -= 1
            candidate = dev.probe(proposal)
            if candidate.status == "budget_exhausted":
                return incumbent.candidate_id
            if better(candidate, incumbent):
                incumbent, incumbent_ops = candidate, proposal
    return incumbent.candidate_id
