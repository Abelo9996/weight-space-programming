"""Seeded searches over the broker's four policy edit primitives."""

from dataclasses import dataclass

import numpy as np


COMPOUND_SEARCH_METHODS = ("compound_random", "compound_greedy", "compound_es")
SEARCH_METHODS = ("random", "greedy", "es", *COMPOUND_SEARCH_METHODS)
EDITOR_METHODS = ("claude", "codex")
VELOCITY_CONTROLLER = "velocity_controller"
METHODS = ("no_edit", *SEARCH_METHODS, VELOCITY_CONTROLLER, *EDITOR_METHODS)


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

    def compound_length(self):
        maximum = self.limits.max_operations
        return int(self.rng.integers(min(4, maximum), maximum + 1))

    def compound_sample(self):
        return [self.sample() for _ in range(self.compound_length())]

    def coordinate_blocks(self, parent):
        maximum = self.limits.max_operations
        count = (
            int(self.rng.integers(1, min(4, maximum) + 1)) if parent else self.compound_length()
        )
        prefix = parent[:maximum - count]
        block, reflected = list(prefix), list(prefix)
        for _ in range(count):
            pair = self.atomic(self.rng.choice(FAMILIES), coordinate=True)
            direction = int(self.rng.integers(2)) if len(pair) == 2 else 0
            block.append(pair[direction])
            reflected.append(pair[-(direction + 1)])
        return [block] if block == reflected else [block, reflected]

    def compound_mutate(self, parent):
        if not parent:
            return self.compound_sample()
        count = int(self.rng.integers(1, min(4, self.limits.max_operations) + 1))
        for _ in range(count):
            parent = self.mutate(parent) or [self.sample()]
        return parent


FAMILIES = ("scale_neuron", "zero_neuron", "shift_bias", "rank_one")


def run_search(method, dev, initial, seed, *, offspring=4):
    """Return a scored candidate ID without access to evaluator data."""
    if method == "no_edit":
        return initial.candidate_id
    if method not in SEARCH_METHODS:
        raise ValueError(f"unknown search method {method!r}")
    if method in COMPOUND_SEARCH_METHODS:
        return _run_compound_search(method, dev, initial, seed, offspring)
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


def _run_compound_search(method, dev, initial, seed, offspring):
    """Score ordered recipe batches relative to the initial checkpoint.

    Greedy compares a signed block to its reflection, and ES compares siblings
    from one parent before replacing it. A final partial batch uses only the
    proposals the broker can score. With one remaining slot after scoring the
    initial checkpoint, each method scores at most one multi-operation recipe
    from the empty parent. The broker also logs any legality rejections.
    """
    if method == "compound_es" and (type(offspring) is not int or offspring < 1):
        raise ValueError("offspring must be a positive integer")
    method_index = COMPOUND_SEARCH_METHODS.index(method)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3, method_index]))
    space = EditSpace(tuple(dev.task.sizes), dev.limits, rng)
    attempts_remaining = 12 * dev.budget.candidates_remaining
    incumbent, parent = initial, []
    while dev.budget.candidates_remaining and attempts_remaining:
        if method == "compound_random":
            proposals = [space.compound_sample()]
        elif method == "compound_greedy":
            proposals = space.coordinate_blocks(parent)
        else:
            proposals = [space.compound_mutate(parent) for _ in range(offspring)]
        winner, winner_ops = incumbent, parent
        for proposal in proposals:
            if not attempts_remaining or not dev.budget.candidates_remaining:
                break
            attempts_remaining -= 1
            candidate = dev.probe(proposal)
            if candidate.status == "budget_exhausted":
                return winner.candidate_id
            if better(candidate, winner):
                winner, winner_ops = candidate, proposal
        incumbent, parent = winner, winner_ops
    return incumbent.candidate_id
