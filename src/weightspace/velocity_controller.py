"""Compile a task-aware MountainCar controller into a masked ReLU policy."""

from math import frexp, ldexp

import numpy as np

from .edits import EditError, EditLimits, apply_edits
from .policies import Policy, copy_policy
from .search import better
from .tasks import TASKS


VELOCITY_CONTROLLER_TASK = "TR02-CTRL-002"


def _masked_units(policy, layer):
    incoming = policy[f"fc{layer}.weight"]
    biases = policy[f"fc{layer}.bias"]
    outgoing = policy[f"fc{layer + 1}.weight"]
    indices = np.flatnonzero((incoming == 0).all(axis=1) & (biases == 0) & (outgoing == 0).all(axis=0))
    if len(indices) < 2:
        raise EditError(f"velocity controller requires two structurally masked units in fc{layer}")
    return indices[:2]


def compile_velocity_controller(policy: Policy, limits: EditLimits) -> list[dict]:
    """Build exact logits [ReLU(-z), 0, ReLU(z)] for z = velocity / 0.07.

    Pushing in the direction of motion adds energy to the car. At zero
    velocity all logits tie and the public argmax chooses action zero (left).
    Two previously disconnected paths carry the positive and negative parts.
    Each existing output row is replaced via a bounded rank-one difference,
    so the trained network cannot override the sign rule. Compilation uses
    only the visible weights and edit limits, with no policy evaluations.
    """
    sizes = TASKS[VELOCITY_CONTROLLER_TASK].sizes
    _, first_width, second_width, actions = sizes
    policy = copy_policy(policy, sizes)
    left_source, right_source = _masked_units(policy, 0)
    left_relay, right_relay = _masked_units(policy, 1)

    # Use powers of two so scaling a stored weight to an edit coefficient
    # and back preserves exact cancellation wherever it is representable.
    alpha_exponent = frexp(min(4.0, limits.max_rank_alpha))[1] - 1
    vector_exponent = frexp(min(1.0, limits.max_rank_vector))[1] - 1
    alpha = ldexp(1.0, alpha_exponent)
    unit = ldexp(1.0, vector_exponent)
    ops = []

    def rank_one(layer, direction, delta):
        if not np.any(delta):
            return
        with np.errstate(over="ignore", under="ignore"):
            coefficients = np.ldexp(delta, -(alpha_exponent + vector_exponent))
            recovered = alpha * (unit * coefficients)
        if not np.array_equal(recovered, delta) or np.max(np.abs(coefficients)) > limits.max_rank_vector:
            raise EditError(f"velocity controller cannot exactly encode fc{layer} within the rank-one bounds")
        ops.append({
            "op": "rank_one", "layer": f"fc{layer}", "alpha": alpha,
            "u": (unit * direction).tolist(), "v": coefficients.tolist(),
        })

    first = np.eye(first_width)[[left_source, right_source]]
    second = np.eye(second_width)[[left_relay, right_relay]]
    rank_one(0, first[1] - first[0], np.array([0.0, 1.0]))
    for source, relay in zip(first, second, strict=True):
        rank_one(1, relay, source)

    output_rows = [second[0], np.zeros(second_width), second[1]]
    for action, selector in enumerate(np.eye(actions)):
        rank_one(2, selector, output_rows[action] - policy["fc2.weight"][action])
        bias = float(policy["fc2.bias"][action])
        if bias:
            ops.append({"op": "shift_bias", "layer": "fc2", "neuron": action, "shift": -bias})

    return apply_edits(policy, ops, limits).ops


def run_velocity_controller(dev, initial):
    """Offer the fixed controller once, then apply ordinary development selection."""
    if dev.task.id != VELOCITY_CONTROLLER_TASK:
        raise ValueError(f"velocity_controller requires {VELOCITY_CONTROLLER_TASK}")
    if not dev.budget.candidates_remaining:
        return initial.candidate_id
    candidate = dev.probe(compile_velocity_controller(dev.policy(), dev.limits))
    if candidate.status in ("rejected", "failed"):
        raise RuntimeError(f"velocity controller candidate {candidate.status}: {candidate.reason}")
    return candidate.candidate_id if better(candidate, initial) else initial.candidate_id
