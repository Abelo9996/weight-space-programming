"""Ordered repair edits. Compensated symmetry transforms are separate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from .policies import Policy, copy_policy


class EditError(ValueError):
    """The submitted edit does not satisfy the shared edit protocol."""


@dataclass(frozen=True, slots=True)
class EditLimits:
    max_changed_fraction: float = 0.04
    max_operations: int = 16
    min_scale: float = 0.25
    max_scale: float = 4.0
    max_bias_shift: float = 4.0
    max_rank_alpha: float = 4.0
    max_rank_vector: float = 1.0

    def __post_init__(self) -> None:
        if type(self.max_operations) is not int or self.max_operations < 1:
            raise ValueError("max_operations must be a positive integer")
        if not 0 <= _number(self.max_changed_fraction, "max_changed_fraction") <= 1:
            raise ValueError("max_changed_fraction must be in [0, 1]")
        lo = _number(self.min_scale, "min_scale")
        hi = _number(self.max_scale, "max_scale")
        if not 0 < lo <= 1 <= hi:
            raise ValueError("scale limits must be positive and contain 1")
        for field in ("max_bias_shift", "max_rank_alpha", "max_rank_vector"):
            if _number(getattr(self, field), field) <= 0:
                raise ValueError(f"{field} must be positive")


@dataclass(frozen=True, slots=True)
class AppliedEdit:
    policy: Policy
    ops: list[dict]
    changed_entries: int
    touched_entries: int


def _unique_keys(items: list[tuple[str, object]]) -> dict:
    output: dict = {}
    for key, value in items:
        if key in output:
            raise EditError(f"duplicate JSON field {key}")
        output[key] = value
    return output


def _invalid_constant(value: str) -> None:
    raise EditError(f"nonfinite JSON literal {value}")


def parse_edits(text: str) -> list[dict]:
    """Decode a JSON array, rejecting duplicate keys and non-standard constants."""
    try:
        ops = json.loads(text, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    except (ValueError, RecursionError) as error:
        raise EditError(str(error)) from error
    if type(ops) is not list:
        raise EditError("operations must be a JSON array")
    return ops


def _number(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise EditError(f"{field} must be a finite JSON number")
    try:
        number = float(value)
    except OverflowError as error:
        raise EditError(f"{field} must be a finite JSON number") from error
    if not math.isfinite(number):
        raise EditError(f"{field} must be a finite JSON number")
    return number


def _bounded(value: object, field: str, lo: float, hi: float) -> float:
    number = _number(value, field)
    if not lo <= number <= hi:
        raise EditError(f"{field} must be in [{lo}, {hi}]")
    return number


def _index(value: object, size: int) -> int:
    if type(value) is not int or not 0 <= value < size:
        raise EditError(f"neuron must be an integer in [0, {size})")
    return value


def _vector(value: object, name: str, size: int, limit: float) -> list[float]:
    if type(value) is not list or len(value) != size:
        raise EditError(f"{name} must be a JSON array of length {size}")
    return [_bounded(element, name, -limit, limit) for element in value]


_FIELDS = {
    "scale_neuron": {"op", "layer", "neuron", "factor"},
    "zero_neuron": {"op", "layer", "neuron"},
    "shift_bias": {"op", "layer", "neuron", "shift"},
    "rank_one": {"op", "layer", "alpha", "u", "v"},
}


def apply_edits(base: Policy, ops: list[dict] | str, limits: EditLimits = EditLimits()) -> AppliedEdit:
    """Apply all operations to a copy, with a cap on distinct entries ever changed.

    Scaling multiplies a hidden unit's incoming row and bias. Zeroing sets its
    outgoing column to zero. Neither operation applies a compensating change.
    """
    if isinstance(ops, str):
        ops = parse_edits(ops)
    if type(ops) is not list or len(ops) > limits.max_operations:
        raise EditError(f"operations must be a JSON array of at most {limits.max_operations} edits")
    policy = copy_policy(base)
    layers = [f"fc{i}" for i in range(len(policy) // 2)]
    touched = {key: np.zeros(value.shape, dtype=bool) for key, value in policy.items()}
    max_changed = math.floor(limits.max_changed_fraction * sum(value.size for value in policy.values()))
    normalized = []
    for entry in ops:
        if type(entry) is not dict:
            raise EditError("each operation must be a JSON object")
        op, layer = entry.get("op"), entry.get("layer")
        if type(op) is not str or op not in _FIELDS or entry.keys() != _FIELDS[op]:
            raise EditError("unknown operation or incorrect operation fields")
        if type(layer) is not str or layer not in layers:
            raise EditError("unknown layer")
        layer_index = layers.index(layer)
        weight, bias = f"{layer}.weight", f"{layer}.bias"
        if op in {"scale_neuron", "zero_neuron"} and layer_index + 1 == len(layers):
            raise EditError(f"{op} requires a hidden layer")
        canonical = {"op": op, "layer": layer}
        if op != "rank_one":
            neuron = _index(entry["neuron"], len(policy[bias]))
            canonical["neuron"] = neuron
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            if op == "scale_neuron":
                factor = _bounded(entry["factor"], "factor", limits.min_scale, limits.max_scale)
                canonical["factor"] = factor
                _update(policy, touched, weight, neuron, policy[weight][neuron] * factor)
                _update(policy, touched, bias, neuron, policy[bias][neuron] * factor)
            elif op == "zero_neuron":
                outgoing = f"{layers[layer_index + 1]}.weight"
                _update(policy, touched, outgoing, (slice(None), neuron), 0.0)
            elif op == "shift_bias":
                shift = _bounded(entry["shift"], "shift", -limits.max_bias_shift, limits.max_bias_shift)
                canonical["shift"] = shift
                _update(policy, touched, bias, neuron, policy[bias][neuron] + shift)
            else:
                rows, cols = policy[weight].shape
                alpha = _bounded(entry["alpha"], "alpha", -limits.max_rank_alpha, limits.max_rank_alpha)
                u = _vector(entry["u"], "u", rows, limits.max_rank_vector)
                v = _vector(entry["v"], "v", cols, limits.max_rank_vector)
                canonical.update(alpha=alpha, u=u, v=v)
                _update(policy, touched, weight, slice(None), policy[weight] + alpha * np.outer(u, v))
        if _count(touched) > max_changed:
            raise EditError(f"edit exceeds cumulative cap of {max_changed} changed entries")
        normalized.append(canonical)
    changed = sum(int(np.count_nonzero(value != base[name])) for name, value in policy.items())
    return AppliedEdit(policy, normalized, changed, _count(touched))


def _update(policy: Policy, touched: dict, name: str, index, value) -> None:
    if not np.isfinite(value).all():
        raise EditError(f"edit produces nonfinite values in {name}")
    touched[name][index] |= policy[name][index] != value
    policy[name][index] = value


def _count(touched: dict) -> int:
    return sum(int(np.count_nonzero(value)) for value in touched.values())
