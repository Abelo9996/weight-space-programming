import json

import numpy as np
import pytest

from weightspace.edits import EditError, EditLimits, apply_edits


def policy():
    return {
        "fc0.weight": np.array([[1.0, -2.0], [3.0, 4.0]]),
        "fc0.bias": np.array([0.5, -0.5]),
        "fc1.weight": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "fc1.bias": np.array([0.0, 0.0]),
    }


def op(name, layer="fc0", **fields):
    return {"op": name, "layer": layer, **fields}


def test_edit_order_nonmutation_and_replay():
    base = policy()
    ops = [
        op("shift_bias", neuron=0, shift=0.5),
        op("scale_neuron", neuron=0, factor=2),
        op("rank_one", "fc1", alpha=2, u=[0, 1], v=[0, 1]),
        op("zero_neuron", neuron=0),
    ]
    limits = EditLimits(max_changed_fraction=1)
    result = apply_edits(base, json.dumps(ops), limits)
    assert result.changed_entries == result.touched_entries == 6
    np.testing.assert_array_equal(result.policy["fc0.weight"], [[2, -4], [3, 4]])
    np.testing.assert_array_equal(result.policy["fc0.bias"], [2, -0.5])
    np.testing.assert_array_equal(result.policy["fc1.weight"], [[0, 2], [0, 6]])
    for key in base:
        np.testing.assert_array_equal(base[key], policy()[key])
        assert not np.shares_memory(base[key], result.policy[key])
    reversed_order = apply_edits(base, [ops[1], ops[0], *ops[2:]], limits)
    assert reversed_order.policy["fc0.bias"][0] == 1.5
    replay = apply_edits(base, result.ops, limits)
    for key in base:
        np.testing.assert_array_equal(replay.policy[key], result.policy[key])


@pytest.mark.parametrize("ops", [
    {},
    [op("scale_neuron", "fc1", neuron=0, factor=2.0)],
    [op("zero_neuron", "fc1", neuron=0)],
    [op("shift_bias", "fc2", neuron=0, shift=0.5)],
    [op("shift_bias", neuron=True, shift=1)],
    [op("shift_bias", neuron=-1, shift=1)],
    [op("shift_bias", neuron=2, shift=1)],
    [op("shift_bias", neuron=0, shift="0.1")],
    [op("shift_bias", neuron=0, shift=5.0)],
    [op("shift_bias", neuron=0, shift=float("nan"))],
    [op("shift_bias", neuron=0, shift=10**400)],
    [op("shift_bias", neuron=0, shift=1, undeclared=True)],
    [op("scale_neuron", neuron=0, factor=0)],
    [op("rank_one", alpha=1, u=[1], v=[0, 1])],
    [op("rank_one", alpha=5, u=[1, 0], v=[0, 1])],
    [op("rank_one", alpha=1, u=[1.1, 0], v=[0, 1])],
    [op("rank_one", alpha=1, u=[1, 0], v=[0, float("inf")])],
    '[{"op":"shift_bias","layer":"fc0","neuron":0,"shift":1e500}]',
    '[{"op":"shift_bias","layer":"fc0","neuron":0,"shift":NaN}]',
    '[{"op":"shift_bias","layer":"fc0","neuron":0,"neuron":1,"shift":1}]',
])
def test_reject_invalid_edits(ops):
    with pytest.raises(EditError):
        apply_edits(policy(), ops, EditLimits(max_changed_fraction=1))


def test_reject_overflow_without_mutation():
    base = policy()
    base["fc0.weight"][0, 0] = np.finfo(np.float64).max
    huge = base["fc0.weight"].copy()
    with pytest.raises(EditError, match="nonfinite"):
        apply_edits(base, [op("scale_neuron", neuron=0, factor=2)], EditLimits(max_changed_fraction=1))
    np.testing.assert_array_equal(base["fc0.weight"], huge)


def test_cap_keeps_canceled_entries_and_checks_length():
    shift = op("shift_bias", neuron=0, shift=1)
    cancel = op("shift_bias", neuron=0, shift=-1)
    another = op("shift_bias", neuron=1, shift=1)
    limits = EditLimits(max_changed_fraction=0.1)
    result = apply_edits(policy(), [shift, cancel], limits)
    assert (result.changed_entries, result.touched_entries) == (0, 1)
    with pytest.raises(EditError, match="cumulative cap"):
        apply_edits(policy(), [shift, cancel, another], limits)
    with pytest.raises(EditError, match="at most"):
        apply_edits(policy(), [shift, cancel], EditLimits(max_operations=1))
