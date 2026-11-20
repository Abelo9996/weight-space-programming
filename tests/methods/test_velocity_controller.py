import json

import numpy as np
import pytest

from weightspace import broker, experiment
from weightspace.edits import EditError, EditLimits, apply_edits
from weightspace.policies import forward, policy_digest
from weightspace.tasks import Score, StartSplit, TASKS, TaskBundle
from weightspace.velocity_controller import VELOCITY_CONTROLLER_TASK, compile_velocity_controller


@pytest.fixture
def starting_policy():
    rng = np.random.default_rng(78)
    sizes = TASKS[VELOCITY_CONTROLLER_TASK].sizes
    policy = {
        f"fc{layer}.{field}": rng.uniform(-1.25, 1.25, size=shape)
        for layer, (source, target) in enumerate(zip(sizes, sizes[1:]))
        for field, shape in (("weight", (target, source)), ("bias", target))
    }
    for layer, lanes in enumerate(([7, 36], [19, 51])):
        policy[f"fc{layer}.weight"][lanes, :] = 0
        policy[f"fc{layer}.bias"][lanes] = 0
        policy[f"fc{layer + 1}.weight"][:, lanes] = 0
    return policy


@pytest.mark.parametrize("condition", ("original", "permutation_relu_scale"))
def test_compiler_implements_exact_velocity_logits_within_shared_limits(starting_policy, condition):
    sizes = TASKS[VELOCITY_CONTROLLER_TASK].sizes
    if condition != "original":
        pytest.importorskip("torch")
        starting_policy, _ = experiment._transform_policy(starting_policy, sizes, condition, seed=541)
    original_digest = policy_digest(starting_policy)
    limits = EditLimits()
    ops = compile_velocity_controller(starting_policy, limits)
    edited = apply_edits(starting_policy, ops, limits)
    assert len(ops) <= 9 <= limits.max_operations
    assert set(op["op"] for op in ops) <= {"rank_one", "shift_bias"}
    assert edited.changed_entries == edited.touched_entries
    assert edited.touched_entries <= 295 <= int(limits.max_changed_fraction * sum(
        tensor.size for tensor in starting_policy.values()
    ))
    assert original_digest == policy_digest(starting_policy)

    velocities = [-1, -0.5, -np.finfo(float).eps, -0.0, 0.0, np.finfo(float).eps, 0.5, 1]
    states = np.array([(position, velocity) for position in (-1, -0.3, 0, 0.4, 1) for velocity in velocities])
    z = states[:, 1]
    logits = forward(edited.policy, states)
    np.testing.assert_array_equal(logits, np.column_stack((np.maximum(-z, 0), np.zeros(len(z)), np.maximum(z, 0))))
    np.testing.assert_array_equal(np.argmax(logits, axis=1), np.where(z > 0, 2, 0))


def test_compiler_fails_explicitly_when_no_legal_recipe_can_be_constructed(starting_policy):
    with pytest.raises(EditError, match="at most 2"):
        compile_velocity_controller(starting_policy, EditLimits(max_operations=2))
    with pytest.raises(EditError, match="cumulative cap"):
        compile_velocity_controller(starting_policy, EditLimits(max_changed_fraction=0.01))
    with pytest.raises(EditError, match="within the rank-one bounds"):
        compile_velocity_controller(starting_policy, EditLimits(max_rank_alpha=0.5))
    starting_policy["fc0.bias"][7] = 1
    with pytest.raises(EditError, match="two structurally masked units in fc0"):
        compile_velocity_controller(starting_policy, EditLimits())


@pytest.mark.parametrize("candidate_cap, improves", [(1, True), (2, True), (2, False)])
def test_controller_cell_uses_broker_slots_and_ordinary_selection(starting_policy, tmp_path, monkeypatch,
                                                           candidate_cap, improves):
    spec = TASKS[VELOCITY_CONTROLLER_TASK]
    dev = StartSplit(np.array([[0.0, 0.0]]), np.array([[0.1, 0.0]]))
    heldout = StartSplit(np.array([[0.2, 0.0]]), np.array([[0.3, 0.0]]))
    task = TaskBundle(spec, starting_policy, dev, heldout, {"fixture_format": "weightspace-v1"})
    calls = []

    def evaluate(spec, policy, split, charge):
        analytic = np.count_nonzero(policy["fc2.weight"]) == 2
        calls.append((split.digest(), analytic))
        charge(2)
        failure = int(improves and analytic)
        return Score(failure, 1, 1, 1, 2, float(failure), 1.0, (failure + 1) / 2)

    monkeypatch.setattr(experiment, "load_task", lambda *args, **kwargs: task)
    monkeypatch.setattr(broker, "evaluate_split", evaluate)
    config = experiment.ExperimentConfig(
        phase="smoke", fixtures=".", fixture_source="generated", fixture_seed=0,
        tasks=(spec.id,), seeds=(41,), conditions=("original",), methods=("velocity_controller",),
        candidate_cap=candidate_cap, dev_per_region=1, heldout_per_region=1,
    )
    output = tmp_path / "controller"
    assert experiment.run_experiment(config, tmp_path, output)["completed"] == 1
    run = next((output / "runs").iterdir())
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    selected_edit = candidate_cap == 2 and improves
    assert events[-1]["candidate_id"] == int(selected_edit)
    assert events[-1]["budget"]["candidates_used"] == candidate_cap
    assert events[-1]["budget"]["rows_reserved"] == candidate_cap * 2 * spec.horizon
    assert events[-1]["budget"]["rows_used"] == candidate_cap * 2
    expected = [(dev.digest(), False)]
    if candidate_cap == 2:
        expected.append((dev.digest(), True))
    assert calls == [*expected, (heldout.digest(), selected_edit)]
