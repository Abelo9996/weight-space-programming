import json
import os
from pathlib import Path

import numpy as np
import pytest

from weightspace.tasks import TASKS, evaluate_split, load_task

@pytest.mark.parametrize("task_id", TASKS)
def test_local_seed_zero_fixture(task_id):
    fixtures = os.getenv("WEIGHTSPACE_TEST_FIXTURES")
    if fixtures is None:
        pytest.skip("set WEIGHTSPACE_TEST_FIXTURES to a seed-zero fixture directory")
    bundle = load_task(Path(fixtures), task_id, seed=0)
    for split_name in ("dev", "heldout"):
        rows = []
        score = evaluate_split(bundle.spec, bundle.policy, getattr(bundle, split_name), rows.append)
        assert score.failure_success == 0.0
        assert score.normal_success == 1.0
        assert score.rows == sum(rows)
        assert score.rows <= getattr(bundle, split_name).max_rows(bundle.spec.horizon)


def write_fixture(root, *, overlap=False):
    task_id = "TR02-CTRL-002"
    path = root / task_id
    (path / "public").mkdir(parents=True)
    (path / "private").mkdir()
    policy = {}
    sizes = TASKS[task_id].sizes
    for i, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
        policy[f"fc{i}.weight"] = np.zeros((outputs, inputs), dtype=np.float32)
        policy[f"fc{i}.bias"] = np.zeros(outputs, dtype=np.float32)
    np.savez(path / "public" / "policy.npz", **policy)
    fail_train = np.column_stack((np.linspace(-0.6, -0.5, 16), np.zeros(16)))
    fail_eval = fail_train[:2] if overlap else [[-0.8, 0.0], [-0.75, 0.0]]
    norm_train = np.column_stack((np.linspace(0.1, 0.15, 16), np.zeros(16)))
    traj_start = np.array([[-0.63, 0.0]])
    np.savez(
        path / "private" / "eval_sets.npz", fail_train=fail_train, fail_eval=fail_eval, norm_train=norm_train,
        norm_eval=[[0.2, 0.0], [0.21, 0.0]], traj_start=traj_start,
    )
    np.savez(
        path / "public" / "failure_trajectory.npz",
        start=traj_start[0], obs=traj_start, action=[0], reward=[-1], success=False,
    )
    (path / "private" / "stats.json").write_text(json.dumps({"seed": 53}), encoding="utf-8")
    return task_id


def test_dev_seed_is_reproducible_and_independent_of_heldout(tmp_path):
    task_id = write_fixture(tmp_path)
    kwargs = {"dev_per_region": 4, "heldout_per_region": 2, "seed": 53}
    first = load_task(tmp_path, task_id, split_seed=11, **kwargs)
    repeat = load_task(tmp_path, task_id, split_seed=11, **kwargs)
    other = load_task(tmp_path, task_id, split_seed=12, **kwargs)
    assert first.provenance["dev_sha256"] == repeat.provenance["dev_sha256"]
    assert first.provenance["dev_sha256"] != other.provenance["dev_sha256"]
    assert first.provenance["heldout_sha256"] == other.provenance["heldout_sha256"]
    assert first.provenance["policy_sha256"] == other.provenance["policy_sha256"]
    assert all(value.dtype == np.float64 for value in first.policy.values())
    with pytest.raises(ValueError, match="fixture seed"):
        load_task(tmp_path, task_id, seed=54)


def test_fixture_splits_must_be_disjoint(tmp_path):
    task_id = write_fixture(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="disjoint"):
        load_task(tmp_path, task_id, dev_per_region=1, heldout_per_region=1)
