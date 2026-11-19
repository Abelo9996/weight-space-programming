"""CPU preparation of WeightBench-compatible fixtures.

The default task preparation follows WeightBench at
2acd60172fc9758eb9030b1cab4a2ff29be5c993, excluding its reference and baseline
searches. See _vendor/WEIGHTBENCH_LICENSE.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ._vendor.training import train_bc
from .artifacts import canonical_json
from .policies import array_digest, copy_policy, forward, policy_digest
from .tasks import FIXTURE_FORMAT, _ENVS, TaskSpec

# Pruned units per hidden layer, behavior cloning samples/epochs, and start pool size.
_SETTINGS = {
    "TR02-CTRL-002": ((6, 4), 60_000, 40, 24_000),
    "TR02-CTRL-003": ((8, 4), 80_000, 40, 6_000),
    "TR02-CTRL-004": ((12, 6), 100_000, 30, 6_000),
}
_STARTS_PER_REGION = 100


def _cartpole_states(env, rng: np.random.Generator, count: int) -> np.ndarray:
    # WeightBench mixes states visited by a noisy expert and a uniform sample.
    low, high = [-2.0, -1.5, -0.1, -0.3], [2.0, 1.5, 0.1, 0.3]
    state = rng.uniform(low, high, size=(200, 4))
    visited = []
    for _ in range(400):
        visited.append(state.copy())
        expert = env.expert_target(state)
        actions = np.where(rng.random(len(state)) < 0.15, 1 - expert, expert)
        state, _, done = env.step(state, actions)
        if done.any():
            state[done] = rng.uniform(low, high, size=(int(done.sum()), 4))
    visited = np.concatenate(visited)[:count // 2]
    uniform = rng.uniform([-2.4, -2.5, -0.21, -2.5], [2.4, 2.5, 0.21, 2.5], size=(count - len(visited), 4))
    return np.concatenate([visited, uniform])


def _make_policy(spec: TaskSpec, env, rng: np.random.Generator, seed: int) -> dict:
    pruned_counts, sample_count, epochs, _ = _SETTINGS[spec.id]
    pruned = {
        layer: sorted(rng.choice(spec.sizes[layer + 1], count, replace=False).tolist())
        for layer, count in enumerate(pruned_counts)
    }
    if spec.id == "TR02-CTRL-002":
        states = rng.uniform([-1.2, -0.07], [0.6, 0.07], size=(sample_count, 2))
    elif spec.id == "TR02-CTRL-003":
        states = rng.uniform([-np.pi, -8.0], [np.pi, 8.0], size=(sample_count, 2))
    else:
        states = _cartpole_states(env, rng, sample_count)
    obs = env.obs(states)
    return train_bc(
        env.norm(obs), env.expert_target(obs), list(spec.sizes), "ce" if env.discrete else "mse",
        seed, pruned, epochs=epochs, log=lambda _: None,
    )


def _make_splits(spec: TaskSpec, env, policy: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    pool = env.sample_starts(rng, _SETTINGS[spec.id][-1])

    def predict(obs: np.ndarray) -> np.ndarray:
        return forward(policy, obs)

    outcomes = env.rollout(predict, pool)["success"]
    failures = rng.permutation(np.flatnonzero(~outcomes))
    normal = rng.permutation(np.flatnonzero(outcomes))
    count = _STARTS_PER_REGION
    if len(failures) < 2 * count + 1 or len(normal) < 2 * count:
        raise RuntimeError(f"insufficient fixture starts for {spec.id}: {len(failures)} failure and {len(normal)} normal")
    sets = {
        "fail_eval": pool[failures[:count]],
        "norm_eval": pool[normal[:count]],
        "fail_train": pool[failures[count:2 * count]],
        "norm_train": pool[normal[count:2 * count]],
        "traj_start": pool[failures[2 * count:2 * count + 1]],
    }
    trajectory = env.rollout(predict, sets["traj_start"], record=True)
    demo = {key: trajectory[key].astype(np.float32) for key in ("obs", "action", "reward")}
    demo.update(start=sets["traj_start"][0], success=bool(trajectory["success"][0]))
    return sets, demo


def prepare(root: Path, spec: TaskSpec, seed: int) -> Path:
    target = root / spec.id
    if target.exists():
        raise FileExistsError(target)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("fixture generation requires PyTorch") from error

    rng = np.random.default_rng(seed * 1000 + 17)
    env = _ENVS[spec.id]()
    previous_threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            params = _make_policy(spec, env, rng, seed)
    finally:
        torch.set_num_threads(previous_threads)
    policy = copy_policy(params, spec.sizes)
    sets, demo = _make_splits(spec, env, policy, rng)
    manifest = {
        "fixture_format": FIXTURE_FORMAT,
        "task_id": spec.id,
        "seed": seed,
        "evaluator_source": spec.evaluator_source,
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "policy_sha256": policy_digest(policy),
        "starts_sha256": array_digest(sets),
        "demo_sha256": array_digest(demo),
    }
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{spec.id}-", dir=root) as directory:
        staging = Path(directory) / "fixture"
        public, private = staging / "public", staging / "private"
        public.mkdir(parents=True)
        private.mkdir(mode=0o700)
        np.savez(public / "policy.npz", **params)
        np.savez(public / "failure_trajectory.npz", **demo)
        np.savez(private / "eval_sets.npz", **sets)
        (staging / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        lock = root / f".{spec.id}.publish"
        lock.mkdir()
        try:
            if target.exists():
                raise FileExistsError(target)
            staging.rename(target)
        finally:
            lock.rmdir()
    return target
