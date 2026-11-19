"""WeightBench task fixtures and experiment scores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import numpy as np

from ._vendor.classic import CartPole, MountainCar, Pendulum
from .policies import Policy, array_digest, forward, load_policy, policy_digest

WEIGHTBENCH_COMMIT = "2acd60172fc9758eb9030b1cab4a2ff29be5c993"
FIXTURE_FORMAT = "weightspace-v1"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    id: str
    name: str
    sizes: tuple[int, ...]
    horizon: int
    description: str
    normalization: str
    action_rule: str
    success_rule: str
    evaluator_source: str = WEIGHTBENCH_COMMIT


_ENVS = {
    "TR02-CTRL-002": MountainCar,
    "TR02-CTRL-003": Pendulum,
    "TR02-CTRL-004": CartPole,
}
TASKS = MappingProxyType({
    "TR02-CTRL-002": TaskSpec(
        "TR02-CTRL-002", "MountainCar-v0", (2, 96, 96, 3), MountainCar.horizon,
        "The car repeatedly stalls near the bottom of the valley. Repair it so it reaches the goal "
        "from failure starts while retaining that ability on normal starts.",
        "[(position + 0.3) / 0.9, velocity / 0.07]",
        "The largest of the three logits chooses left, coast, or right in that order.",
        "Reach position at least 0.5 with a nonnegative velocity within 200 steps.",
    ),
    "TR02-CTRL-003": TaskSpec(
        "TR02-CTRL-003", "Pendulum-v1", (3, 128, 128, 1), Pendulum.horizon,
        "The pendulum stalls before rising to the top from low-energy starts. Repair it so it can "
        "swing up while retaining its ability to stay upright from normal starts.",
        "[cos(theta), sin(theta), theta_dot / 8]",
        "Clip the single output to the torque range [-2, 2].",
        "Keep the angle within 0.3 radians of upright for each of the last 50 steps of 200.",
    ),
    "TR02-CTRL-004": TaskSpec(
        "TR02-CTRL-004", "CartPole-v1", (4, 256, 256, 2), CartPole.horizon,
        "The cart drifts off the track from some starts. Repair it so it remains on track with "
        "the pole upright while retaining this ability from normal starts.",
        "[position / 2.4, velocity / 2, angle / 0.21, angular_velocity / 2]",
        "The larger of the two logits chooses left or right in that order.",
        "Complete all 500 steps without crossing the position or angle termination threshold.",
    ),
})


@dataclass(frozen=True, slots=True, repr=False)
class StartSplit:
    failure: np.ndarray
    normal: np.ndarray

    def __post_init__(self):
        for name in ("failure", "normal"):
            data = np.asarray(getattr(self, name), dtype=np.float64)
            if data.ndim != 2 or not all(data.shape) or not np.isfinite(data).all():
                raise ValueError("each start region requires a nonempty matrix of finite states")
            immutable = np.frombuffer(data.tobytes(), dtype=np.float64).reshape(data.shape)
            object.__setattr__(self, name, immutable)
        if self.failure.shape[1] != self.normal.shape[1]:
            raise ValueError("start regions have inconsistent state widths")

    def max_rows(self, horizon: int) -> int:
        return (len(self.failure) + len(self.normal)) * horizon

    def digest(self) -> str:
        return array_digest({"failure": self.failure, "normal": self.normal})


@dataclass(frozen=True, slots=True, repr=False)
class TaskBundle:
    """Trusted fixture material. Pass only TaskSpec and broker interface to methods."""

    spec: TaskSpec
    policy: Policy
    dev: StartSplit
    heldout: StartSplit
    provenance: dict
    demo: dict[str, np.ndarray] | None = None


@dataclass(frozen=True, slots=True)
class Score:
    failure_wins: int
    failure_count: int
    normal_wins: int
    normal_count: int
    rows: int
    failure_success: float
    normal_success: float
    objective: float


def evaluate_split(
    spec: TaskSpec,
    policy: Policy,
    split: StartSplit,
    charge: Callable[[int], None],
) -> Score:
    """Evaluate both regions, charging each row before inference."""
    env = _ENVS[spec.id]()
    rows = 0

    def predict(observations: np.ndarray) -> np.ndarray:
        nonlocal rows
        count = len(observations)
        charge(count)
        rows += count
        return forward(policy, observations)

    failure_wins = int(env.rollout(predict, split.failure)["success"].sum())
    normal_wins = int(env.rollout(predict, split.normal)["success"].sum())
    failure_count, normal_count = len(split.failure), len(split.normal)
    failure_success = failure_wins / failure_count
    normal_success = normal_wins / normal_count
    return Score(
        failure_wins, failure_count, normal_wins, normal_count, rows, failure_success, normal_success,
        (failure_success + normal_success) / 2,
    )


def _task_spec(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError:
        raise ValueError(f"unknown task {task_id}") from None


def _validate_seed(seed: int, name: str) -> None:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError(f"{name} must be an integer in [0, 2**32)")


def _load_archive(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if missing := set(names) - set(archive.files):
            raise ValueError(f"{path.name} is missing {', '.join(sorted(missing))}")
        return {name: archive[name] for name in names}


def _validate_sets(spec: TaskSpec, sets: dict[str, np.ndarray]) -> None:
    env = _ENVS[spec.id]()
    rows = set()
    for name, values in sets.items():
        if values.dtype.kind not in "fiu":
            raise ValueError(f"invalid values in {name}")
        starts = np.asarray(values, dtype=np.float64)
        if (
            starts.ndim != 2 or starts.shape[1] != len(env.start_low) or not len(starts)
            or not np.isfinite(starts).all() or np.any(starts < env.start_low - 1e-9)
            or np.any(starts > env.start_high + 1e-9)
        ):
            raise ValueError(f"invalid starts in {name}")
        for row in starts:
            key = tuple(row)
            if key in rows:
                raise ValueError("fixture start regions must be disjoint and contain unique states")
            rows.add(key)


def _validate_demo(spec: TaskSpec, demo: dict[str, np.ndarray], start: np.ndarray) -> None:
    if len(start) != 1 or not np.array_equal(demo["start"], start[0]):
        raise ValueError("public trajectory start must match the standalone fixture start")
    obs, actions, rewards, success = (demo[name] for name in ("obs", "action", "reward", "success"))
    if (
        obs.ndim != 2 or obs.shape[1] != spec.sizes[0] or not 0 < len(obs) <= spec.horizon
        or actions.shape != (len(obs),) or rewards.shape != (len(obs),)
        or success.shape != () or success.dtype.kind != "b" or bool(success)
    ):
        raise ValueError("invalid public failure trajectory")
    if any(array.dtype.kind not in "fiu" or not np.isfinite(array).all() for array in (obs, actions, rewards)):
        raise ValueError("public trajectory must contain finite real values")


def _load_manifest(path: Path, task_id: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required_text = (
        "fixture_format", "task_id", "evaluator_source", "numpy_version", "torch_version",
        "policy_sha256", "starts_sha256", "demo_sha256",
    )
    if not isinstance(data, dict) or any(type(data.get(key)) is not str for key in required_text):
        raise ValueError("invalid fixture manifest")
    if (data["fixture_format"], data["task_id"], data["evaluator_source"]) != (
        FIXTURE_FORMAT, task_id, WEIGHTBENCH_COMMIT,
    ):
        raise ValueError("fixture manifest is incompatible with the evaluator")
    _validate_seed(data.get("seed"), "fixture seed")
    return {key: data[key] for key in (*required_text, "seed")}


def load_task(
    root: str | Path,
    task_id: str,
    *,
    seed: int | None = None,
    dev_per_region: int = 100,
    heldout_per_region: int = 100,
    split_seed: int | None = None,
) -> TaskBundle:
    """Load a generated or upstream fixture, with an optional paired dev subsample."""
    for name, value in (("seed", seed), ("split_seed", split_seed)):
        if value is not None:
            _validate_seed(value, name)
    for count in (dev_per_region, heldout_per_region):
        if type(count) is not int or count < 1:
            raise ValueError("split sample counts must be positive integers")
    spec = _task_spec(task_id)
    path = Path(root) / task_id
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        generation = _load_manifest(manifest_path, task_id)
        fixture_format = FIXTURE_FORMAT
        actual_seed = generation["seed"]
    else:
        generation = None
        fixture_format = "weightbench"
        stats = path / "private" / "stats.json"
        actual_seed = json.loads(stats.read_text(encoding="utf-8")).get("seed") if stats.is_file() else None
    if actual_seed is not None:
        _validate_seed(actual_seed, "fixture seed")
    if seed is not None and actual_seed != seed:
        raise ValueError(f"fixture seed is {actual_seed}, expected {seed}")

    policy = load_policy(path / "public" / "policy.npz", spec.sizes)
    sets = _load_archive(
        path / "private" / "eval_sets.npz", ("fail_train", "norm_train", "fail_eval", "norm_eval", "traj_start"),
    )
    _validate_sets(spec, sets)
    demo = _load_archive(path / "public" / "failure_trajectory.npz", ("start", "obs", "action", "reward", "success"))
    _validate_demo(spec, demo, sets["traj_start"])

    def select(name: str, count: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
        available = sets[name]
        if len(available) < count:
            raise ValueError(f"requested {count} starts but {name} contains only {len(available)}")
        return available[:count] if rng is None else available[rng.permutation(len(available))[:count]]

    dev_seeds = np.random.SeedSequence(split_seed).spawn(2) if split_seed is not None else (None, None)
    dev = StartSplit(*[
        select(name, dev_per_region, rng=np.random.default_rng(child) if child is not None else None)
        for name, child in zip(("fail_train", "norm_train"), dev_seeds)
    ])
    heldout = StartSplit(select("fail_eval", heldout_per_region), select("norm_eval", heldout_per_region))
    provenance = {
        "task_id": task_id,
        "seed": actual_seed,
        "split_seed": split_seed,
        "evaluator_source": WEIGHTBENCH_COMMIT,
        "fixture_format": fixture_format,
        "policy_sha256": policy_digest(policy),
        "dev_sha256": dev.digest(),
        "heldout_sha256": heldout.digest(),
        "demo_sha256": array_digest(demo),
        "numpy_version": np.__version__,
        "generation": generation,
    }
    if generation is not None:
        for key, computed in (
            ("policy_sha256", provenance["policy_sha256"]), ("starts_sha256", array_digest(sets)),
            ("demo_sha256", provenance["demo_sha256"]),
        ):
            if generation[key] != computed:
                raise ValueError(f"fixture digest mismatch for {key}")
    return TaskBundle(spec, policy, dev, heldout, provenance, demo)


def prepare_task(root: str | Path, task_id: str, seed: int) -> Path:
    """Generate a seed-specific fixture. Requires the optional PyTorch dependency."""
    from ._prepare import prepare

    _validate_seed(seed, "seed")
    return prepare(Path(root), _task_spec(task_id), seed)
