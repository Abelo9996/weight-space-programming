"""Run a predeclared paired policy-editing experiment."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform

import numpy as np

from .artifacts import RunJournal, canonical_json, json_digest
from .broker import BudgetBroker
from .editors import ClaudeEditor, ClaudeOptions, run_editor
from .edits import EditLimits
from .policies import forward, policy_digest
from .search import METHODS, run_search
from .tasks import FIXTURE_FORMAT, TASKS, load_task


CONDITIONS = ("original", "permutation", "relu_scale", "permutation_relu_scale")
PHASES = ("smoke", "pilot", "study")


@dataclass(frozen=True)
class ExperimentConfig:
    phase: str
    fixtures: str
    fixture_source: str
    fixture_seed: int
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    conditions: tuple[str, ...]
    methods: tuple[str, ...]
    candidate_cap: int
    dev_per_region: int
    heldout_per_region: int
    es_offspring: int = 4
    edit_limits: dict = field(default_factory=dict)
    editor: dict | None = None

    def __post_init__(self):
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if not isinstance(self.fixtures, str) or not self.fixtures:
            raise ValueError("fixtures must be a directory path")
        if self.fixture_source not in ("generated", "public_fixture"):
            raise ValueError("fixture_source must be generated or public_fixture")
        for name, choices in (("tasks", TASKS), ("conditions", CONDITIONS), ("methods", METHODS)):
            values = getattr(self, name)
            if (
                not isinstance(values, (tuple, list)) or not values
                or any(not isinstance(value, str) or value not in choices for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"{name} must be a nonempty list of unique choices from {tuple(choices)}")
            object.__setattr__(self, name, tuple(values))
        if "original" not in self.conditions:
            raise ValueError("conditions must include the original checkpoint")
        if (
            not isinstance(self.seeds, (tuple, list)) or not self.seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must contain unique integers in [0, 2**32)")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if type(self.fixture_seed) is not int or not 0 <= self.fixture_seed < 2**32:
            raise ValueError("fixture_seed must be an integer in [0, 2**32)")
        for name in ("candidate_cap", "dev_per_region", "heldout_per_region", "es_offspring"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.edit_limits, dict):
            raise ValueError("edit_limits must be an object")
        EditLimits(**self.edit_limits)
        if "claude" in self.methods:
            if not isinstance(self.editor, dict):
                raise ValueError("Claude method requires an editor settings object")
            ClaudeOptions(**self.editor)
        elif self.editor is not None:
            raise ValueError("editor settings require the Claude method")


def load_config(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected an experiment config object")
    return ExperimentConfig(**data)


def planned_cells(config):
    return [
        {"task": task, "seed": seed, "condition": condition, "method": method,
         "run_id": f"{task}.s{seed}.{condition}.{method}"}
        for task in config.tasks for seed in config.seeds
        for condition in config.conditions for method in config.methods
    ]


def _transform_policy(policy, sizes, condition, seed):
    try:
        import torch
        from torch import nn
        from .transforms import Permute, ReluScale, apply_transforms
    except ImportError as error:
        raise RuntimeError("paired policy conditions require the transforms extra") from error

    layers = []
    operations, serialized = [], []
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    for index, (source, target) in enumerate(zip(sizes, sizes[1:])):
        # Construction initializes discarded values. Preserve the caller's global torch RNG.
        with torch.random.fork_rng(devices=[]):
            layer = nn.Linear(source, target, dtype=torch.float64)
        layer.load_state_dict({
            part: torch.from_numpy(policy[f"fc{index}.{part}"])
            for part in ("weight", "bias")
        })
        layers.append(layer)
        if index == len(sizes) - 2:
            continue
        layers.append(nn.ReLU())
        order = rng.permutation(target).tolist()
        scales = rng.choice([0.5, 1.0, 2.0], size=target).tolist()
        if "permutation" in condition:
            operations.append(Permute(2 * index, order))
            serialized.append({"kind": "permutation", "layer": f"fc{index}", "order": order})
        if "relu_scale" in condition:
            operations.append(ReluScale(2 * index, scales))
            serialized.append({"kind": "relu_scale", "layer": f"fc{index}", "scales": scales})
    model = apply_transforms(nn.Sequential(*layers).eval(), *operations)
    transformed = {
        f"fc{index}.{part}": tensor.detach().numpy().copy()
        for index in range(len(sizes) - 1) for part, tensor in model[2 * index].state_dict().items()
    }
    direct_inputs = np.random.default_rng(np.random.SeedSequence([seed, 2])).standard_normal((1000, sizes[0]))
    error = float(np.max(np.abs(forward(policy, direct_inputs) - forward(transformed, direct_inputs))))
    if error >= 1e-5:
        raise RuntimeError(f"compensated {condition} changed fixed-input logits by {error}")
    return transformed, {
        "operations": serialized, "fixed_inputs": 1000, "fixed_inputs_seed_stream": 2,
        "max_abs_logit_error": error, "absolute_tolerance": 1e-5,
        "torch_version": str(torch.__version__),
    }


def _exclusive_json(path, record):
    with path.open("xb") as file:
        file.write(canonical_json(record) + b"\n")


def _source_provenance():
    root = Path(__file__).resolve().parent
    sources = {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    try:
        version = metadata.version("weightspace")
    except metadata.PackageNotFoundError:
        version = None
    return {"distribution_version": version, "source_sha256": json_digest(sources), "modules_sha256": sources}


def _save_policy(path, policy):
    with path.open("xb") as file:
        np.savez_compressed(file, **policy)
    return policy_digest(policy)


def _prepare_pair(config, fixtures, task_id, seed):
    task = load_task(
        fixtures, task_id, seed=config.fixture_seed, split_seed=seed,
        dev_per_region=config.dev_per_region, heldout_per_region=config.heldout_per_region,
    )
    expected_format = FIXTURE_FORMAT if config.fixture_source == "generated" else "weightbench"
    if task.provenance["fixture_format"] != expected_format:
        raise ValueError(f"{config.fixture_source} source requires fixture format {expected_format}")
    return task


def _run_cell(config, cell, journal, task, policy, transform, *, allow_live):
    digest = _save_policy(journal.path / "starting_policy.npz", policy)
    journal.append(
        "checkpoint", starting_policy_sha256=digest, provenance=task.provenance, transform=transform,
    )
    broker = BudgetBroker(
        task, policy=policy, candidate_cap=config.candidate_cap, limits=EditLimits(**config.edit_limits),
        journal=journal,
    )
    try:
        initial = broker.dev.probe([])
        if initial.status != "scored":
            raise RuntimeError(f"starting checkpoint could not be scored: {initial.status}")
        if cell["method"] == "claude":
            editor = ClaudeEditor(ClaudeOptions(**config.editor), journal, allow_live=allow_live)
            selected = run_editor(broker.dev, initial, editor)
        else:
            selected = run_search(
                cell["method"], broker.dev, initial, cell["seed"], offspring=config.es_offspring,
            )
        final = broker.finalize(selected)
        selected_policy = broker.selected_policy()
        digest = _save_policy(journal.path / "selected_policy.npz", selected_policy)
        if digest != final.policy_sha256:
            raise RuntimeError("selected artifact digest disagrees with evaluated checkpoint")
        if cell["condition"] == "original":
            for condition in config.conditions:
                if condition == "original":
                    continue
                transported, check = _transform_policy(
                    selected_policy, task.spec.sizes, condition, cell["seed"],
                )
                journal.append(
                    "selected_transport", condition=condition, selected_policy_sha256=digest,
                    transported_policy_sha256=policy_digest(transported), fixed_input_check=check,
                )
        journal.append(
            "run_completed", candidate_id=final.candidate_id, ops=final.ops,
            starting_policy_sha256=policy_digest(policy), selected_policy_sha256=digest,
            dev=asdict(final.dev), heldout=asdict(final.heldout), budget=asdict(final.budget),
            heldout_rows_reserved=final.heldout_rows_reserved, heldout_rows_used=final.heldout_rows_used,
        )
    except Exception as error:
        broker.abort(type(error).__name__)
        raise


def run_experiment(config, fixtures, output, *, allow_live=False):
    """Execute all cells and leave an auditable failure record for each bad cell."""
    if "claude" in config.methods and not allow_live:
        raise ValueError("the config includes Claude; pass --allow-live-editor to run it")
    output.mkdir(parents=True, exist_ok=False)
    config_data = asdict(config)
    config_hash = json_digest(config_data)
    cells = planned_cells(config)
    launch_bounds = None
    if "claude" in config.methods:
        options = ClaudeOptions(**config.editor)
        max_launches = options.max_calls * sum(cell["method"] == "claude" for cell in cells)
        launch_bounds = {
            "editor_cli_launches_upper_bound": max_launches,
            "sum_of_cli_api_equivalent_thresholds_usd": max_launches * options.max_api_equivalent_usd_per_call,
            "billing_assumption": options.billing_assumption, "actual_charge_usd": None,
            "effective_options": asdict(options),
        }
    _exclusive_json(output / "experiment.json", {
        "schema_version": 1, "config": config_data, "config_sha256": config_hash,
        "cells": cells, "python_version": platform.python_version(), "numpy_version": np.__version__,
        "source": _source_provenance(),
        "live_editor_launch_bounds": launch_bounds,
    })
    completed = 0
    for task_id in config.tasks:
        for seed in config.seeds:
            try:
                task = _prepare_pair(config, fixtures, task_id, seed)
                task_error = None
            except Exception as error:
                task_error = error
            pair = [cell for cell in cells if cell["task"] == task_id and cell["seed"] == seed]
            for condition in config.conditions:
                condition_error = task_error
                transform = None
                if condition_error is None:
                    try:
                        policy = task.policy
                        if condition != "original":
                            policy, transform = _transform_policy(policy, task.spec.sizes, condition, seed)
                    except Exception as error:
                        condition_error = error
                for cell in (cell for cell in pair if cell["condition"] == condition):
                    journal = RunJournal.create(
                        output / "runs", {**cell, "config_sha256": config_hash, "phase": config.phase},
                        run_id=cell["run_id"],
                    )
                    try:
                        if condition_error is not None:
                            raise condition_error
                        _run_cell(config, cell, journal, task, policy, transform, allow_live=allow_live)
                    except Exception as error:
                        journal.append("run_failed", error_type=type(error).__name__, reason=str(error))
                    else:
                        completed += 1
    return {"phase": config.phase, "expected": len(cells), "completed": completed, "failed": len(cells) - completed}
