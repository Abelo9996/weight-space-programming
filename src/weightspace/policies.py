"""Float64 ReLU checkpoint validation, serialization, and inference."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

Policy = dict[str, np.ndarray]


def copy_policy(params: Mapping[str, np.ndarray], sizes: Sequence[int] | None = None) -> Policy:
    """Validate layer names, dimensions, and finite values, then copy a checkpoint."""
    count = len(params) // 2
    layers = [f"fc{i}" for i in range(count)]
    names = {f"{layer}.{part}" for layer in layers for part in ("weight", "bias")}
    if not layers or set(params) != names:
        raise ValueError("policy requires consecutive fcN.weight and fcN.bias tensors")
    output: Policy = {}
    architecture = []
    for layer in layers:
        weight, bias = [np.asarray(params[f"{layer}.{part}"]) for part in ("weight", "bias")]
        if weight.ndim != 2 or not all(weight.shape) or bias.shape != (weight.shape[0],):
            raise ValueError(f"invalid weight or bias shape for {layer}")
        if architecture and architecture[-1] != weight.shape[1]:
            raise ValueError(f"incorrect input width for {layer}")
        if not architecture:
            architecture.append(weight.shape[1])
        architecture.append(weight.shape[0])
        for part, tensor in (("weight", weight), ("bias", bias)):
            name = f"{layer}.{part}"
            if tensor.dtype.kind not in "fiu":
                raise ValueError(f"{name} requires real numeric values")
            with np.errstate(over="ignore", invalid="ignore"):
                converted = np.array(tensor, dtype=np.float64, order="C", copy=True)
            if not np.isfinite(converted).all():
                raise ValueError(f"{name} contains nonfinite values")
            output[name] = converted
    if sizes is not None and tuple(sizes) != tuple(architecture):
        raise ValueError(f"expected architecture {tuple(sizes)}, got {tuple(architecture)}")
    return output


def load_policy(path: Path, sizes: Sequence[int] | None = None) -> Policy:
    with np.load(path, allow_pickle=False) as archive:
        return copy_policy({name: archive[name] for name in archive.files}, sizes)


def forward(params: Policy, observations: np.ndarray) -> np.ndarray:
    """Evaluate a batch of normalized observations using a validated checkpoint."""
    hidden = np.asarray(observations, dtype=np.float64)
    if hidden.ndim != 2 or hidden.shape[1] != params["fc0.weight"].shape[1]:
        raise ValueError("observations do not match the checkpoint input width")
    if not np.isfinite(hidden).all():
        raise FloatingPointError("nonfinite observation")
    layers = len(params) // 2
    for i in range(layers):
        # Some Accelerate builds raise spurious floating-point status flags on matmul.
        # Inspect every intermediate array, including preactivations before clipping.
        with np.errstate(all="ignore"):
            hidden = hidden @ params[f"fc{i}.weight"].T + params[f"fc{i}.bias"]
        if not np.isfinite(hidden).all():
            raise FloatingPointError(f"nonfinite preactivation in fc{i}")
        if i + 1 < layers:
            hidden = np.maximum(hidden, 0)
    return hidden


def array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash shape, name, and canonical little-endian float64 values."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name], dtype="<f8", order="C")
        digest.update(json.dumps([name, array.shape], separators=(",", ":")).encode())
        digest.update(b"\n")
        digest.update(array.tobytes())
    return digest.hexdigest()


def policy_digest(params: Policy) -> str:
    return array_digest(params)
