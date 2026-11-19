"""Load a pinned Small CNN Zoo CIFAR-10-GS fixture as a PyTorch model."""

import ast
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _verified_bytes(path: Path, expected_hash: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"fixture digest mismatch for {path}")
    return data


def load_cifar10_fixture(
    directory: Path, activation: str, *, dtype: torch.dtype = torch.float64
) -> nn.Sequential:
    """Restore HWIO and IO release tensors into an eval-mode NCHW CNN."""
    directory = Path(directory)
    activation_type = {"relu": nn.ReLU, "tanh": nn.Tanh}.get(activation)
    if activation_type is None:
        raise ValueError("fixture activation must be relu or tanh")

    manifest = json.loads((directory / "manifest.json").read_text())
    fixture = manifest["fixtures"][activation]
    if fixture["metrics_row"]["config.activation"] != activation:
        raise ValueError("release metadata activation disagrees with the requested fixture")
    fixture_bytes = _verified_bytes(directory / fixture["path"], fixture["sha256"])
    weights = np.load(io.BytesIO(fixture_bytes), allow_pickle=False)
    if weights.shape != (4_970,) or weights.dtype.str != "<f4":
        raise ValueError("unexpected fixture array format")

    layout_bytes = _verified_bytes(directory / "layout.csv", manifest["layout_sha256"])
    tensors = {}
    for row in csv.DictReader(io.StringIO(layout_bytes.decode())):
        start, end = int(row["start_idx"]), int(row["end_idx"])
        shape = ast.literal_eval(row["shape"])
        tensors[row["varname"]] = torch.tensor(weights[start:end].reshape(shape), dtype=dtype)

    # Original Keras Conv2D uses valid padding and the release training default stride of 2.
    # Training-only dropout does not affect the evaluation graph.
    modules = []
    release_layers = ("conv2d", "conv2d_1", "conv2d_2", "dense")
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for index, name in enumerate(release_layers):
            path = f"sequential/{name}"
            kernel = tensors[f"{path}/kernel:0"]
            bias = tensors[f"{path}/bias:0"]
            if name == "dense":
                affine = nn.Linear(16, 10, dtype=dtype)
                affine.weight.copy_(kernel.transpose(0, 1))
                modules.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), affine])
            else:
                affine = nn.Conv2d(1 if index == 0 else 16, 16, 3, stride=2, dtype=dtype)
                affine.weight.copy_(kernel.permute(3, 2, 0, 1))
                modules.extend([affine, activation_type()])
            affine.bias.copy_(bias)
    return nn.Sequential(*modules).eval()
