"""Load explicit checkpoint panels and bind split identities to their provenance."""

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import unicodedata

import numpy as np
import torch
from torch import nn

from ..artifacts import canonical_json, json_digest
from ..transforms.cnn_zoo import load_cifar10_fixture


ZOO_ARCHITECTURE = {
    "input_shape": [1, 32, 32],
    "convolutions": [{"out_channels": 16, "kernel_size": 3, "stride": 2} for _ in range(3)],
    "classes": 10,
}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def loads(text, **options):
    return json.loads(text, object_pairs_hook=_unique_keys, **options)


def fields(record, required, optional=()):
    if not isinstance(record, dict) or not set(required) <= record.keys() <= set(required) | set(optional):
        raise ValueError(f"expected fields {', '.join(required)}")
    return record


def finite_range(value, lower, upper, name):
    if type(value) not in (int, float) or not lower <= value <= upper:
        raise ValueError(f"{name} must be a number in [{lower}, {upper}]")
    return float(value)


def integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _string(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError(f"{name} must be a nonempty string of at most 2048 characters")
    return value


def _identity(value):
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _digest(value, name):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _file_bytes(path, digest):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("checkpoint file SHA-256 mismatch")
    return data


def _architecture(spec):
    fields(spec, ("input_shape", "convolutions", "classes"))
    shape = spec["input_shape"]
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("input_shape must be [channels, height, width]")
    _, height, width = [integer(size, 1, 1024, "input dimension") for size in shape]
    layers = spec["convolutions"]
    if not isinstance(layers, list) or not 1 <= len(layers) <= 12:
        raise ValueError("architecture must have 1 to 12 convolution layers")
    for layer in layers:
        fields(layer, ("out_channels", "kernel_size", "stride"))
        integer(layer["out_channels"], 1, 1024, "out_channels")
        kernel = integer(layer["kernel_size"], 1, 31, "kernel_size")
        stride = integer(layer["stride"], 1, 31, "stride")
        height, width = (height - kernel) // stride + 1, (width - kernel) // stride + 1
        if min(height, width) < 1:
            raise ValueError("convolutions do not fit the input shape")
    integer(spec["classes"], 2, 4096, "classes")
    return spec


def _build_model(architecture, activation):
    channels = architecture["input_shape"][0]
    activation_type = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    modules = []
    with torch.random.fork_rng(devices=[]):
        for layer in architecture["convolutions"]:
            modules.extend([nn.Conv2d(channels, **layer, dtype=torch.float64), activation_type()])
            channels = layer["out_channels"]
        modules.extend([
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, architecture["classes"], dtype=torch.float64),
        ])
    return nn.Sequential(*modules).eval()


def _load_npz(raw, architecture, activation):
    model = _build_model(architecture, activation)
    state = model.state_dict()
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays, torch.no_grad():
        if len(arrays.files) != len(state) or set(arrays.files) != state.keys():
            raise ValueError("NPZ tensors must match the declared CNN state_dict")
        for name, param in state.items():
            source = arrays[name]
            if source.dtype.kind != "f" or source.dtype.itemsize not in (4, 8) or source.shape != tuple(param.shape):
                raise ValueError("NPZ tensors must match the declared CNN shapes and use float32 or float64")
            tensor = torch.from_numpy(source.astype(np.float64, copy=True))
            if not tensor.isfinite().all():
                raise ValueError("checkpoint weights must be finite")
            param.copy_(tensor)
    return model


def tensor_digest(model):
    """Hash tensor meaning independently of NPZ/NPY packaging and byte order."""
    digest = sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().numpy().astype("<f8", copy=True)
        if not np.isfinite(array).all():
            raise ValueError("checkpoint weights must be finite")
        array[array == 0] = 0.0
        digest.update(canonical_json([name, list(array.shape)]))
        digest.update(b"\n")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class Checkpoint:
    id: str
    split: str
    accuracy: float
    activation: str
    network_sha256: str
    tensors_sha256: str
    checkpoint: dict
    model: nn.Sequential

    def provenance(self):
        return {
            "id": self.id, "split": self.split, "accuracy": self.accuracy,
            "activation": self.activation, "network_sha256": self.network_sha256,
            "tensors_sha256": self.tensors_sha256, "checkpoint": self.checkpoint,
        }


@dataclass(frozen=True)
class Panel:
    architecture: dict
    dataset: dict
    checkpoints: tuple[Checkpoint, ...]
    sha256: str

    def split(self, name):
        return tuple(checkpoint for checkpoint in self.checkpoints if checkpoint.split == name)

    def provenance(self):
        return {
            "manifest_sha256": self.sha256, "dataset": self.dataset, "architecture": self.architecture,
            "architecture_sha256": json_digest(self.architecture),
            "networks": [row.provenance() for row in self.checkpoints],
        }


def _fixture_entry(archive, activation):
    item = archive["fixtures"][activation]
    metrics = item["metrics_row"]
    source = _string(archive["source"], "fixture source")
    row = integer(item["zero_based_archive_row"], 0, 10**12, "archive row")
    checkpoint_id = f"row{row}"
    return (
        # The public extractor omits model directories. Its two final rows are from distinct runs.
        json_digest([source, checkpoint_id]),
        finite_range(float(metrics["test_accuracy"]), 0, 1, "fixture accuracy"),
        {
            "source_checkpoint": checkpoint_id,
            "weights_sha256": _digest(item["sha256"], "weights_sha256"),
            "layout_sha256": _digest(archive["layout_sha256"], "layout_sha256"),
            "layout": "keras_hwio_io",
        },
    )


def _load_checkpoint(base, dataset, architecture, entry):
    fields(entry, ("id", "training_run_id", "split", "accuracy", "activation", "checkpoint"))
    name = entry["id"]
    if not isinstance(name, str) or not _ID.fullmatch(name):
        raise ValueError("network id must contain only letters, digits, periods, underscores or hyphens")
    split, activation = entry["split"], entry["activation"]
    if split not in ("train", "test") or activation not in ("relu", "tanh"):
        raise ValueError("network split must be train or test, and activation must be relu or tanh")
    accuracy = finite_range(entry["accuracy"], 0, 1, "accuracy")
    run = _identity(_string(entry["training_run_id"], "training_run_id"))
    asset = fields(entry["checkpoint"], ("format", "path", "sha256", "source_checkpoint"),
                   ("weights_sha256", "layout_sha256", "layout"))
    relative = Path(_string(asset["path"], "checkpoint path"))
    if relative.is_absolute():
        raise ValueError("checkpoint paths must be relative to the panel manifest")
    path = base / relative
    raw = _file_bytes(path, _digest(asset["sha256"], "checkpoint sha256"))
    source_checkpoint = _identity(_string(asset["source_checkpoint"], "source_checkpoint"))
    provenance = {key: value for key, value in asset.items() if key != "path"}
    match asset["format"]:
        case "cnn-zoo-fixture":
            if path.name != "manifest.json" or architecture != ZOO_ARCHITECTURE:
                raise ValueError("CNN Zoo fixture requires manifest.json and the release architecture")
            archive = loads(raw)
            expected_run, expected_accuracy, expected = _fixture_entry(archive, activation)
            if (archive["source"] != dataset["source"] or run != expected_run or accuracy != expected_accuracy
                    or any(asset.get(key) != value for key, value in expected.items())):
                raise ValueError("CNN Zoo reading manifest disagrees with the archived checkpoint provenance")
            model = load_cifar10_fixture(path.parent, activation)
        case "pytorch-npz":
            if asset.get("layout") != "state_dict_oihw_oi" or {"weights_sha256", "layout_sha256"} & asset.keys():
                raise ValueError("pytorch-npz requires layout state_dict_oihw_oi")
            if not raw.startswith(b"PK\x03\x04"):
                raise ValueError("pytorch-npz checkpoint must be a NumPy .npz archive")
            model = _load_npz(raw, architecture, activation)
        case _:
            raise ValueError("checkpoint format must be cnn-zoo-fixture or pytorch-npz")
    network_identity = json_digest([_identity(dataset["source"]), run])
    checkpoint_identity = json_digest([_identity(dataset["source"]), source_checkpoint])
    provenance["source_checkpoint_sha256"] = checkpoint_identity
    return Checkpoint(name, split, accuracy, activation, network_identity, tensor_digest(model), provenance, model)


def load_panel(path):
    path = Path(path)
    raw = path.read_bytes()
    manifest = fields(loads(raw), ("schema_version", "dataset", "architecture", "networks"))
    integer(manifest["schema_version"], 1, 1, "panel schema_version")
    dataset = fields(manifest["dataset"], ("name", "source", "sampling"))
    _string(dataset["name"], "dataset name")
    _string(dataset["source"], "dataset source")
    if dataset["sampling"] not in ("fixtures", "panel"):
        raise ValueError("dataset sampling must be fixtures or panel")
    architecture = _architecture(manifest["architecture"])
    rows = manifest["networks"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("panel must list at least one checkpoint")
    checkpoints = tuple(sorted(
        (_load_checkpoint(path.parent, dataset, architecture, row) for row in rows),
        key=lambda row: row.id,
    ))
    keys = (
        ([row.id.casefold() for row in checkpoints], "network ID"),
        ([row.network_sha256 for row in checkpoints], "source training run"),
        ([row.checkpoint["source_checkpoint_sha256"] for row in checkpoints], "source checkpoint"),
        ([row.tensors_sha256 for row in checkpoints], "checkpoint tensors"),
    )
    for values, name in keys:
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {name}. Use one original checkpoint per network across both splits")
    if not any(row.split == "test" for row in checkpoints):
        raise ValueError("panel must contain at least one test checkpoint")
    return Panel(architecture, dataset, checkpoints, sha256(raw).hexdigest())


def fixture_manifest(directory, output):
    """Generate a local test-only manifest for the two archived fixtures."""
    directory, output = Path(directory), Path(output)
    archive_path = directory / "manifest.json"
    raw = archive_path.read_bytes()
    archive = loads(raw)
    rows = []
    for activation in ("relu", "tanh"):
        source_run, accuracy, provenance = _fixture_entry(archive, activation)
        rows.append({
            "id": f"fixture-{activation}", "training_run_id": source_run, "split": "test",
            "accuracy": accuracy, "activation": activation,
            "checkpoint": {
                "format": "cnn-zoo-fixture", **provenance, "sha256": sha256(raw).hexdigest(),
                "path": Path(os.path.relpath(archive_path.resolve(), output.parent.resolve())).as_posix(),
            },
        })
    return {
        "schema_version": 1,
        "dataset": {"name": archive["dataset"], "source": archive["source"], "sampling": "fixtures"},
        "architecture": ZOO_ARCHITECTURE,
        "networks": rows,
    }
