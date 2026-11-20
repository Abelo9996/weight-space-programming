"""Prepare identical, round-trippable prompt text for both providers."""

from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
import torch
from torch import nn

from ..artifacts import canonical_json, json_digest
from ..transforms import Permute, ReluScale, TanhSignFlip, apply_transforms, shuffle_weights
from .data import tensor_digest


EQUIVALENT = ("permutation", "activation_symmetry", "composition")
CONDITIONS = ("original", *EQUIVALENT, "shuffle")
VERIFY_INPUT_COUNT = 1000
LOGIT_TOLERANCE = 1e-5
PROMPT_VERSION = 1
PROMPT_HEADER = (
    "Predict the top-1 classification accuracy of the target CNN on unseen data from its task. "
    "Use its evaluation architecture and raw weights. Each parameter is flattened in row-major order. "
    "Conv2d weights have axes output channel, input channel, kernel row, kernel column. "
    "Linear weights have axes output class, input channel. Images use channel, height, width order. "
    'Return exactly one JSON object of the form {"accuracy": number}, with a finite number from 0 to 1.\n'
)


def _seed(*parts):
    digest = sha256(canonical_json(parts)).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def serialize_model(model, architecture, activation):
    """Emit binary64 round-trip values, without IDs, filenames or split metadata."""
    layers = []
    for name, tensor in model.state_dict().items():
        data = tensor.detach().cpu().numpy()
        if not np.isfinite(data).all():
            raise ValueError("prompt tensors must be finite")
        layers.append({"parameter": name, "shape": list(data.shape), "values": data.reshape(-1).tolist()})
    return canonical_json({
        "architecture": {
            **architecture, "hidden_activation": activation, "conv_padding": 0,
            "head": "global_average_pool_then_linear_logits",
        },
        "parameters": layers,
    }).decode("utf-8")


@dataclass(frozen=True)
class Variant:
    text: str
    provenance: dict


def _make_variant(model, architecture, row, operations, check):
    return Variant(
        serialize_model(model, architecture, row.activation),
        {
            "network_sha256": row.network_sha256,
            "tensors_sha256": tensor_digest(model),
            "transform_sha256": json_digest(operations),
            "operations": operations,
            "logit_check": check,
        },
    )


def _operations(model, activation, layers, condition, generator):
    operations = []
    for index in layers:
        width = model[index].out_channels
        if condition in ("permutation", "composition"):
            order = generator.permutation(width)
            if np.array_equal(order, np.arange(width)):
                order = np.roll(order, 1)
            operations.append(Permute(index, tuple(map(int, order))))
        if condition in ("activation_symmetry", "composition"):
            if activation == "relu":
                # Power-of-two scales stay exact in binary64 over the supported range.
                exponents = generator.choice(np.array([-2, -1, 1, 2]), size=width)
                operations.append(ReluScale(index, tuple(map(float, np.exp2(exponents)))))
            else:
                signs = generator.choice(np.array([-1, 1]), size=width)
                signs[0] = -1
                operations.append(TanhSignFlip(index, tuple(map(int, signs))))
    return operations


@torch.inference_mode()
def variants_for(row, architecture, seed):
    """Verify each variant against the original's logits before any call can run."""
    model = row.model
    originals = _make_variant(model, architecture, row, [], None)
    variants = {"original": originals}
    layers = [index for index, module in enumerate(model) if type(module) is nn.Conv2d]
    if not any(model[index].out_channels > 1 for index in layers):
        raise ValueError("permutation conditions require at least one convolution with multiple channels")
    input_seed = _seed(seed, row.network_sha256, "verify")
    shape = tuple(architecture["input_shape"])
    # Batch the fixed inputs to keep memory practical across supported architectures.
    batch_size = min(32, max(1, 262_144 // int(np.prod(shape))))

    def input_batches():
        generator = np.random.default_rng(input_seed)
        for i in range(0, VERIFY_INPUT_COUNT, batch_size):
            size = min(batch_size, VERIFY_INPUT_COUNT - i)
            yield generator.uniform(-1, 1, size=(size, *shape))

    input_hash = sha256()
    reference = []
    for array in input_batches():
        input_hash.update(array.astype("<f8", copy=False).tobytes())
        reference.append(model(torch.from_numpy(array)))
    if any(not logits.isfinite().all() for logits in reference):
        raise ValueError("checkpoint produces nonfinite verification logits")

    def verify(candidate):
        differences = []
        for array, expected in zip(input_batches(), reference, strict=True):
            actual = candidate(torch.from_numpy(array))
            difference = (actual - expected).abs().amax().item()
            if not np.isfinite(difference):
                raise ValueError("transformed network produces nonfinite verification differences")
            differences.append(difference)
        return {
            "input_seed": input_seed, "input_sha256": input_hash.hexdigest(), "input_count": VERIFY_INPUT_COUNT,
            "distribution": "uniform [-1, 1)", "dtype": "float64", "tolerance": LOGIT_TOLERANCE,
            "max_abs_difference": max(differences),
        }

    for condition in EQUIVALENT:
        generator = np.random.default_rng(_seed(seed, row.network_sha256, condition))
        operations = _operations(model, row.activation, layers, condition, generator)
        candidate = apply_transforms(model, *operations)
        check = verify(candidate)
        if check["max_abs_difference"] >= LOGIT_TOLERANCE:
            raise ValueError(f"{condition} failed its logit invariance check")
        operation_records = [{"type": type(operation).__name__, **asdict(operation)} for operation in operations]
        variant = _make_variant(candidate, architecture, row, operation_records, check)
        if variant.provenance["tensors_sha256"] in {v.provenance["tensors_sha256"] for v in variants.values()}:
            raise ValueError(f"{condition} did not create a distinct weight representation")
        variants[condition] = variant

    for index in layers:
        weights = model[index].weight
        if not torch.any(weights != weights.flatten()[0]):
            continue
        shuffle_seed = _seed(seed, row.network_sha256, "shuffle", index)
        candidate = shuffle_weights(model, index, seed=shuffle_seed)
        check = verify(candidate)
        if check["max_abs_difference"] > LOGIT_TOLERANCE:
            variants["shuffle"] = _make_variant(
                candidate, architecture, row, [{"type": "ShuffleWeights", "layer": index, "seed": shuffle_seed}], check,
            )
            return variants
    raise ValueError("weight shuffle did not change the verified logits in any convolution")


def example_prefix(examples, architecture):
    parts = [PROMPT_HEADER]
    for row in examples:
        if row.split != "train":
            raise ValueError("in-context examples must be training checkpoints")
        parts.extend([
            "\nLabeled example network:\n", serialize_model(row.model, architecture, row.activation),
            "\nExample accuracy:\n", canonical_json({"accuracy": row.accuracy}).decode("utf-8"), "\n",
        ])
    parts.append("\nTarget network:\n")
    return "".join(parts)
