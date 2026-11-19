import hashlib
import json
import math
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import unittest
from uuid import uuid4

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch import nn

from weightspace.transforms import Permute, ReluScale, TanhSignFlip, apply_transforms, shuffle_weights
from weightspace.transforms.cnn_zoo import load_cifar10_fixture


FIXTURES = os.getenv("WEIGHTSPACE_CNN_ZOO_FIXTURES")
REQUIRES_FIXTURES = unittest.skipUnless(
    FIXTURES,
    "Fetch fixtures with scripts/fetch_cnn_zoo_fixture.py and set WEIGHTSPACE_CNN_ZOO_FIXTURES",
)
INPUT_COUNT = 1_000
TOLERANCE = 1e-5
MODEL_SEED = 723
INPUT_SEED = 17021
TRANSFORM_SEED = 6151
SHUFFLE_SEED = 5539
RUN_ID = os.getenv("WSP_TRANSFORM_RUN_ID", str(uuid4()))


def write_metric(record):
    path = os.getenv("WSP_TRANSFORM_LOG")
    if path is None:
        return
    meta = {
        "run_id": RUN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "python -m unittest discover -s tests/transforms -v",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
    }
    with Path(path).open("a") as stream:
        stream.write(json.dumps(meta | record, allow_nan=False, sort_keys=True) + "\n")


def toy_mlp(*activations):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_SEED)
        return nn.Sequential(
            nn.Linear(5, 9, dtype=torch.float64), activations[0](),
            nn.Linear(9, 7, bias=False, dtype=torch.float64), activations[1](),
            nn.Linear(7, 4, dtype=torch.float64),
        ).eval()


def synthetic_flat_cnn(activation):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_SEED)
        return nn.Sequential(
            nn.Conv2d(1, 4, 3, dtype=torch.float64), activation(),
            nn.Conv2d(4, 5, 3, dtype=torch.float64), activation(),
            nn.AvgPool2d(2), nn.Flatten(),
            nn.Linear(5 * 2 * 2, 3, dtype=torch.float64),
        ).eval()


def transform_cases(model, layer_activations):
    generator = torch.Generator().manual_seed(TRANSFORM_SEED)
    permutations = []
    diagonals = []
    for layer, activation in layer_activations:
        width = model[layer].weight.shape[0]
        order = torch.randperm(width, generator=generator)
        if torch.equal(order, torch.arange(width)):
            order = order.roll(1)
        permutations.append(Permute(layer, tuple(order.tolist())))
        if activation is nn.ReLU:
            scales = 4 ** (2 * torch.rand(width, generator=generator, dtype=torch.float64) - 1)
            diagonals.append(ReluScale(layer, tuple(scales.tolist())))
        else:
            signs = 2 * torch.randint(2, (width,), generator=generator) - 1
            signs[0] = -1
            signs[-1] = 1
            diagonals.append(TanhSignFlip(layer, tuple(signs.tolist())))
    cases = {"permutation": tuple(permutations)}
    if len(set(activation for _, activation in layer_activations)) == 1:
        name = "relu_scale" if layer_activations[0][1] is nn.ReLU else "tanh_sign_flip"
        cases[name] = tuple(diagonals)
    # These operations share an affine layer at each adjacent hidden boundary.
    cases["composition"] = tuple(
        transform
        for same_layer in zip(permutations, diagonals, strict=True)
        for transform in same_layer
    )
    return cases


class NumericTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.prior_threads)

    @torch.no_grad()
    def compare_cases(self, model, fixture_name, layer_activations, shape, checkpoint=None, controls=True):
        generator = torch.Generator().manual_seed(INPUT_SEED)
        if checkpoint is None:
            inputs = torch.randn(shape, generator=generator, dtype=torch.float64)
            distribution = "standard normal"
        else:
            inputs = 2 * torch.rand(shape, generator=generator, dtype=torch.float64) - 1
            distribution = "uniform [-1, 1)"
        original = model(inputs)
        source_state = {name: param.clone() for name, param in model.state_dict().items()}
        cases = transform_cases(model, layer_activations)
        if controls:
            cases["shuffle_weights"] = ()

        for label, transforms in cases.items():
            with self.subTest(fixture=fixture_name, transform=label):
                first_layer = layer_activations[0][0]
                if label == "shuffle_weights":
                    transformed = shuffle_weights(model, first_layer, seed=SHUFFLE_SEED)
                    operations = [{"type": label, "layer": first_layer, "seed": SHUFFLE_SEED}]
                else:
                    transformed = apply_transforms(model, *transforms)
                    operations = [{"type": type(operation).__name__} | asdict(operation) for operation in transforms]

                output = transformed(inputs)
                self.assertEqual(original.dtype, torch.float64)
                self.assertEqual(output.dtype, torch.float64)
                self.assertEqual(inputs.shape[0], INPUT_COUNT)
                self.assertEqual(original.shape, output.shape)
                self.assertTrue(original.isfinite().all())
                self.assertTrue(output.isfinite().all())
                differences = (original - output).abs().flatten(1).amax(1)
                max_difference = differences.max().item()
                shuffled = label == "shuffle_weights"
                passed = max_difference > TOLERANCE if shuffled else max_difference < TOLERANCE
                transformed_state = transformed.state_dict()
                changed = [name for name, param in source_state.items()
                           if not torch.equal(param, transformed_state[name])]

                write_metric({
                    "fixture": fixture_name,
                    "checkpoint": checkpoint,
                    "synthetic_model_seed": MODEL_SEED if checkpoint is None else None,
                    "input_seed": INPUT_SEED,
                    "input_dtype": str(inputs.dtype),
                    "input_distribution": distribution,
                    "input_shape": list(inputs.shape),
                    "input_sha256": hashlib.sha256(inputs.numpy().tobytes()).hexdigest(),
                    "output_shape": list(output.shape),
                    "transform_seed": TRANSFORM_SEED if not shuffled else SHUFFLE_SEED,
                    "transform_case": label,
                    "operations": operations,
                    "changed_parameters": changed,
                    "per_input_max_abs_logit_difference": differences.tolist(),
                    "max_abs_logit_difference": max_difference,
                    "criterion": ">" if shuffled else "<",
                    "tolerance": TOLERANCE,
                    "passed": passed,
                })
                self.assertTrue(passed, f"{fixture_name} {label}: max difference {max_difference}")
                self.assertTrue(changed, "expected an actual change to the weights")
                for name, current in model.state_dict().items():
                    self.assertTrue(torch.equal(current, source_state[name]), f"source {name} mutated")
                    self.assertNotEqual(current.data_ptr(), transformed_state[name].data_ptr())
                if shuffled:
                    self.assertEqual(changed, [f"{first_layer}.weight"])
                    self.assertTrue(torch.equal(
                        source_state[changed[0]].flatten().sort().values,
                        transformed_state[changed[0]].flatten().sort().values,
                    ))

    def test_relu_mlp(self):
        self.compare_cases(toy_mlp(nn.ReLU, nn.ReLU), "toy_mlp_relu", [(0, nn.ReLU), (2, nn.ReLU)],
                           (INPUT_COUNT, 5))

    def test_tanh_mlp(self):
        self.compare_cases(toy_mlp(nn.Tanh, nn.Tanh), "toy_mlp_tanh", [(0, nn.Tanh), (2, nn.Tanh)],
                           (INPUT_COUNT, 5))

    def test_mixed_mlp(self):
        self.compare_cases(toy_mlp(nn.ReLU, nn.Tanh), "toy_mlp_mixed", [(0, nn.ReLU), (2, nn.Tanh)],
                           (INPUT_COUNT, 5), controls=False)

    def test_relu_flattened_cnn(self):
        self.compare_cases(synthetic_flat_cnn(nn.ReLU), "synthetic_flat_cnn_relu",
                           [(0, nn.ReLU), (2, nn.ReLU)], (INPUT_COUNT, 1, 8, 8))

    def test_tanh_flattened_cnn(self):
        self.compare_cases(synthetic_flat_cnn(nn.Tanh), "synthetic_flat_cnn_tanh",
                           [(0, nn.Tanh), (2, nn.Tanh)], (INPUT_COUNT, 1, 8, 8))

    def compare_cnn_zoo(self, name, activation):
        directory = Path(FIXTURES)
        fixture = json.loads((directory / "manifest.json").read_text())["fixtures"][name]
        checkpoint = {key: fixture[key] for key in ("path", "sha256", "zero_based_archive_row")}
        checkpoint["activation"] = fixture["metrics_row"]["config.activation"]
        checkpoint["step"] = int(fixture["metrics_row"]["step"])
        self.compare_cases(load_cifar10_fixture(directory, name), f"small_cnn_zoo_cifar10_gs_{name}",
                           [(0, activation), (2, activation), (4, activation)],
                           (INPUT_COUNT, 1, 32, 32), checkpoint)

    @REQUIRES_FIXTURES
    def test_relu_cnn_zoo(self):
        self.compare_cnn_zoo("relu", nn.ReLU)

    @REQUIRES_FIXTURES
    def test_tanh_cnn_zoo(self):
        self.compare_cnn_zoo("tanh", nn.Tanh)

    @REQUIRES_FIXTURES
    @torch.no_grad()
    def test_zoo_loader_matches_independent_nhwc_reference(self):
        # NumPy interprets the release HWIO kernels directly, independent of the loader.
        generator = np.random.default_rng(INPUT_SEED)
        inputs = generator.uniform(-1, 1, size=(8, 32, 32, 1))
        directory = Path(FIXTURES)
        manifest = json.loads((directory / "manifest.json").read_text())
        for activation, fixture in manifest["fixtures"].items():
            with self.subTest(activation=activation):
                weights = np.load(directory / fixture["path"], allow_pickle=False).astype(np.float64)
                nhwc = inputs
                for start, end, channels in [(0, 160, 1), (160, 2480, 16), (2480, 4800, 16)]:
                    bias = weights[start:start + 16]
                    kernel = weights[start + 16:end].reshape(3, 3, channels, 16)
                    patches = sliding_window_view(nhwc, (3, 3), axis=(1, 2))[:, ::2, ::2]
                    nhwc = np.einsum("nxycij,ijco->nxyo", patches, kernel) + bias
                    nhwc = np.maximum(nhwc, 0) if activation == "relu" else np.tanh(nhwc)
                expected = nhwc.mean(axis=(1, 2)) @ weights[4810:].reshape(16, 10) + weights[4800:4810]
                pytorch_inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).contiguous()
                actual = load_cifar10_fixture(directory, activation)(pytorch_inputs)
                np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=1e-12)


class TransformContractTests(unittest.TestCase):
    def test_flatten_column_order_and_bias(self):
        model = synthetic_flat_cnn(nn.Tanh)
        order = (2, 0, 4, 3, 1)
        transformed = apply_transforms(model, Permute(layer=2, order=order))
        torch.testing.assert_close(transformed[2].weight, model[2].weight[list(order)], rtol=0, atol=0)
        torch.testing.assert_close(transformed[2].bias, model[2].bias[list(order)], rtol=0, atol=0)
        expected = torch.cat([model[-1].weight[:, i * 4:(i + 1) * 4] for i in order], dim=1)
        torch.testing.assert_close(transformed[-1].weight, expected, rtol=0, atol=0)
        torch.testing.assert_close(transformed[-1].bias, model[-1].bias, rtol=0, atol=0)

    def test_composition_preserves_order(self):
        model = toy_mlp(nn.ReLU, nn.Tanh)
        order = tuple(reversed(range(9)))
        scales = tuple(float(unit + 1) for unit in range(9))
        flips = TanhSignFlip(2, (-1, 1, -1, 1, -1, 1, -1))
        composed = apply_transforms(model, Permute(0, order), ReluScale(0, scales), flips)
        reversed_order = apply_transforms(model, ReluScale(0, scales), Permute(0, order), flips)
        expected = model[0].weight[list(order)] * torch.tensor(scales, dtype=torch.float64)[:, None]
        torch.testing.assert_close(composed[0].weight, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(composed[0].weight, reversed_order[0].weight))

    def test_successor_bias_untouched_and_biasless_path_supported(self):
        model = toy_mlp(nn.ReLU, nn.Tanh)
        transformed = apply_transforms(model, TanhSignFlip(2, (-1,) * 7))
        self.assertIsNone(transformed[2].bias)
        torch.testing.assert_close(transformed[2].weight, -model[2].weight, rtol=0, atol=0)
        torch.testing.assert_close(transformed[4].weight, -model[4].weight, rtol=0, atol=0)
        torch.testing.assert_close(transformed[4].bias, model[4].bias, rtol=0, atol=0)

    def test_shuffle_is_repeatable(self):
        model = toy_mlp(nn.ReLU, nn.ReLU)
        first = shuffle_weights(model, 0, seed=SHUFFLE_SEED)
        second = shuffle_weights(model, 0, seed=SHUFFLE_SEED)
        torch.testing.assert_close(first[0].weight, second[0].weight, rtol=0, atol=0)
        self.assertFalse(torch.equal(first[0].weight, model[0].weight))

    def test_shuffle_rejects_constant_layer(self):
        model = toy_mlp(nn.ReLU, nn.ReLU)
        nn.init.ones_(model[0].weight)
        with self.assertRaisesRegex(ValueError, "distinct"):
            shuffle_weights(model, 0, seed=SHUFFLE_SEED)

    def test_invalid_transforms_do_not_mutate_the_original(self):
        model = toy_mlp(nn.ReLU, nn.Tanh)
        original = model[0].weight.detach().clone()
        valid = Permute(0, tuple(reversed(range(9))))
        invalid = [
            (Permute(0, (0,) * 9), "permutation"),
            (Permute(0, (0.0,) * 9), "integers"),
            (Permute(0, tuple(range(8))), "9 integers"),
            (Permute(-1, (0,)), "outside"),
            (Permute(4, (0, 1, 2, 3)), "succeeding"),
            (Permute(1, (0,)), "must be a Linear"),
            (ReluScale(0, (1,) * 8 + (0,)), "positive"),
            (ReluScale(0, (1,) * 8 + (-1,)), "positive"),
            (ReluScale(0, (1,) * 8 + (math.nan,)), "positive"),
            (ReluScale(0, (1,) * 8 + (math.inf,)), "positive"),
            (TanhSignFlip(2, (1,) * 6 + (0,)), "must be -1 or 1"),
            (TanhSignFlip(0, (-1,) * 9), "does not support ReLU"),
            (ReluScale(2, (1.5,) * 7), "does not support Tanh"),
        ]
        for transform, error in invalid:
            with self.subTest(transform=transform):
                with self.assertRaisesRegex(ValueError, error):
                    apply_transforms(model, valid, transform)
                torch.testing.assert_close(model[0].weight, original, rtol=0, atol=0)

    def test_missing_activation_rejects_activation_specific_transforms(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.Identity(), nn.Linear(4, 2))
        with self.assertRaisesRegex(ValueError, "requires ReLU"):
            apply_transforms(model, ReluScale(0, (1.5,) * 4))
        with self.assertRaisesRegex(ValueError, "requires Tanh"):
            apply_transforms(model, TanhSignFlip(0, (-1,) * 4))

    def test_max_pool_rejects_tanh_sign_flip(self):
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3), nn.Tanh(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16, 3)
        )
        with self.assertRaisesRegex(ValueError, "MaxPool2d"):
            apply_transforms(model, TanhSignFlip(0, (-1, 1, 1, -1)))

    def test_dropout_must_be_in_eval_mode(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Dropout(), nn.Linear(4, 2))
        transform = Permute(0, (3, 2, 1, 0))
        with self.assertRaisesRegex(ValueError, "Dropout"):
            apply_transforms(model, transform)
        transformed = apply_transforms(model.eval(), transform)
        self.assertFalse(transformed.training)

    def test_grouped_convolution_and_unsupported_flatten_axes_rejected(self):
        grouped = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU(), nn.Conv2d(4, 4, 1, groups=2))
        with self.assertRaisesRegex(ValueError, "grouped"):
            apply_transforms(grouped, Permute(0, (3, 2, 1, 0)))
        flattened = nn.Sequential(nn.Conv2d(1, 4, 1), nn.ReLU(), nn.Flatten(2), nn.Linear(16, 3))
        with self.assertRaisesRegex(ValueError, "Flatten"):
            apply_transforms(flattened, Permute(0, (3, 2, 1, 0)))

    def test_rejects_tied_parameter_storage(self):
        first, second = nn.Linear(4, 4), nn.Linear(4, 4)
        second.weight = nn.Parameter(first.weight.detach())
        model = nn.Sequential(first, nn.ReLU(), second)
        with self.assertRaisesRegex(ValueError, "shared parameter storage"):
            apply_transforms(model, Permute(0, (3, 2, 1, 0)))

    def test_rejects_scales_unrepresentable_in_layer_dtype(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        with self.assertRaisesRegex(ValueError, "reciprocals"):
            apply_transforms(model, ReluScale(0, (1e-50,) * 4))


if __name__ == "__main__":
    unittest.main()
