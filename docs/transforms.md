# Weight-space transforms

Run the CPU float64 tests from the repository root. The synthetic MLP and CNN cases run offline. The archived CNN Zoo tests run when `WEIGHTSPACE_CNN_ZOO_FIXTURES` points to [fetched fixtures](cnn-zoo.md).

```sh
python -m pip install -e '.[dev]'
python -m pytest -ra tests/transforms
```

Set `WSP_TRANSFORM_LOG` to an existing directory's JSONL file path to append numeric cases with the input and transform seeds, operations, and per-input logit discrepancies. `WSP_TRANSFORM_RUN_ID` can supply a run label. A run reuses its input batch across the different transformations of a fixture.

```sh
mkdir -p runs/transforms
WSP_TRANSFORM_LOG=runs/transforms/cases.jsonl \
  python -m unittest discover -s tests/transforms -v
```

## API

```python
from torch import nn
from weightspace.transforms import Permute, ReluScale, apply_transforms, shuffle_weights

model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
equivalent = apply_transforms(
    model,
    Permute(layer=0, order=(2, 0, 3, 1)),
    ReluScale(layer=0, scales=(1.5, 0.5, 3.0, 2.0)),
)
control = shuffle_weights(model, 0, seed=12)
```

The layer index refers to an entry in a flat PyTorch `nn.Sequential`. A permutation puts the original unit at `order[i]` into position `i`. `ReluScale(layer, scales)` requires explicit ReLU and finite positive factors. `TanhSignFlip(layer, signs)` requires explicit tanh and signs of -1 or +1. The transforms update the incoming weight rows and biases, then compensate the next affine layer's input weights. Compositions run from left to right, so each operation uses the unit order produced by earlier operations.

The supported affine layers are `nn.Linear` and ungrouped `nn.Conv2d`. The channelwise path can contain the corresponding activation, identity, evaluation-mode dropout, and convolutional average pooling. Permutations also support standard elementwise activations. Permutations and positive ReLU scales support channelwise max pooling without returned indices. Tanh sign flips require average pooling or direct input consumption. A convolution-to-dense path requires `nn.Flatten(1, -1)` in NCHW order. Each channel corresponds to a contiguous block of the dense input weights.

Both entry points return independent model copies. They raise an error for unsupported paths, incompatible arguments, hooks, and shared parameter storage. Use these routines for inference checkpoints. For a live training run, updating an optimizer's state would require a separate transformation.

The control shuffles the scalar values of one weight tensor using a seeded permutation and preserves its bias. A network can be insensitive to changes in a particular layer. The numerical tests check that shuffling changes logits by more than `1e-5` and that the symmetry transforms change them by less than `1e-5` over 1,000 seeded float64 inputs per case.

Install `.[transforms]` to use the API without the development test tools.
