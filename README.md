# Weight-Space Programming

Does a language-model editor repair the same policy equally well when its weights are rearranged without changing its starting function? Does it beat search when both use the same edit primitives and policy-evaluation limits?

This research package pairs equivalent policy checkpoints, runs search baselines and optional language-model editors, and evaluates selected edits on held-out starts. It also includes PyTorch transforms and a reproducible Small CNN Zoo fixture check for a separate study of weight reading. The fixture checks validate the transforms and model loader. The CNN Zoo reading comparison is a research protocol described in the documentation.

## Run a CPU smoke test

Use Python 3.11 or 3.12 from the repository root. Task preparation and the paired transform require PyTorch. The smoke config runs local baselines and requires no model-provider account.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -ra

weightspace prepare --output examples/fixtures/smoke --task TR02-CTRL-002 --seed 0
weightspace run --config examples/smoke.json --output runs/smoke
weightspace summarize --input runs/smoke --output runs/smoke/summary.json
```

The base package uses NumPy. Use `pip install -e '.[prepare,transforms]'` for experiment workflows without development tools. On Linux you can install a [CPU PyTorch wheel](https://pytorch.org/get-started/locally/) before installing the extras. The CLI also runs as `python -m weightspace`.

## Experiments and reference material

- [Policy repair](docs/policy-repair.md) explains task preparation, search and editor runs, budget units, and how to read held-out results.
- [Weight transforms](docs/transforms.md) documents the PyTorch API, numerical checks, and shuffle control.
- [Small CNN Zoo](docs/cnn-zoo.md) documents optional fixture acquisition and the proposed reading comparison.
- [Related work and attribution](docs/related-work.md) lists relevant precedent and licenses.
- [Contributing](CONTRIBUTING.md) describes how to share changes and reproducible research outputs.

Software citation metadata is in [CITATION.cff](CITATION.cff). Original code is available under the [MIT License](LICENSE). Vendored components retain their upstream licenses.
