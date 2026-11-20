# Weight-Space Programming

Does a language-model editor repair the same policy equally well when its weights are rearranged without changing its starting function? Does it beat search when both use the same edit primitives and policy-evaluation limits?

This research package pairs equivalent policy checkpoints, runs search baselines and optional language-model editors, and evaluates selected edits on held-out starts. It also provides CNN accuracy readers, a train-only layer-statistics baseline, and checked PyTorch transforms for ReLU and tanh networks. The reading workflow supports offline planning before funded API runs.

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
- [Offline policy repair pilot](results/baseline-pilot-2026-11-19/) reports three CPU tasks with paired search baselines, full event data, and replay instructions.
- [Five-fixture MountainCar comparison](results/mountaincar-panel-2026-11-20/) reports Codex, compound search, and an analytic velocity controller under a shared policy-evaluation budget, with fixed-recipe replay.
- [Weight transforms](docs/transforms.md) documents the PyTorch API, numerical checks, and shuffle control.
- [CNN accuracy reading](docs/cnn-reading.md) explains labeled panels, API planning, prediction scoring, and the statistics baseline.
- [Small CNN Zoo](docs/cnn-zoo.md) documents optional fixture acquisition, source metadata, and the larger benchmark protocol.
- [Related work and attribution](docs/related-work.md) lists relevant precedent and licenses.
- [Contributing](CONTRIBUTING.md) describes how to share changes and reproducible research outputs.

Software citation metadata is in [CITATION.cff](CITATION.cff). Original code is available under the [MIT License](LICENSE). Vendored components retain their upstream licenses.
