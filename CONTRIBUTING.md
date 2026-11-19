# Contributing

Create a branch for a focused code or protocol change. Open an issue for a new experiment or a change to an existing comparison and include the question, conditions, shared budget, outcome measure, and planned analysis. Link the issue from the pull request and include reproducing commands and their results.

Run the offline test suite before opening a pull request.

```sh
python -m pip install -e '.[dev]'
python -m pytest -ra
python -m build
```

Document added tasks with the upstream revision, license, training and evaluation assumptions, fixture generation command, and split derivation. For changes to policy comparisons, include an offline run and explain any changes to candidate selection, budget accounting, or scoring. For transforms, include seeded float64 equivalence checks and a function-changing control where applicable.

Commit source, documentation, and small example configs. Keep generated datasets, checkpoints, evaluation splits, reference edits, and raw run logs in local output directories. Review experiment exports before sharing them. Include the source commit, config, fixture digests, seeds, and tool versions so another contributor can reproduce a figure or table. Keep account credentials out of configs and logs.
