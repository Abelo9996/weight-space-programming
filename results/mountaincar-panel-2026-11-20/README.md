# MountainCar policy repair comparison

**59 of 60 planned cells completed.** The original-form Codex cell for fixture seed `202611205` failed after an invalid first-turn proposal. Its initial checkpoint received a development score. The cell reached no final selection or held-out evaluation. The logger retained a stdout hash and discarded the rejected response and raw stdout. The exact response-shape defect is unavailable. All other cells completed.

**The primary five-fixture contrast and its interval are withheld.** Four of five fixtures had both Codex forms available. The [frozen protocol](protocol.md) requires every planned fixture pair. That requirement also withholds all five-fixture Codex versus search and Codex versus unchanged contrasts, and the Codex original-minus-transformed contrast. Compound ES had the highest mean development objective among compound methods. Its prespecified five-fixture contrast with Codex is also withheld.

## Observed held-out scores

The objective gives equal weight to recovering from failure starts and retaining success on normal starts. Every completed cell retained success on all 100 normal starts. The table shows **descriptive means among completed cells**. The original Codex group has one fewer fixture than the other groups. These means do not supply a paired comparison.

| Method | Original mean (n) | Compensated permutation and ReLU scale mean (n) |
| --- | ---: | ---: |
| Unchanged | 0.500 (5) | 0.500 (5) |
| Compound random | 0.523 (5) | 0.500 (5) |
| Compound greedy | 0.500 (5) | 0.500 (5) |
| Compound ES | 0.572 (5) | 0.524 (5) |
| Analytic velocity controller | 0.998 (5) | 0.998 (5) |
| Codex | 0.976 (4) | 0.911 (5) |

The analytic controller recovered 98 to 100 failure starts per completed cell. It compiles the sign of velocity into masked ReLU channels as a task-specific control rule. All nine completed Codex cells selected an edit and recovered 41 to 100 failure starts. The individual held-out objectives for the primary comparison show where observations exist.

| Fixture seed | Original Codex | Original velocity controller | Transformed Codex | Transformed velocity controller |
| ---: | ---: | ---: | ---: | ---: |
| 202611201 | 0.990 | 1.000 | 0.945 | 1.000 |
| 202611202 | 0.970 | 0.990 | 1.000 | 0.990 |
| 202611203 | 0.945 | 1.000 | 0.935 | 1.000 |
| 202611204 | 1.000 | 1.000 | 0.705 | 1.000 |
| 202611205 | Failed | 1.000 | 0.970 | 1.000 |

The compound search baselines and the editor shared a two-slot development budget, including the initial checkpoint. Greedy and ES ran only their initial proposal stage. The experiment spans five separately seeded checkpoints and their checkpoint-dependent start panels from one MountainCar generator. Each fixture and form has one editor replicate, so any observed form difference can incorporate editor variation. Five fixtures also give limited precision for the complete non-editor contrasts.

## Budget and native editor

The broker scored 109 development candidates and structurally rejected two recipes before reservation. Development evaluation used 250,968 of 348,800 reserved rows. The 59 selected held-out evaluations used 1,588,927 of 2,360,000 reserved rows. The protocol records the candidate, row, and native launch caps separately.

The run journal records 19 model-command attempts with reported usage. Codex CLI `0.169.0` requested `gpt-6.2-sol` at `low` reasoning effort through a native ChatGPT subscription. The recorded totals were 1,696,499 input tokens and 16,993 output tokens. The CLI request identifies the configured model. Backend serving identity and actual subscription charge are unavailable.

## Data and fixed-recipe replay

- [cells.csv](cells.csv) records all 60 planned statuses and the available selected outcomes, component counts, row budgets and hashes. A blank score means the corresponding outcome is unavailable.
- [summary.json](summary.json) records the arm-by-form descriptive statistics, all prespecified contrasts and available within-fixture differences, and native attempt usage. The withheld aggregate estimates and intervals are null.
- [candidates.jsonl](candidates.jsonl) records structured development requests, reservations and results. Candidate IDs are local to a cell. A null request recipe indicates that the attempt lacks a publishable operation list.
- [config.json](config.json) records replay parameters, transform operations, and hashes of the frozen plan, job configs, splits and starting policies. [manifest.json](manifest.json) records source, package, release file and fixture archive hashes.
- The [fixture archive](https://github.com/Abelo9996/weight-space-programming/releases/download/mountaincar-panel-2026-11-20/mountaincar-panel-2026-11-20-fixtures.tar.gz) contains the exact five starting checkpoints, start sets and demonstrations, with attribution and checksums. These held-out starts are public historical evaluation data. Use fresh starts in prospective experiments.

From the repository root, run the following with Python 3.12. The run used Python 3.12.13, NumPy 2.2.6 and PyTorch 2.7.1 with single-thread numerical settings. Exact agreement can depend on the CPU and numerical libraries. Replay uses the existing fixture bytes and needs no training, account login, catalog or model calls.

```sh
replay_dir="$(mktemp -d)"
mkdir "$replay_dir/source"
git archive 88707c6a475f71b7397be9807374a82d4c212a46 | tar -x -C "$replay_dir/source"
python3.12 -m venv "$replay_dir/venv"
"$replay_dir/venv/bin/python" -m pip install \
  'numpy==2.2.6' 'torch==2.7.1' "$replay_dir/source"

archive_name=mountaincar-panel-2026-11-20-fixtures.tar.gz
curl -fL -o "$replay_dir/$archive_name" \
  "https://github.com/Abelo9996/weight-space-programming/releases/download/mountaincar-panel-2026-11-20/$archive_name"
printf '%s  %s\n' feeccf39aefb3701a446055b0704752788f1e04404a6a2c5955940b353128c83 \
  "$replay_dir/$archive_name" | shasum -a 256 -c -

VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$replay_dir/venv/bin/python" scripts/export_mountaincar_panel.py replay \
  --results results/mountaincar-panel-2026-11-20 \
  --fixture-archive "$replay_dir/$archive_name"
```
