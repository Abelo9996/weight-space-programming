# Policy repair

The comparison changes a failing policy's weight coordinates while keeping its initial input-output function fixed. Each method gets the original or the transformed checkpoint, a task specification, and access to an evaluator for development starts. The evaluator scores the selected candidate on separate starts.

## Prepare tasks

From the repository root, install the CPU workflow and generate fixtures locally.

```sh
python -m pip install -e '.[prepare,transforms]'
weightspace prepare --output data/policies --seed 0 \
  --task TR02-CTRL-002 --task TR02-CTRL-003 --task TR02-CTRL-004
```

| Task | Repair objective | Horizon |
| --- | --- | ---: |
| `TR02-CTRL-002` MountainCar | Reach the goal from failure starts and retain success on normal starts | 200 |
| `TR02-CTRL-003` Pendulum | Swing up from low-energy starts and retain balance from upright starts | 200 |
| `TR02-CTRL-004` CartPole | Keep the pole upright and the cart within the track limits | 500 |

Preparation uses CPU behavior cloning with the upstream flawed expert and pruned ReLU networks. It classifies start states by the initial policy's success, then samples disjoint failure and normal subsets for development and held-out evaluation. It records the task, seed, dependency versions, and digests alongside the checkpoint and splits. Use a fresh fixture directory for each seed. Set `fixture_seed` in the experiment config to the same seed and point `--fixtures` at the prepared directory to override the config's fixture path. Compare fixture digests before pooling results across hosts, as CPU training can vary across numerical libraries. The [upstream code and license](related-work.md#software-and-data) are documented separately.

After generating all three seed-zero tasks, a development installation can run the additional fixture checks.

```sh
WEIGHTSPACE_TEST_FIXTURES=data/policies python -m pytest -ra tests/core/test_tasks.py
```

## Compare methods

See the [README](../README.md#run-a-cpu-smoke-test) for the offline workflow. Search and editor proposals are ordered JSON lists applied to the starting checkpoint by a shared validator. The four primitives scale a hidden neuron's incoming weights and bias, zero its outgoing weights, shift a bias, or add a rank-one matrix. Bounds on magnitudes, operations and changed entries are common across methods. The compensated transform used to make an equivalent starting checkpoint is a separate operation.

- **No edit** selects the starting policy.
- **Random search** independently samples lists of one to four operations, subject to the configured operation limit. It samples the primitive, an eligible layer, and the relevant coordinates. The default scale factors are `0.5`, `0.8`, `1.25`, and `2`. Bias shifts and rank-one coefficients use `±0.25` or `±1`. Each rank-one vector has up to four positive entries. Configured bounds can clip the sampled magnitudes.
- **Greedy coordinate search** samples a coordinate and tests opposing magnitudes where the primitive supports them. Rank-one trials use one nonzero entry per vector. The new trial appends to the incumbent edit list or replaces an operation when that list is at its limit.
- **Evolution strategy** generates a fixed number of offspring from the current parent by adding, replacing, or deleting an operation. The best candidate among the parent and scored offspring survives.
- **Language-model editor** can propose any edit list that passes the shared validator and select any scored candidate, including the starting policy.

Search ranks candidates by the development objective, the mean of failure recovery and retention. Ties favor fewer changed entries and then earlier candidates. The comparisons apply to these specific search distributions and editor instructions.

The unedited development probe uses one candidate slot. Each subsequent valid probe uses a slot and reserves the worst-case number of policy rows, calculated as the number of development starts times the task horizon. A rejected edit uses no slot. A probe that fails during evaluation keeps its reservation and incurred row count. One actual row is one policy forward for an active start at a simulator step. Early termination reduces the actual count, while the fixed reservation makes the candidate allowance independent of termination speed.

Methods share candidate and policy-row ceilings. Realized counts can differ because of early selection, rejection limits, failures, or episode termination. Report candidate counts, reserved and used rows, and editor CLI launches. The broker seals development before evaluating the selected candidate on its held-out starts and accounts for those rows separately.

## Optional live editor

Both native adapters use the same proposal protocol, serialized weights, and development feedback. Select one editor and pin its model for all conditions of a comparison. Keep raw prompts and responses local pending review. For an evaluation whose starts must stay private, use a private fixture seed and config until evaluation is complete.

### Codex CLI

The [Codex pilot config](../examples/codex-pilot.json) runs one MountainCar pair. Each editing or search cell has two candidate slots, including the starting checkpoint, so it allows one additional edit. The two Codex cells can launch the CLI a total of four times with zero harness retries. The CLI can retry provider requests internally. The pilot exercises the pipeline with a single paired seed.

```sh
weightspace prepare --output examples/fixtures/codex-pilot --task TR02-CTRL-002 --seed 90217
weightspace run --config examples/codex-pilot.json --output runs/codex-pilot --allow-live-editor
weightspace summarize --input runs/codex-pilot --output runs/codex-pilot/summary.json
```

The adapter checks the configured CLI version and native ChatGPT login, forces ChatGPT authentication with the built-in OpenAI provider, and filters out environment credentials and endpoint overrides. Each request runs in a fresh working directory. The command skips user configuration and rules, requests disabled shell and connector features, and supplies the shared protocol as replacement instructions. It accepts a single assistant message followed by usage and checks it with the shared strict JSON parser. Stream checks cover activity reported by the CLI. The native process still uses the local OS account and its credential store, system-managed settings may apply, and the model catalog can enable built-in utilities. [Codex CLI reference](https://developers.openai.com/codex/cli/reference) · [Configuration reference](https://developers.openai.com/codex/config-reference)

The journal records the requested model ID, CLI version, observed token usage, and bounded output for accepted proposals. The backend model identity is unavailable in Codex CLI 0.169.0's JSONL. For a rejected stream the journal retains event types, available numeric usage and output digests. Hard CLI failures stop the experiment before the next cell. The configured token limits are acceptance checks applied after the CLI reports usage. The journal records subscription charges as unknown. [Codex 0.169.0 stream types](https://github.com/openai/codex/blob/rust-v0.169.0/codex-rs/exec/src/exec_events.rs)

### Claude Code CLI

The Claude Code CLI adapter uses a first-party subscription login on Linux or macOS. The example assumes extra usage is disabled on the subscription. Review the task, call limits, and pinned model in [the Claude pilot config](../examples/pilot.json) before running it.

```sh
weightspace prepare --output examples/fixtures/pilot --task TR02-CTRL-002 --seed 84103
weightspace run --config examples/pilot.json --output runs/pilot --allow-live-editor
weightspace summarize --input runs/pilot --output runs/pilot/summary.json
```

The adapter requests an empty tool and connector configuration, checks emitted CLI metadata, logs the requested and reported model identities, and captures bounded raw output and usage for audit.

The broker keeps held-out starts and per-episode held-out feedback out of the editor protocol. The CLI starts in a temporary working directory and still runs with the local OS account's permissions. Organization-managed hooks can run even in safe mode. The stream checks cover behavior reported by the CLI, so use a trusted host and verify the applicable managed configuration. Use OS-level isolation for untrusted executable methods. [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference) · [Managed settings](https://code.claude.com/docs/en/settings)

The experiment manifest records a ceiling on CLI launches for edit proposals and selection. The CLI can make internal provider requests during a single launch. Its `--max-budget-usd` setting applies a client-side threshold to estimated API-equivalent spend. Estimates can overshoot that threshold and differ from charges. Treat the sum of thresholds as a record of configured values. For a strict dollar limit, verify account-side spending protection before running the experiment. Disable usage credits for a subscription-only run and verify the setting in the account dashboard. [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference) · [Claude usage credits](https://support.claude.com/en/articles/12429409-manage-usage-credits-for-paid-claude-plans)

## Read the summary

Use a fresh output directory per experiment. The experiment manifest records the config. Each cell has a run manifest and append-only event log alongside its starting and selected checkpoints. Failed cells can leave partial artifacts. The summary reads the experiment directory and records the status of every planned cell.

Pair original and transformed runs on task, fixture and repeat. Keep the same editor model, tool, budget, information and start states within each pair. Check the numerical equivalence record before interpreting any difference. Discrete policy actions and continuous feedback can amplify rounding differences, so the equivalence check and the recorded tolerance matter.

Report the held-out repair rate, retention rate, and objective together with the development-to-held-out change. The sensitivity contrast subtracts transformed performance from original performance. A difference can arise because a fixed coordinate-space edit has a different behavioral effect after rescaling. The editor comparison includes each search baseline and the best mean search score on the matched seeds. The summary bootstraps whole paired seeds and recomputes that best mean in each resample. Its intervals are conditional on the tasks, starting policies, and held-out panel. A small smoke run checks the pipeline. More independent repetitions and task variation are needed to estimate reliability.
