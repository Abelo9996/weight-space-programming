# CNN accuracy reading

The reader predicts a CNN's held-out top-1 accuracy as a number from 0 to 1. OpenAI Responses and Anthropic Messages receive the same prompt text for a given network, condition, and example count. Text contains the evaluation architecture, round-trip float64 parameters, and any labeled training examples. The target's accuracy, checkpoint identity, split metadata, dataset name, and local paths stay outside the prompt. [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create) · [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create)

## Offline smoke with the CNN Zoo fixtures

Install the reader and point it to an already available [CNN Zoo fixture directory](cnn-zoo.md). The commands work from any directory after installation. Paths to the example config and fixture directory can be absolute.

```sh
python -m pip install '.[reading]'
weightspace reading fixture-manifest --fixtures data/cnn_zoo_fixtures --output cnn-fixtures.json
weightspace reading plan --config examples/reading-smoke.json --panel cnn-fixtures.json --output cnn-plan.json
```

The two archived checkpoints are nonrandom numerical fixtures, one ReLU and one tanh. Both are assigned to the test split. This smoke checks the real checkpoint loader, response schedule, and transformations with no model calls. It cannot estimate benchmark performance or train the statistics baseline. The archived test accuracies are release metadata.

The plan lists every request, provider and requested model, estimated tokens, estimated reservations, prompt and body SHA-256 hashes, and any live preflight blockers. Example configurations set funding to zero. The funded smoke schedules 60 requests across 2 networks, 5 conditions, 3 repeats, and 2 readers. `reading-panel.json` illustrates a schedule for three readers with 0, 8, and 32 examples. At 32 examples, full-text CNN Zoo prompts can exceed the example models' context estimates. Preflight requires enough training networks and identifies any oversized configuration before execution.

## Prepare a labeled panel

Specify the split for each original checkpoint before running any analysis. Use a stable source training-run identifier for all checkpoints and weight representations derived from that training run. Select one original checkpoint per training run. Checkpoint paths are relative to the panel manifest. This example shows one row with a placeholder file digest. A benchmark panel also needs explicit training rows for the statistics baseline or in-context examples.

```json
{
  "schema_version": 1,
  "dataset": {
    "name": "Small CNN Zoo CIFAR-10-GS final-checkpoint panel",
    "source": "https://storage.googleapis.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz?generation=1617698236426937",
    "sampling": "panel"
  },
  "architecture": {
    "input_shape": [1, 32, 32],
    "convolutions": [
      {"out_channels": 16, "kernel_size": 3, "stride": 2},
      {"out_channels": 16, "kernel_size": 3, "stride": 2},
      {"out_channels": 16, "kernel_size": 3, "stride": 2}
    ],
    "classes": 10
  },
  "networks": [
    {
      "id": "test-001",
      "training_run_id": "release-training-run-001",
      "split": "test",
      "accuracy": 0.61,
      "activation": "relu",
      "checkpoint": {
        "format": "pytorch-npz",
        "path": "checkpoints/test-001.npz",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "source_checkpoint": "release-checkpoint-001",
        "layout": "state_dict_oihw_oi"
      }
    }
  ]
}
```

The NPZ format uses float32 or float64 arrays and loads with pickle disabled. The CNN uses biased, valid convolutions, ReLU or tanh after each convolution, global average pooling, flattening, and a biased linear classification head. Store convolution weights in output-channel, input-channel, kernel-row, kernel-column order, and head weights in output-class, input-channel order. With three convolution blocks, the array names are `0.weight`, `0.bias`, `2.weight`, `2.bias`, `4.weight`, `4.bias`, `8.weight`, and `8.bias`. More generally, each convolution and activation occupies two sequential module indices. The head follows the two pooling and flattening modules. For a model in this module order, use `np.savez(path, **{name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()})` and record the SHA-256 of the file. For the archived two-row format, use the fixture manifest command. It checks the release row, activation, test accuracy, and file and layout hashes through the CNN Zoo loader.

The loader rejects duplicate normalized run IDs, source checkpoint IDs, display IDs, and identical tensors across the panel. Generated variants inherit the original network identity. Planning checks for a generated variant whose tensors duplicate another listed network. Grouping externally prepared checkpoints still depends on accurate source training-run identifiers. Keep dataset sampling and split assignment fixed for model and baseline comparisons.

## Request protocol and funding

Each test network has five conditions. The original is unmodified. Three equivalent variants use a channel permutation, activation-appropriate positive ReLU scales or tanh signs, and an independently seeded composition. The shuffle control permutes scalar weights in the first convolution for which this changes the verified logits. The plan checks all variant logits on 1,000 fixed uniform float64 inputs. It requires a maximum absolute difference below `1e-5` for equivalent variants and above `1e-5` for the control. These unlabeled inputs serve as numerical probes. Independent requests repeat within each condition. Example selection is deterministic and nested across example counts, and uses only the explicit train split.

Review the planned requests and current model rates. Set a positive `budget_usd` and a positive `confirmed_credits_usd` balance for each selected provider in a local config copy. The combined budget and each provider's confirmed balance must cover the corresponding full projection. Jobs estimated above USD 40 require separate approval and the `--approve-over-40` flag. Supply `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` through the environment for the selected providers. Use a workspace-scoped Anthropic Console API key. [OpenAI Standard pricing](https://developers.openai.com/api/docs/pricing) · [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) · [Anthropic API authentication](https://platform.claude.com/docs/en/api/overview#authentication)

```sh
weightspace reading plan --config funded-reading.json --panel benchmark-panel.json --output benchmark-plan.json
weightspace reading run --config funded-reading.json --panel benchmark-panel.json \
  --output runs/cnn-reading-001 --allow-api --confirm-credits
weightspace reading summarize --input runs/cnn-reading-001 --output cnn-summary.json
```

The input estimate rounds up prompt UTF-8 bytes times the configured input-tokens-per-byte factor and adds the framing reserve. The rate card uses its stated long-context threshold and the highest listed input-category rate for the reservation. The output limit is sent to the provider and covers output usage including billed reasoning, where applicable. Reject context-window or funding projections before sending requests. Reservations stay consumed for attempted calls, including timeouts, malformed outputs, and unknown usage. The run stops after a reported token overrun, reported model or service drift, or an authentication, access, billing, or rate-limit error. A run has a single attempt per scheduled request.

The example rate cards cite Standard API rates checked on 19 November 2026. Refresh them and check context and model availability before funding a comparison. Input bounds and dollar reservations are planning estimates. Provider tokenization, additional pricing terms, and invoiced charges can differ. Check provider-side account controls for billing limits. The journal keeps actual charge unknown and reports list-price estimates from normalized usage when measured. It records each attempt before sending and appends the sanitized response body, digest, request ID, usage, and outcome. It records the received byte count and flags truncation or sanitization. Credentials and request authentication headers are excluded. Journals can include held-out labels and provider responses, so review them before sharing.

## Scoring and the statistics baseline

Scores summarize mean predictions from complete repeats per network using Kendall's tau-b and mean absolute error. Tau-b handles ties and is undefined for a constant ranking or fewer than two networks. Report coverage alongside scores because failed requests can change which networks are available. The summary reports all activations together and by activation.

Fix the agreement tolerance before running. Two predictions agree if their absolute difference is no greater than that tolerance. For each network, compute repeat agreement over distinct pairs of original requests and equivalent agreement over original-by-transformed pairs. The symmetry gap is repeat agreement minus equivalent agreement in complete matched panels. Each network has equal weight. The pooled gap averages across the three equivalent variants within each network. Bootstrap percentile intervals resample whole source training networks. The summary omits intervals for the fixture panel and for groups with fewer than 10 complete paired networks. Intervals describe the sampled panel under the independent-network assumption.

The shuffle control has its own prediction-change diagnostics. Its true accuracy is unknown in this protocol. Tau-b and accuracy-error calculations use original and function-preserving conditions only.

```sh
python -m pip install '.[reading-baseline]'
weightspace reading baseline --panel benchmark-panel.json --output cnn-baseline.json
```

This statistics baseline uses per-tensor means, population standard deviations, and quantiles for weights and biases, then fits a fixed gradient-boosted regressor. It fits on explicit training rows and scores the original test checkpoints with Kendall's tau-b and mean absolute error. The feature and model choices are inspired by StatNN. This baseline provides a new result specific to the declared panel. [Small CNN Zoo paper](https://arxiv.org/abs/2002.11448)
