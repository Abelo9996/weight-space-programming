# Small CNN Zoo

The optional fixtures are two final-checkpoint weight rows from the Small CNN Zoo CIFAR-10-GS release by Thomas Unterthiner, Daniel Keysers, Sylvain Gelly, Olivier Bousquet, and Ilya Tolstikhin. The weights and layout are distributed by the authors under CC BY 4.0. The extractor copies the float32 values into separate NumPy files and preserves the original layout. It records the source generation, range and file hashes, zero-based row indices, and selected release metadata. [Paper](https://arxiv.org/abs/2002.11448) · [Authors' release](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

| Activation | Archive row | Training step | Archived test accuracy |
| --- | ---: | ---: | ---: |
| ReLU | 263357 | 86 | 0.5018 |
| tanh | 263330 | 86 | 0.4137 |

The extractor selects the first fully available final row per activation with archived test accuracy of at least 0.20. These rows are numerical fixtures selected to avoid a near-chance control. They use a nonrandom subset and reported accuracies from the release CSV.

## Fetch and test the fixtures

```sh
python -m pip install -e '.[dev]'
python scripts/fetch_cnn_zoo_fixture.py --output data/cnn_zoo_fixtures
WEIGHTSPACE_CNN_ZOO_FIXTURES=data/cnn_zoo_fixtures python -m pytest -ra tests/transforms
```

The script makes three bounded HTTP range requests for about 134 MB from a pinned object generation. Allow about 420 MB of temporary disk space under the output's parent directory. For an offline extraction, pass `--ranges-dir` pointing to the three saved ranges named `prefix.xz`, `block.xz`, and `suffix.xz`. The byte offsets and SHA-256 digests are in the script. The extractor reconstructs and decompresses the final XZ block, checks its CRC, and writes the fixtures with a manifest. The advertised archive MD5 identifies the source object. Verification covers the fetched ranges and exported files.

The loader uses the release's HWIO convolution and IO dense layout, three stride-2 valid convolutions, evaluation-mode dropout, global average pooling, and linear logits. The stride comes from the authors' code default. A NumPy forward using NHWC layout independently checks conversion into the PyTorch model. The tests use random inputs to compare logits. [Authors' training implementation](https://github.com/google-research/google-research/blob/master/dnn_predict_accuracy/train_network.py)

## Proposed reader comparison

The implemented [reading workflow](cnn-reading.md) supports manifest-based API evaluation, a train-only layer-statistics baseline, and offline planning. A larger benchmark still requires a defined train/test panel and funded runs.

Predict held-out test accuracy from serialized classifier weights. Sample eligible final checkpoints from the target split and group linked checkpoints by training run before splitting. Keep labels, checkpoint IDs, and any in-context examples for held-out checkpoints out of prompts. Check that the serialization precision preserves the symmetry to the stated numerical tolerance. Collect fresh predictions for original weights, equivalent weights, and a verified function-changing control.

Report Kendall's tau for accuracy prediction separately from representation sensitivity. For numeric predictions, compare the discrepancy between original and transformed predictions to discrepancy across fresh repeats of the same condition. Bootstrap at the network level and report ReLU and tanh subsets separately. Constant predictions can appear invariant, so report predictive accuracy alongside stability.

Compare an architecture-only baseline, weight summary statistics, and trained weight-space predictors on shared splits when feasible. Historical values for trained readers come from their published splits and protocols. [Small CNN Zoo](https://arxiv.org/abs/2002.11448) · [NFN](https://arxiv.org/abs/2302.14040) · [NFT](https://arxiv.org/abs/2305.13546) · [Neural Graphs](https://arxiv.org/abs/2403.12143)
