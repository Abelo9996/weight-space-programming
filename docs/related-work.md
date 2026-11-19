# Related work and attribution

Takehi et al. describe language models examining and editing serialized network weights in *Sparks of Weight-Space Programming*. WeightBench supplies seeded repair tasks and evaluation protocols. This package uses a subset of its classic-control tasks to pair original and equivalent starting parameterizations and compare editing methods under common edit and budget rules. [Sparks](https://rikiyat.github.io/blog/weight-space-programming/) · [WeightBench paper](https://arxiv.org/abs/2611.18553)

- Cha et al. use repeated reads of functionally equivalent neural-field weights to test sensitivity to their representation. The proposed CNN Zoo reader comparison takes its paired, repeat-aware design from that work. [Function-Preserving Audits of Visual Weight Readers](https://arxiv.org/abs/2611.05798)
- Tran et al. compare trained weight-space predictors under permutations and activation-specific rescaling on the Small CNN Zoo. Their protocols give a reference point for any comparison with general-purpose language-model readers. [Monomial Matrix Group Equivariant Neural Functional Networks](https://openreview.net/pdf?id=rQYyWGYuzK)
- Oksanen compares language-model repair with an evolution strategy and a gain grid under a shared evaluation count on a tractor controller. Gao et al. compare editing methods under a matched candidate budget on diffusion tasks. Kundu checks reading sensitivity to equivalent policy weights and studies repair separately. [Tractor guidance](https://arxiv.org/abs/2611.21958) · [Diffusion editing](https://arxiv.org/abs/2611.22228) · [Same Fix, Different Weights](https://arxiv.org/abs/2611.20953)
- Unterthiner et al. introduce the Small CNN Zoo and a weight-statistics baseline. NFN, NFT, and Neural Graphs provide trained predictor references for CNN Zoo accuracy ranking. [Small CNN Zoo](https://arxiv.org/abs/2002.11448) · [NFN](https://arxiv.org/abs/2302.14040) · [NFT](https://arxiv.org/abs/2305.13546) · [Neural Graphs](https://arxiv.org/abs/2403.12143)

## Software and data

The task code incorporates MIT-licensed components from WeightBench at commit `2acd60172fc9758eb9030b1cab4a2ff29be5c993`. The classic-control dynamics also retain Gymnasium's MIT attribution. See the notices packaged with [vendored code](../src/weightspace/_vendor/). [WeightBench source](https://github.com/Aayush-Marasini/WeightBench/tree/2acd60172fc9758eb9030b1cab4a2ff29be5c993) · [Gymnasium source](https://github.com/Farama-Foundation/Gymnasium)

Optional Small CNN Zoo fixture data are acquired and attributed separately under CC BY 4.0. [CNN Zoo fixture guide](cnn-zoo.md)
