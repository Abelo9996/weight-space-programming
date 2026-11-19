"""Behavior cloning adapted from the WeightBench classic-control builder.

Source: https://github.com/Aayush-Marasini/WeightBench/blob/2acd60172fc9758eb9030b1cab4a2ff29be5c993/weightbench/tracks/tr02_control_lib/mlp.py
Copyright (c) 2026 WeightBench contributors. See WEIGHTBENCH_LICENSE.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Params = dict[str, np.ndarray]


def init_params(sizes: list[int], rng: np.random.Generator) -> Params:
    """Orthogonal init (gain sqrt 2 hidden, 0.01 output), zero bias, like CleanRL/Kinetix PPO."""
    p: Params = {}
    for i in range(len(sizes) - 1):
        gain = np.sqrt(2.0) if i < len(sizes) - 2 else 0.01
        a = rng.normal(size=(max(sizes[i], sizes[i + 1]), min(sizes[i], sizes[i + 1])))
        q, r = np.linalg.qr(a)
        q = q * np.sign(np.diag(r))
        w = q if sizes[i + 1] >= sizes[i] else q.T
        p[f"fc{i}.weight"] = (gain * w[: sizes[i + 1], : sizes[i]]).astype(np.float32)
        p[f"fc{i}.bias"] = np.zeros(sizes[i + 1], np.float32)
    return p


def pruning_masks(params: Params, pruned: dict[int, list[int]]) -> Params:
    m = {k: np.ones_like(v) for k, v in params.items()}
    for layer, units in pruned.items():
        units = list(units)
        m[f"fc{layer}.weight"][units, :] = 0
        m[f"fc{layer}.bias"][units] = 0
        m[f"fc{layer + 1}.weight"][:, units] = 0
    return m


def train_bc(X: np.ndarray, Y: np.ndarray, sizes: list[int], loss: str, seed: int,
             pruned: dict[int, list[int]] | None = None, epochs: int = 30, lr: float = 1e-3,
             batch: int = 512, weight_decay: float = 0.0, threads: int = 1,
             log: Callable[[str], None] = print) -> Params:
    """Supervised imitation. loss='mse' (Y [N, A] targets) or 'ce' (Y [N] int labels).

    ``threads`` defaults to 1 and is not read from the environment: torch's CPU matmuls pick a different
    kernel and blocking with the thread count, so the trained weights would not be a function of (task, seed)
    alone when thread counts differ between hosts or jobs. A single thread fixes
    that setting.
    """
    import torch

    torch.set_num_threads(max(1, int(threads)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    p0 = init_params(sizes, rng)
    masks = pruning_masks(p0, pruned or {})
    tp = {k: torch.nn.Parameter(torch.tensor(v * masks[k])) for k, v in p0.items()}
    tm = {k: torch.tensor(v) for k, v in masks.items()}
    opt = torch.optim.AdamW(tp.values(), lr=lr, weight_decay=weight_decay)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.long if loss == "ce" else torch.float32)
    n = len(Xt)
    nl = len(sizes) - 1
    steps = epochs * max(1, n // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def fwd(x):
        h = x
        for i in range(nl):
            h = h @ tp[f"fc{i}.weight"].T + tp[f"fc{i}.bias"]
            if i < nl - 1:
                h = torch.relu(h)
        return h

    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n - batch + 1, batch):
            idx = perm[s:s + batch]
            out = fwd(Xt[idx])
            l = (torch.nn.functional.cross_entropy(out, Yt[idx]) if loss == "ce"
                 else torch.nn.functional.mse_loss(out, Yt[idx]))
            opt.zero_grad()
            l.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for k in tp:
                    tp[k].mul_(tm[k])
            tot += float(l.detach()) * len(idx)
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            log(f"  bc epoch {ep} loss {tot / max(1, n - n % batch):.5f}")
    return {k: v.detach().numpy().astype(np.float32) for k, v in tp.items()}
