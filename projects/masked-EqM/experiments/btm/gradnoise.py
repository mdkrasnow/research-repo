"""Direct measurement of stochastic PARAMETER-gradient variance per arm (§19).

At FIXED parameters, draw many independent minibatches and compute the per-arm
stochastic parameter gradient g_i.  Report

    E|g|^2,  |E g|^2,  noise_scale = E|g - gbar|^2 / (|gbar|^2 + eps),
    mean pairwise cosine similarity,

which is what separates "Arm F's endpoint estimator is simply too noisy" from
"the mixed derivative is the problem".  Note |E g|^2 is estimated as
|gbar|^2 - E|g-gbar|^2/(n-1) is NOT applied: gbar's own bias is O(1/n) and n is
large here; the raw |gbar|^2 is reported alongside n so the reader can judge.
"""

from __future__ import annotations

import torch

from .objectives import FD_ARMS, FDConfig, compute_loss
from .fd import assert_no_double_backward


def _flat_grad(model):
    return torch.cat([
        (p.grad.detach().reshape(-1) if p.grad is not None
         else torch.zeros(p.numel(), device=p.device))
        for p in model.parameters()
    ])


def gradient_noise(model, batch_fn, arm: str, cfg: FDConfig, n_batches: int = 200,
                   generator=None, max_pairs: int = 2000):
    """batch_fn() -> dict(z, zdot, x0, x1); called n_batches times."""
    grads = []
    for _ in range(n_batches):
        batch = batch_fn()
        loss = compute_loss(arm, model, batch, cfg, generator=generator)
        model.zero_grad(set_to_none=True)
        if arm in FD_ARMS:
            with assert_no_double_backward():
                loss.backward()
        else:
            loss.backward()
        grads.append(_flat_grad(model).clone())
    model.zero_grad(set_to_none=True)

    G = torch.stack(grads).double()                     # [n, P]
    gbar = G.mean(0)
    dev = G - gbar
    E_sq = float((G ** 2).sum(1).mean())
    mean_sq = float((gbar ** 2).sum())
    noise = float((dev ** 2).sum(1).mean())
    Gn = G / (G.norm(dim=1, keepdim=True) + 1e-30)
    n = Gn.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    if idx.shape[1] > max_pairs:
        sel = torch.randperm(idx.shape[1])[:max_pairs]
        idx = idx[:, sel]
    cos = (Gn[idx[0]] * Gn[idx[1]]).sum(1)
    return {
        "arm": arm, "n_batches": n_batches,
        "E_norm_sq": E_sq,
        "mean_norm_sq": mean_sq,
        "noise_norm_sq": noise,
        "noise_scale": noise / (mean_sq + 1e-30),
        "mean_pairwise_cosine": float(cos.mean()),
        "median_pairwise_cosine": float(cos.median()),
        "snr": mean_sq / (noise + 1e-30),
    }
