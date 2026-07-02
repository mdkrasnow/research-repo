"""v11_hard_example_equivariant — equivariant-flavored hard-example mining.

PRIMARY FALLBACK if v10 (invariance-flavored) collapses per Briglia 2025
mechanism threat. See documentation/v11_fallback_sketch.md §v11.

Loss:
    L_total = L_base(x_γ, target) + λ · ||f(x_γ + δ*) − f(x_γ).detach()||²
    δ* = argmax_{||δ||₂ ≤ ε}  ||f(x_γ + δ) − f(x_γ).detach()||²

Key difference from v10:
- v10 (invariance):    L_aux = ||f(x+δ*) − target||²       — predict SAME target
- v11 (equivariant):   L_aux = ||f(x+δ*) − f(x).detach()||² — predict CLEAN output
- v11 is local Lipschitz / smoothness regularization, aligning with Briglia's
  "equivariance not invariance" prescription for diffusion AT.
- Mining objective is also changed: maximize OUTPUT DISCREPANCY, not regression
  error vs ground-truth target. This means PGA seeks places where f is locally
  non-smooth, which is a different failure mode than v10's "PGA finds places
  where regression fails."

When to activate:
- v10 CIFAR sanity collapses (NaN, FID >> baseline, base-loss runaway).
- v10 IN-1K Phase 2 gate: combined-FID worse than CAFM-only at any λ ∈ {0.03, 0.1, 0.3, 1.0}.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ._common import TrainArgs, eqm_ct, train_loop


def _project_l2(delta: torch.Tensor, eps_radius: float) -> torch.Tensor:
    b = delta.size(0)
    flat = delta.flatten(1).norm(dim=1, keepdim=True).view(b, 1, 1, 1)
    scale = torch.clamp(eps_radius / (flat + 1e-8), max=1.0)
    return delta * scale


def _mine_equivariant(model, x_t, t_model, pred_clean_detached, K, eps_radius, lr):
    """PGA to maximize ||f(x + δ) − f(x).detach()||²."""
    delta = torch.zeros_like(x_t).normal_(0.0, eps_radius / 2.0)
    delta = _project_l2(delta, eps_radius)
    for _ in range(K):
        delta = delta.detach().requires_grad_(True)
        pred_perturbed = model(x_t + delta, t_model)
        loss_adv = ((pred_perturbed - pred_clean_detached) ** 2).mean()
        g = torch.autograd.grad(loss_adv, delta)[0]
        with torch.no_grad():
            delta = delta + lr * g.sign()
            delta = _project_l2(delta, eps_radius)
    return delta.detach()


def step_fn(model, x1, step, device, args: TrainArgs):
    e = args.extras or {}

    B = x1.size(0)
    eps_train = args.train_eps if args.train_eps is not None else 1e-3
    a = args.a if args.a is not None else 0.8
    gain = args.gain if args.gain is not None else 4.0

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device) * (1.0 - 2.0 * eps_train) + eps_train
    t_ = t.view(B, 1, 1, 1)
    x_t = (1.0 - t_) * x0 + t_ * x1
    ut = x1 - x0
    ct = eqm_ct(t, a=a, gain=gain).view(B, 1, 1, 1)
    target = ct * ut
    t_model = (t * 999.0).clamp_min(0.0)

    pred_clean = model(x_t, t_model)
    loss_base = F.mse_loss(pred_clean, target)

    lam = e.get("lambda_v11", 0.1)
    K = e.get("mining_K", 1)
    eps_radius = e.get("eps_radius", 0.3)
    lr = e.get("mining_lr", 0.05)
    mine_every = e.get("mine_every", 1)

    loss_smooth_val = 0.0
    delta_norm_val = 0.0
    if lam > 0 and mine_every > 0 and (step % mine_every == 0):
        pred_clean_detached = pred_clean.detach()
        delta = _mine_equivariant(
            model, x_t, t_model, pred_clean_detached,
            K=K, eps_radius=eps_radius, lr=lr,
        )
        pred_perturbed = model(x_t + delta, t_model)
        loss_smooth = F.mse_loss(pred_perturbed, pred_clean_detached)
        total = loss_base + lam * loss_smooth
        loss_smooth_val = loss_smooth.item()
        delta_norm_val = delta.flatten(1).norm(dim=1).mean().item()
    else:
        total = loss_base

    return total, {
        "base": loss_base.item(),
        "smooth": loss_smooth_val,
        "ratio": loss_smooth_val / max(loss_base.item(), 1e-8),
        "delta_norm": delta_norm_val,
    }


def train(args: TrainArgs) -> float:
    return train_loop(
        args, step_fn,
        diag_keys=["base", "smooth", "ratio", "delta_norm"],
    )
