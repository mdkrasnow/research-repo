"""v10_hard_example_eqm — Madry-style PGD hard-example mining on EqM base loss.

Per documentation/v10_hard_example_eqm_proposal.md and updated mechanism analysis
in documentation/literature/SYNTHESIS.md §3.

Loss:
    L_total = L_base(x_t) + λ · L_base(x_t + δ*, target)
    δ* = argmax_{||δ||₂ ≤ ε}  ||f(x_t + δ) − target||²

Where target is the CLEAN (un-perturbed) EqM regression target (ε − x) · c(γ).
This is invariance-flavored adversarial regression: the model must satisfy the
SAME target despite perturbed input.

Defaults (post-lit-review revision per SYNTHESIS §3):
- λ = 0.1 (small, per Briglia 2025 stable regime).
- K = 1 (FGSM; Briglia shows single-step sufficient + 3× cheaper than K=3).
- ε = 0.3 (CIFAR-validated regime from v02).
- mining_lr = 0.05.
- mine_every = 1 (every gen step).

Diagnostics expose L_hard / L_base ratio + delta-norm for CIFAR-sanity collapse
detection (Briglia threat assessment per SYNTHESIS §3.1).
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


def _mine_hard(model, x_t, t_model, target, K: int, eps_radius: float, lr: float):
    delta = torch.zeros_like(x_t).normal_(0.0, eps_radius / 2.0)
    delta = _project_l2(delta, eps_radius)
    for _ in range(K):
        delta = delta.detach().requires_grad_(True)
        pred = model(x_t + delta, t_model)
        loss_adv = ((pred - target) ** 2).mean()
        g = torch.autograd.grad(loss_adv, delta)[0]
        with torch.no_grad():
            delta = delta + lr * g.sign()
            delta = _project_l2(delta, eps_radius)
    return delta.detach()


def step_fn(model, x1, step, device, args: TrainArgs):
    """One training step: base EqM loss + (optional) v10 PGA hard-example loss."""
    e = args.extras or {}

    B = x1.size(0)
    eps_train = args.train_eps if args.train_eps is not None else 1e-3
    a = args.a if args.a is not None else 0.8
    gain = args.gain if args.gain is not None else 4.0

    # Sample γ and noise → EqM forward path.
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device) * (1.0 - 2.0 * eps_train) + eps_train
    t_ = t.view(B, 1, 1, 1)
    x_t = (1.0 - t_) * x0 + t_ * x1
    ut = x1 - x0
    ct = eqm_ct(t, a=a, gain=gain).view(B, 1, 1, 1)
    target = ct * ut
    t_model = (t * 999.0).clamp_min(0.0)

    # Clean prediction + base loss.
    pred_clean = model(x_t, t_model)
    loss_base = F.mse_loss(pred_clean, target)

    lam = e.get("lambda_v10", 0.1)
    K = e.get("mining_K", 1)
    eps_radius = e.get("eps_radius", 0.3)
    lr = e.get("mining_lr", 0.05)
    mine_every = e.get("mine_every", 1)

    loss_hard_val = 0.0
    delta_norm_val = 0.0
    if lam > 0 and mine_every > 0 and (step % mine_every == 0):
        delta = _mine_hard(model, x_t, t_model, target, K=K, eps_radius=eps_radius, lr=lr)
        pred_hard = model(x_t + delta, t_model)
        loss_hard = F.mse_loss(pred_hard, target)
        total = loss_base + lam * loss_hard
        loss_hard_val = loss_hard.item()
        delta_norm_val = delta.flatten(1).norm(dim=1).mean().item()
    else:
        total = loss_base

    return total, {
        "base": loss_base.item(),
        "hard": loss_hard_val,
        "ratio": loss_hard_val / max(loss_base.item(), 1e-8),
        "delta_norm": delta_norm_val,
    }


def train(args: TrainArgs) -> float:
    return train_loop(
        args, step_fn,
        diag_keys=["base", "hard", "ratio", "delta_norm"],
    )
