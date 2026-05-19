"""v10 PGD hard-example mining on the EqM regression target.

Used standalone (v10-only training) and as an additive term in the v10+CAFM
combined trainer. Mining objective: find δ in L2 ball that maximizes the base
EqM regression loss.

    δ* = argmax_{||δ||₂ ≤ ε}  ||f(x_γ + δ, γ) − target(x, ε_noise, γ)||²

Per `documentation/v10-cafm-combination-design.md`:
- Default K=1 (FGSM-style; cheaper; Briglia 2025 shows single-step sufficient).
- L2-bounded perturbation (ε=0.3 default; CIFAR-validated regime).
- Detach δ before computing the auxiliary loss (standard adversarial training
  practice; gradients should NOT backprop through PGD inner loop).
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .eqm_target import eqm_base_loss


def project_l2_ball(delta: Tensor, eps_radius: float) -> Tensor:
    """Project delta back into the L2 ball of given radius (per-sample)."""
    b = delta.size(0)
    flat = delta.flatten(1).norm(dim=1, keepdim=True).view(b, 1, 1, 1)
    scale = torch.clamp(eps_radius / (flat + 1e-8), max=1.0)
    return delta * scale


@torch.no_grad()
def _init_delta(x_gamma: Tensor, eps_radius: float) -> Tensor:
    """Random initialization inside the L2 ball; std = eps_radius / 2."""
    delta = torch.zeros_like(x_gamma).normal_(0.0, eps_radius / 2.0)
    return project_l2_ball(delta, eps_radius)


def mine_hard_example(
    forward_fn: Callable[[Tensor], Tensor],
    x_gamma: Tensor,
    target: Tensor,
    *,
    K: int = 1,
    eps_radius: float = 0.3,
    lr: float = 0.05,
) -> Tensor:
    """Return adversarial perturbation δ* that maximizes ||f(x+δ) − target||².

    Args:
        forward_fn: function (x_perturbed) → model_output. Must be callable inside
            an autograd-enabled context. Wrap the generator + γ + class label
            outside this call (closure) so we can vary only δ.
        x_gamma: noised latents [B, C, H, W]
        target: regression target [B, C, H, W]
        K: number of PGD steps (default 1 = FGSM)
        eps_radius: L2 bound on ||δ||
        lr: PGA step size

    Returns:
        Detached δ tensor [B, C, H, W].
    """
    delta = _init_delta(x_gamma, eps_radius)
    for _ in range(K):
        delta = delta.detach().requires_grad_(True)
        pred = forward_fn(x_gamma + delta)
        loss_adv = eqm_base_loss(pred, target)
        grad = torch.autograd.grad(loss_adv, delta)[0]
        with torch.no_grad():
            delta = delta + lr * grad.sign()
            delta = project_l2_ball(delta, eps_radius)
    return delta.detach()


def v10_aux_loss(
    forward_fn: Callable[[Tensor], Tensor],
    x_gamma: Tensor,
    target: Tensor,
    *,
    K: int = 1,
    eps_radius: float = 0.3,
    lr: float = 0.05,
) -> tuple[Tensor, dict]:
    """Compute v10 auxiliary loss + diagnostics.

    Returns:
        L_hard: scalar Tensor for the auxiliary loss term.
        diag: dict with mean/std of ||δ||, ratio L_hard/L_base, etc.
    """
    delta = mine_hard_example(
        forward_fn, x_gamma, target, K=K, eps_radius=eps_radius, lr=lr
    )
    pred_hard = forward_fn(x_gamma + delta)
    l_hard = eqm_base_loss(pred_hard, target)

    delta_norms = delta.flatten(1).norm(dim=1)
    diag = {
        "v10/delta_norm_mean": delta_norms.mean().detach(),
        "v10/delta_norm_std": delta_norms.std().detach(),
        "v10/l_hard": l_hard.detach(),
    }
    return l_hard, diag


__all__ = ["project_l2_ball", "mine_hard_example", "v10_aux_loss"]
