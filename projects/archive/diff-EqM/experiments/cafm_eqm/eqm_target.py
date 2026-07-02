"""EqM target geometry: c(γ) truncated decay + target construction.

Matches `eqm-upstream/transport/transport.py:get_ct()` exactly. Hardcoded values
(interp=0.8, λ_multiplier=4.0) per Wang & Du 2025 §5.3 default config — this is
the config that produces FID 32.85 at B/2 80ep (our paper-comparable baseline).

This module is the single source of truth for c(γ) in the diff-EqM repo. Anything
that computes the EqM target should import from here to avoid drift.
"""
from __future__ import annotations

import torch
from torch import Tensor


# Hardcoded EqM defaults (Wang & Du 2025).
EQM_INTERP = 0.8
EQM_LAMBDA = 4.0


def c_gamma(
    gamma: Tensor,
    *,
    interp: float = EQM_INTERP,
    lam: float = EQM_LAMBDA,
) -> Tensor:
    """Truncated-decay c(γ) used in EqM training.

    c(γ) = λ · min(1 − (1−1)/interp · γ, 1/(1−interp) − 1/(1−interp) · γ)
         = λ · min(1, (1−γ)/(1−interp))

    With defaults λ=4, interp=0.8: c = 4 for γ ≤ 0.8, then linear decay to 0 at γ = 1.
    """
    start = 1.0
    # First branch is constant (start - (start-1)/interp · γ); with start=1 this = 1.
    branch_a = start - (start - 1.0) / interp * gamma
    branch_b = 1.0 / (1.0 - interp) - 1.0 / (1.0 - interp) * gamma
    return torch.minimum(branch_a, branch_b) * lam


def interpolate(latents: Tensor, noises: Tensor, gamma: Tensor) -> Tensor:
    """EqM forward path: x_γ = (1 − γ) · x + γ · ε.

    Note: this matches CAFM's convention `(1-t) * latents + t * noises` (x0=image,
    x1=noise). EqM paper uses the same convention internally (Wang & Du §3).
    """
    g = gamma.view(-1, 1, 1, 1)
    return (1.0 - g) * latents + g * noises


def velocity_raw(latents: Tensor, noises: Tensor) -> Tensor:
    """Raw velocity ε − x (CAFM-compatible)."""
    return noises - latents


def eqm_target(latents: Tensor, noises: Tensor, gamma: Tensor) -> Tensor:
    """EqM regression target = (ε − x) · c(γ)."""
    c = c_gamma(gamma).view(-1, 1, 1, 1)
    return (noises - latents) * c


def eqm_base_loss(model_output: Tensor, target: Tensor) -> Tensor:
    """Mean-squared regression on the EqM target."""
    return ((model_output - target) ** 2).mean()


__all__ = [
    "EQM_INTERP",
    "EQM_LAMBDA",
    "c_gamma",
    "interpolate",
    "velocity_raw",
    "eqm_target",
    "eqm_base_loss",
]
