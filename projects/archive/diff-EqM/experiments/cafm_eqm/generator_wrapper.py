"""EqM generator wrapper for CAFM-style adversarial post-training.

Provides a CAFM-compatible generator interface (`forward(x, y, t)`) over our
trusted EqM-B/2 80ep checkpoint. Internally fixes t=0 (EqM convention: time
conditioning removed; γ is sampled but not passed as a model input).

Design rationale (see `documentation/cafm-eqm-port-design.md`):
- CAFM's discriminator JVP machinery expects gen(x_t, y, t) → velocity prediction.
- EqM's network predicts a gradient field, NOT a velocity. The two differ by a
  factor of c(γ) (see eqm_target.py). We adapt by scaling the GENERATOR OUTPUT
  by c(γ) so the discriminator sees a velocity-shaped quantity comparable to the
  CAFM `velocity_real = ε − x` (after we also multiply that by c(γ) — see
  eqm_target.eqm_target).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .eqm_target import c_gamma


class EqMGeneratorWrapper(nn.Module):
    """Wrap an EqM model to expose a CAFM-compatible (x, y, t)→velocity interface.

    The wrapped EqM model is expected to accept (x, t_dummy, y=class_label) where
    t_dummy is ignored (or set to zero by the EqM training code itself).

    Forward returns either:
    - gradient-field output (model_output, unchanged) when ``mode='gradient'``.
    - velocity-shaped output (model_output / c(γ)) when ``mode='velocity'``.

    Default mode is ``'gradient'`` because the cleanest setup is:
    - Generator outputs the EqM gradient field directly (no scaling).
    - We multiply the CAFM 'velocity_real' by c(γ) on the real side so both sides
      of the discriminator's JVP receive the same target geometry.
    """

    SUPPORTED_MODES = ("gradient", "velocity")

    def __init__(self, eqm_model: nn.Module, *, mode: str = "gradient"):
        super().__init__()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"mode must be in {self.SUPPORTED_MODES}, got {mode!r}")
        self.eqm = eqm_model
        self.mode = mode

    @property
    def trainable(self) -> bool:
        return any(p.requires_grad for p in self.eqm.parameters())

    def forward(self, x: Tensor, y: Tensor, gamma: Tensor) -> Tensor:
        """Predict EqM gradient (or velocity-rescaled) at (x, γ).

        Args:
            x: noised latents [B, C, H, W]
            y: class labels [B]
            gamma: EqM mixing γ ∈ [0,1] [B]

        Returns:
            [B, C, H, W] tensor. Mode controls scaling:
              gradient  → raw EqM model output (default; recommended).
              velocity  → model_output / max(c(γ), eps_floor) to convert to
                          velocity-shaped quantity matching CAFM's `noises − latents`.
        """
        # EqM convention: model receives t=0 (time conditioning removed).
        t_dummy = torch.zeros_like(gamma)
        out = self.eqm(x, t_dummy, y=y)

        if self.mode == "gradient":
            return out
        # velocity mode: rescale by 1/c(γ); floor prevents blow-up near γ=1.
        c = c_gamma(gamma).clamp_min(0.1).view(-1, 1, 1, 1)
        return out / c


__all__ = ["EqMGeneratorWrapper"]
