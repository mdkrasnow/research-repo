"""Stochastic interpolants for Beckmann Transport Models (arXiv:2608.01692v2).

Conventions (authoritative, matching the paper):

    z = I_t = alpha_t * x0 + beta_t * x1,     beta_t = 1 - alpha_t
    Idot_t = alpha_dot_t * x0 + beta_dot_t * x1 = alpha_dot_t * (x0 - x1)

    alpha_0 = 1, alpha_1 = 0   (t=0 is the SOURCE mu0, t=1 is the TARGET mu1)

The corrected BTM regression target is b*(x) = E[Idot_t | I_t = x] and the
population drift satisfies the divergence equation

    div( nu * b ) = mu0 - mu1,      nu = int_0^1 mu_t dt

(paper, Theorem 1; verified verbatim against the v2 HTML).

Appendix H "self-stopping" interpolant (paper eq. 57), C^1 by construction:

    alpha_t = 1 - 2t/(1 + tc)              for 0 <= t <= tc
            = (1 - t)^2 / (1 - tc^2)       for tc <  t <= 1

so that alpha(tc) = (1-tc)/(1+tc) on both branches and
alpha_dot(tc) = -2/(1+tc) on both branches, and Idot_1 = 0 (the flow stops on
the support of mu1 -- no stopping criterion is needed at inference).

The legacy EqM target (paper eq. 16) is deliberately implemented separately in
`EqMLinearTarget`: it pairs the LINEAR interpolant with the target
c_t * (x1 - x0), which is NOT E[Idot | I_t] for any interpolant.  That
inconsistency is exactly what the campaign uses as its negative control.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _expand(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a [B] time tensor against an [B, ...] sample tensor."""
    return t.reshape(-1, *([1] * (x.ndim - 1)))


class Interpolant:
    """Base class: alpha_t / alpha_dot_t fully determine the interpolant."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.alpha(t)

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        return -self.alpha_dot(t)

    def interpolate(self, t, x0, x1):
        """Return (z, zdot) = (I_t, Idot_t)."""
        a = _expand(self.alpha(t), x0)
        ad = _expand(self.alpha_dot(t), x0)
        z = a * x0 + (1.0 - a) * x1
        zdot = ad * (x0 - x1)
        return z, zdot


@dataclass
class LinearInterpolant(Interpolant):
    """alpha_t = 1 - t.  Idot_t = x0 - x1 (constant, transverse crossing)."""

    def alpha(self, t):
        return 1.0 - t

    def alpha_dot(self, t):
        return -torch.ones_like(t)


@dataclass
class SelfStoppingInterpolant(Interpolant):
    """BTM Appendix-H piecewise interpolant (paper eq. 57), breakpoint tc."""

    tc: float = 0.8

    def __post_init__(self):
        if not 0.0 < self.tc < 1.0:
            raise ValueError(f"tc must lie in (0, 1), got {self.tc}")

    def alpha(self, t):
        tc = self.tc
        bulk = 1.0 - 2.0 * t / (1.0 + tc)
        tail = (1.0 - t) ** 2 / (1.0 - tc ** 2)
        return torch.where(t <= tc, bulk, tail)

    def alpha_dot(self, t):
        tc = self.tc
        bulk = torch.full_like(t, -2.0 / (1.0 + tc))
        tail = -2.0 * (1.0 - t) / (1.0 - tc ** 2)
        return torch.where(t <= tc, bulk, tail)


@dataclass
class EqMLinearTarget:
    """The ORIGINAL (inconsistent) EqM objective -- negative control only.

    Linear interpolant I_t = (1-t) x0 + t x1 paired with the schedule-scaled
    target c_t (x1 - x0), c_t = (1-t)^kappa.  The BTM five-atom figure uses
    kappa = 0.8, which is the value reproduced here.

    Note the SIGN: EqM's target points from x0 towards x1 (+(x1-x0)), whereas
    the BTM target Idot_t = alpha_dot (x0 - x1) = -(x0 - x1) = (x1 - x0) for
    the linear interpolant.  The two therefore agree exactly at c_t == 1, which
    is the paper's own statement of when eq. (16) coincides with eq. (5).
    """

    kappa: float = 0.8

    def c(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - t).clamp_min(0.0) ** self.kappa

    def interpolate(self, t, x0, x1):
        a = _expand(1.0 - t, x0)
        z = a * x0 + (1.0 - a) * x1
        target = _expand(self.c(t), x0) * (x1 - x0)
        return z, target


def build_interpolant(name: str, tc: float = 0.8, kappa: float = 0.8):
    if name == "self_stopping":
        return SelfStoppingInterpolant(tc=tc)
    if name == "linear":
        return LinearInterpolant()
    if name == "eqm_legacy":
        return EqMLinearTarget(kappa=kappa)
    raise ValueError(f"unknown interpolant {name!r}")
