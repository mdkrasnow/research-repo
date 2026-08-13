"""The five BTM training arms (plus the legacy-EqM negative control).

All losses are normalized PER DIMENSION so that they are directly comparable
across arms and across d:

  V  vector BTM              L_V = (1/2d) E |b_theta(I_t) - Idot_t|^2
  G  exact scalar grad-match L_G = (1/2d) E |grad phi(I_t) - Idot_t|^2
  A  exact Action/Ritz       J_A = (1/2d) E_nu |grad phi|^2
                                   + ( E_mu0 phi(x0) - E_mu1 phi(x1) ) / d
  D  directional FD          L_D = (1/2) E_{z,u} [ D_{h,u} phi(z) - u^T Idot ]^2
  F  FD Action/Ritz          J_F = (1/2) E_{z,u} [ D_{h,u} phi(z) ]^2
                                   + ( E_mu0 phi(x0) - E_mu1 phi(x1) ) / d

Population identities (verified numerically in tests/test_btm_math.py):

  * L_G = J_A + const(phi), because
        E[ grad phi(I_t)^T Idot_t ] = E_mu1 phi - E_mu0 phi
    (chain rule d/dt phi(I_t) = grad phi . Idot, integrated over t in [0,1]).
  * E_u[(u^T e)^2] = |e|^2/d for normalized Rademacher u, hence
        L_D -> L_G  and  J_F -> J_A   as h -> 0.

G and A require the mixed derivative d_theta d_x phi during optimization;
D and F do not, and are run under `assert_no_double_backward()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .fd import (
    assert_no_double_backward,
    directional_fd,
    exact_gradient,
    rademacher_directions,
)

ARMS = (
    "btm_vector",              # V
    "btm_scalar_exact",        # G
    "btm_scalar_action_exact", # A
    "btm_scalar_fd_directional",  # D
    "btm_scalar_fd_action",    # F
    "eqm_legacy_vector",       # negative control (old EqM target, vector net)
    "eqm_legacy_scalar",       # negative control (old EqM target, scalar net)
)

FD_ARMS = ("btm_scalar_fd_directional", "btm_scalar_fd_action")
SCALAR_ARMS = ARMS[1:5] + ("eqm_legacy_scalar",)
EXACT_SCALAR_ARMS = ("btm_scalar_exact", "btm_scalar_action_exact",
                     "eqm_legacy_scalar")


def _flat_dim(x: torch.Tensor) -> int:
    d = 1
    for s in x.shape[1:]:
        d *= int(s)
    return d


def _sqdist_per_dim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = _flat_dim(a)
    return ((a - b) ** 2).reshape(a.shape[0], -1).sum(1) / d


@dataclass
class FDConfig:
    eps_fd: float = 1e-3
    K: int = 1
    fp32_subtract: bool = True
    chunk: Optional[int] = None


def loss_vector(model, z, zdot):
    """Arm V (and the legacy-EqM vector control, which differs only in zdot)."""
    pred = model(z)
    return 0.5 * _sqdist_per_dim(pred, zdot).mean()


def loss_scalar_exact(phi_fn: Callable, z, zdot):
    """Arm G: exact gradient matching.  Requires create_graph=True."""
    z = z.detach().requires_grad_(True)
    g = exact_gradient(phi_fn, z, create_graph=True)
    return 0.5 * _sqdist_per_dim(g, zdot).mean()


def loss_action_exact(phi_fn: Callable, z, x0, x1):
    """Arm A: exact Action/Ritz.  Requires create_graph=True."""
    d = _flat_dim(z)
    z = z.detach().requires_grad_(True)
    g = exact_gradient(phi_fn, z, create_graph=True)
    kinetic = 0.5 * (g ** 2).reshape(g.shape[0], -1).sum(1).mean() / d
    endpoints = (phi_fn(x0).mean() - phi_fn(x1).mean()) / d
    return kinetic + endpoints


def loss_fd_directional(phi_fn: Callable, z, zdot, cfg: FDConfig,
                        generator=None):
    """Arm D: directional finite-difference BTM.  No input-gradient graph."""
    u = rademacher_directions(z.shape, cfg.K, device=z.device, dtype=z.dtype,
                              generator=generator)
    D, _, _ = directional_fd(phi_fn, z, u, cfg.eps_fd,
                             fp32_subtract=cfg.fp32_subtract, chunk=cfg.chunk)
    with torch.no_grad():
        proj = torch.einsum("kb...,b...->kb", u, zdot).to(D.dtype)
    return 0.5 * ((D - proj) ** 2).mean()


def loss_fd_action(phi_fn: Callable, z, x0, x1, cfg: FDConfig, generator=None):
    """Arm F: finite-difference Action/Ritz.  No input-gradient graph."""
    d = _flat_dim(z)
    u = rademacher_directions(z.shape, cfg.K, device=z.device, dtype=z.dtype,
                              generator=generator)
    D, _, _ = directional_fd(phi_fn, z, u, cfg.eps_fd,
                             fp32_subtract=cfg.fp32_subtract, chunk=cfg.chunk)
    kinetic = 0.5 * (D ** 2).mean()
    endpoints = (phi_fn(x0).mean() - phi_fn(x1).mean()) / d
    return kinetic + endpoints.to(kinetic.dtype)


def compute_loss(arm: str, model, batch, cfg: FDConfig, generator=None):
    """Dispatch.  `batch` is a dict with z, zdot, x0, x1.

    The scalar network is called as model(x) -> [B] (see `models.ScalarMLP`).
    FD arms are executed inside the double-backward guard.
    """
    z, zdot, x0, x1 = batch["z"], batch["zdot"], batch["x0"], batch["x1"]

    if arm in ("btm_vector", "eqm_legacy_vector"):
        return loss_vector(model, z, zdot)
    if arm in ("btm_scalar_exact", "eqm_legacy_scalar"):
        return loss_scalar_exact(model, z, zdot)
    if arm == "btm_scalar_action_exact":
        return loss_action_exact(model, z, x0, x1)
    if arm == "btm_scalar_fd_directional":
        with assert_no_double_backward():
            return loss_fd_directional(model, z, zdot, cfg, generator)
    if arm == "btm_scalar_fd_action":
        with assert_no_double_backward():
            return loss_fd_action(model, z, x0, x1, cfg, generator)
    raise ValueError(f"unknown arm {arm!r}")
