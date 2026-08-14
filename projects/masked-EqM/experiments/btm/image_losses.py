"""BTM training step for the ImageNet B/2 EqM stack.

Convention reconciliation with the repository (checked, not assumed):

  transport/path.py ICPlan:  alpha_t = t multiplies x1 (DATA),
                             sigma_t = 1 - t multiplies x0 (NOISE),
                             so xt = (1-t) x0 + t x1 and ut = x1 - x0.
  BTM (this module):         I_t = alpha_t x0 + beta_t x1 with alpha_0 = 1,
                             so for the LINEAR interpolant alpha_t = 1 - t and
                             Idot_t = alpha_dot (x0 - x1) = x1 - x0.

  => the two conventions agree EXACTLY, with no sign flip.  EqM's legacy target
     `ut * transport.get_ct(t)` is the same vector scaled by c(t), which is
     precisely the paper's eq. (16) inconsistency.

  models.py EqM.forward with ebm='direct' returns  field = -grad_z E.
  We define the BTM potential phi = -E, so field = grad_z phi = b_theta.
  Following the field is therefore gradient DESCENT on the energy E.
  `test_btm_image_conventions.py` asserts all of the above numerically.

Loss normalization matches the repository's existing vector loss
`mean_flat((pred - target)**2)` = (1/d)|.|^2 (i.e. NO factor 1/2), so that
Arm V is numerically identical to a `--ebm none` run except for the target, and
existing `none` baselines stay comparable.  The FD arms are scaled to converge
to the same quantity: with E_u[(u^T e)^2] = |e|^2/d,

    L_D_image = E_{z,u} [ D_{h,u} phi(z) - u^T Idot ]^2   ->   (1/d) E|grad phi - Idot|^2.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import torch

from .fd import assert_no_double_backward, directional_fd, rademacher_directions
from .interpolant import EqMLinearTarget, SelfStoppingInterpolant, LinearInterpolant

BTM_MODES = (
    "btm_vector",
    "btm_scalar_exact",
    "btm_scalar_action_exact",
    "btm_scalar_fd_directional",
    "btm_scalar_fd_action",
)
BTM_FD_MODES = ("btm_scalar_fd_directional", "btm_scalar_fd_action")
BTM_SCALAR_MODES = BTM_MODES[1:]


@dataclass
class BTMConfig:
    mode: str = "btm_vector"
    interpolant: str = "self_stopping"   # self_stopping | linear | eqm_legacy
    tc: float = 0.8
    kappa: float = 0.8
    fd_eps: float = 1e-3
    fd_k: int = 1
    fd_direction: str = "rademacher"
    energy_difference_fp32: bool = True
    fd_chunk: Optional[int] = None


def build_image_interpolant(cfg: BTMConfig):
    if cfg.interpolant == "self_stopping":
        return SelfStoppingInterpolant(tc=cfg.tc)
    if cfg.interpolant == "linear":
        return LinearInterpolant()
    if cfg.interpolant == "eqm_legacy":
        return EqMLinearTarget(kappa=cfg.kappa)
    raise ValueError(f"unknown interpolant {cfg.interpolant!r}")


def _d(x):
    return x[0].numel()


@contextlib.contextmanager
def frozen_label_dropout(raw_model, y):
    """Apply classifier-free-guidance label dropout ONCE per training step.

    `LabelEmbedder.forward(labels, train)` resamples `token_drop` on EVERY
    forward while the module is in train mode, and it is the ONLY source of
    stochasticity in this architecture (timm Attention/Mlp are constructed with
    drop=0).  That is fatal for a finite difference: phi(z + hu) and phi(z - hu)
    would be evaluated under INDEPENDENT label-dropout masks, so the numerator
    phi(z+hu) - phi(z-hu) would be dominated by conditioning noise instead of
    the directional derivative -- a bias that grows without bound as h -> 0.

    This samples the drop mask once, substitutes the dropped labels, and puts
    the embedder (and only the embedder) in eval mode for the duration, so
    every evaluation inside one loss shares identical conditioning.  Applied to
    ALL BTM arms, not just the FD ones, so that the paired G-vs-D comparison
    (§27 shared randomness) is not confounded by conditioning noise either.
    """
    emb = raw_model.y_embedder
    saved_prob = emb.dropout_prob
    # NOTE: models.py calls `self.y_embedder(y, self.training)` -- it passes the
    # EqM MODULE's training flag, not the embedder's. So emb.eval() does NOT
    # disable token_drop; the labels would be dropped a second time, with fresh
    # randomness on every forward. Zeroing dropout_prob is what actually
    # disables it, because LabelEmbedder.forward gates on
    # `use_dropout = self.dropout_prob > 0`.
    if raw_model.training and saved_prob > 0:
        with torch.no_grad():
            y = emb.token_drop(y)
        emb.dropout_prob = 0.0
    try:
        yield y
    finally:
        emb.dropout_prob = saved_prob


def phi_closure(raw_model, t, y):
    """phi(z) = -E(z), batched over an arbitrary leading count.

    `t` and `y` are tiled to match z's batch, which is what lets the 2KB
    finite-difference evaluations go through the network in ONE forward call.
    """
    def phi(z):
        n = z.shape[0]
        rep = n // t.shape[0]
        tt = t.repeat(rep) if rep > 1 else t
        yy = y.repeat(rep) if rep > 1 else y
        return -raw_model(z, tt, yy, energy_only=True)
    return phi


def btm_sample(transport, interp, x1, generator=None):
    """Draw (t, x0, x1) with the repo's own sampler, then apply the interpolant."""
    t, x0, x1 = transport.sample(x1)
    z, zdot = interp.interpolate(t, x0, x1)
    return t, x0, x1, z, zdot


def btm_loss(model, raw_model, cfg: BTMConfig, transport, x1, y, generator=None):
    """One BTM training loss.  Returns (loss, stats).

    `model` is the DDP wrapper (used for the vector arm so DDP hooks fire);
    `raw_model` is the underlying module (used for the scalar-energy calls).
    """
    interp = build_image_interpolant(cfg)
    t, x0, x1, z, zdot = btm_sample(transport, interp, x1, generator)
    d = _d(z)
    stats = {}
    with frozen_label_dropout(raw_model, y) as y:
        return _btm_loss_inner(model, raw_model, cfg, t, x0, x1, z, zdot, y,
                               d, stats, generator)


def _btm_loss_inner(model, raw_model, cfg, t, x0, x1, z, zdot, y, d, stats,
                    generator):

    if cfg.mode == "btm_vector":
        pred = model(z, t, y)
        loss = ((pred - zdot) ** 2).reshape(z.shape[0], -1).sum(1).mean() / d
        return loss, stats

    if cfg.mode == "btm_scalar_exact":
        # field = -grad E = grad phi, produced with create_graph=True by the
        # model's own train=True path (the mixed-derivative path under test).
        field = model(z, t, y, train=True)
        loss = ((field - zdot) ** 2).reshape(z.shape[0], -1).sum(1).mean() / d
        return loss, stats

    if cfg.mode == "btm_scalar_action_exact":
        zz = z.detach().requires_grad_(True)
        field = model(zz, t, y, train=True)          # = grad phi
        kinetic = (field ** 2).reshape(z.shape[0], -1).sum(1).mean() / d
        E0 = raw_model(x0, t, y, energy_only=True)
        E1 = raw_model(x1, t, y, energy_only=True)
        # phi = -E, so (E_mu0 phi - E_mu1 phi) = (E1.mean() - E0.mean())
        endpoints = 2.0 * (E1.mean() - E0.mean()) / d
        loss = kinetic + endpoints
        stats["loss_kinetic"] = float(kinetic.detach())
        stats["loss_endpoint"] = float(endpoints.detach())
        return loss, stats

    if cfg.mode in BTM_FD_MODES:
        with assert_no_double_backward():
            phi = phi_closure(raw_model, t, y)
            u = rademacher_directions(z.shape, cfg.fd_k, device=z.device,
                                      dtype=z.dtype, generator=generator)
            D, h, gap = directional_fd(
                phi, z.detach(), u, cfg.fd_eps,
                fp32_subtract=cfg.energy_difference_fp32, chunk=cfg.fd_chunk)
            stats["fd_h_mean"] = float(h.mean())
            stats["fd_gap_abs"] = float(gap.detach().abs().mean())
            if cfg.mode == "btm_scalar_fd_directional":
                with torch.no_grad():
                    proj = torch.einsum("kb...,b...->kb", u, zdot).to(D.dtype)
                loss = ((D - proj) ** 2).mean()
                stats["fd_target_rms"] = float(proj.pow(2).mean().sqrt())
            else:
                kinetic = (D ** 2).mean()
                E0 = raw_model(x0, t, y, energy_only=True)
                E1 = raw_model(x1, t, y, energy_only=True)
                endpoints = 2.0 * (E1.mean() - E0.mean()) / d
                loss = kinetic + endpoints.to(kinetic.dtype)
                stats["loss_kinetic"] = float(kinetic.detach())
                stats["loss_endpoint"] = float(endpoints.detach())
            return loss, stats

    raise ValueError(f"unknown btm mode {cfg.mode!r}")


def btm_fd_backward_accumulate(model, raw_model, cfg: BTMConfig, transport,
                              x1, y, generator=None):
    """FD training step with PER-DIRECTION gradient accumulation.

    `directional_fd` stacks all 2*K*B evaluations into ONE forward so the
    network sees a single large batch.  That is the right call for speed at
    K=1, but at K=4 with 64 images/GPU it means 512 images of retained
    activations and it OOMs an 80 GB A100 (observed: 75 GiB allocated, job
    39078779).  Chunking the forward does NOT fix it, because autograd keeps
    every chunk's graph alive until backward.

    Because the loss is a MEAN over K independent directions,

        L = (1/K) sum_k (D_{h,u_k} phi(z) - u_k^T Idot)^2

    and the directions share no graph, backward-ing each term with weight 1/K
    and accumulating into .grad is EXACTLY equivalent to backward-ing the sum,
    while holding only 2*B activations at a time.  Effective batch, data, and
    the resulting gradient are unchanged -- so G-vs-D comparability at matched
    batch size is preserved, which is the whole point of the experiment.

    Returns (loss_value_detached, stats).  The caller must NOT call backward
    again.
    """
    interp = build_image_interpolant(cfg)
    t, x0, x1, z, zdot = btm_sample(transport, interp, x1, generator)
    d = _d(z)
    stats = {}
    total = 0.0
    with frozen_label_dropout(raw_model, y) as yy:
        phi = phi_closure(raw_model, t, yy)
        with assert_no_double_backward():
            u_all = rademacher_directions(z.shape, cfg.fd_k, device=z.device,
                                          dtype=z.dtype, generator=generator)
            gaps, hs = [], []
            for k in range(cfg.fd_k):
                u = u_all[k:k + 1]                      # [1, B, ...]
                D, h, gap = directional_fd(
                    phi, z.detach(), u, cfg.fd_eps,
                    fp32_subtract=cfg.energy_difference_fp32,
                    chunk=cfg.fd_chunk)
                if cfg.mode == "btm_scalar_fd_directional":
                    with torch.no_grad():
                        proj = torch.einsum("kb...,b...->kb", u, zdot).to(D.dtype)
                    loss_k = ((D - proj) ** 2).mean()
                else:
                    loss_k = (D ** 2).mean()
                (loss_k / cfg.fd_k).backward()
                total += float(loss_k.detach()) / cfg.fd_k
                gaps.append(float(gap.detach().abs().mean()))
                hs.append(float(h.mean()))
            if cfg.mode == "btm_scalar_fd_action":
                E0 = raw_model(x0, t, yy, energy_only=True)
                E1 = raw_model(x1, t, yy, energy_only=True)
                endpoints = 2.0 * (E1.mean() - E0.mean()) / d
                endpoints.backward()
                total += float(endpoints.detach())
                stats["loss_endpoint"] = float(endpoints.detach())
    stats["fd_h_mean"] = sum(hs) / len(hs)
    stats["fd_gap_abs"] = sum(gaps) / len(gaps)
    stats["fd_accumulated"] = True
    return total, stats


def btm_eval_target_match_vector(model, cfg: BTMConfig, transport, x1, y):
    """EVALUATION-ONLY Table-D row for the VECTOR arm.

    Without this the vector baseline has a training loss but no target-cosine,
    so the scalar arms' cosines have no measured ceiling to be read against --
    0.68 means something very different if V reaches 0.70 than if V reaches
    0.95.  Same probe construction and same reported quantities as the scalar
    version, but the field is the network output directly rather than grad phi.
    """
    interp = build_image_interpolant(cfg)
    t, x0, x1, z, zdot = btm_sample(transport, interp, x1)
    d = _d(z)
    with torch.no_grad(), frozen_label_dropout(model, y) as yy:
        field = model(z, t, yy)
        f = field.reshape(z.shape[0], -1)
        u = zdot.reshape(z.shape[0], -1)
        mse = ((f - u) ** 2).sum(1) / d
        cos = torch.nn.functional.cosine_similarity(f, u, dim=1)
        ratio = f.norm(dim=1) / (u.norm(dim=1) + 1e-12)
        near = t > cfg.tc
        out = {
            "target_mse_per_dim": float(mse.mean()),
            "target_cosine": float(cos.mean()),
            "target_norm_ratio": float(ratio.mean()),
            "field_norm": float(f.norm(dim=1).mean()),
        }
        if near.any():
            out["target_cosine_near_data"] = float(cos[near].mean())
            out["target_mse_near_data"] = float(mse[near].mean())
        if (~near).any():
            out["target_cosine_far"] = float(cos[~near].mean())
            out["target_mse_far"] = float(mse[~near].mean())
    return out


def btm_eval_target_match(raw_model, cfg: BTMConfig, transport, x1, y):
    """EVALUATION-ONLY: exact grad_x phi vs Idot on a held-out batch.

    Reports per-dimension MSE, cosine similarity, norm ratio and the same
    quantities split by t, near-data (t > tc) vs far (t <= tc).  Never called
    inside a training step; uses create_graph=False.
    """
    interp = build_image_interpolant(cfg)
    t, x0, x1, z, zdot = btm_sample(transport, interp, x1)
    d = _d(z)
    zz = z.detach().requires_grad_(True)
    with frozen_label_dropout(raw_model, y) as y:
        E = raw_model(zz, t, y, energy_only=True)
    g = torch.autograd.grad(E.sum(), zz, create_graph=False)[0].detach()
    field = -g                                   # = grad phi
    with torch.no_grad():
        f = field.reshape(z.shape[0], -1)
        u = zdot.reshape(z.shape[0], -1)
        mse = ((f - u) ** 2).sum(1) / d
        cos = torch.nn.functional.cosine_similarity(f, u, dim=1)
        ratio = f.norm(dim=1) / (u.norm(dim=1) + 1e-12)
        near = t > cfg.tc
        out = {
            "target_mse_per_dim": float(mse.mean()),
            "target_cosine": float(cos.mean()),
            "target_norm_ratio": float(ratio.mean()),
            "field_norm": float(f.norm(dim=1).mean()),
            "E_mean": float(E.mean()), "E_std": float(E.std()),
        }
        if near.any():
            out["target_cosine_near_data"] = float(cos[near].mean())
            out["target_mse_near_data"] = float(mse[near].mean())
        if (~near).any():
            out["target_cosine_far"] = float(cos[~near].mean())
            out["target_mse_far"] = float(mse[~near].mean())
    return out
