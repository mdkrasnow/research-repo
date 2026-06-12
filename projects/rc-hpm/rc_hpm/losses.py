"""Training losses for the E0.2' arm comparison (torch).

All mining/weights are TEACHER-driven and stop-gradiented (numpy -> torch
constants). Student gradients flow only through student logits s_hat.

Arms:
  plain InfoNCE            — no-mine baseline / negative-control floor
  naive hard-mine InfoNCE  — teacher top-k by similarity, NO certification
  RC-HPM L_out InfoNCE     — certified mined pos/neg + debiased ambiguous
  certified-random-k       — certification without hardness (F15 decomposition)
  RINCE                    — robust-loss baseline (Chuang et al. 2022)
  SupCon                   — full-label oracle ceiling (+ control)
"""
from __future__ import annotations

import numpy as np
import torch

from . import core


def _logits(z: torch.Tensor, tau: float) -> torch.Tensor:
    return (z @ z.T) / tau


def plain_infonce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5):
    """Standard one-direction InfoNCE: anchor=view1, pos=view2, negs=other view2."""
    n = z1.shape[0]
    sim = (z1 @ z2.T) / tau
    return torch.nn.functional.cross_entropy(sim, torch.arange(n, device=z1.device))


def naive_hardmine_infonce(z1: torch.Tensor, z2: torch.Tensor,
                           s_teacher: np.ndarray, k: int = 8, tau: float = 0.5):
    """Denominator = aug positive + top-k teacher-similar in-batch negatives.
    No certification — the K2 damage arm."""
    n = z1.shape[0]
    s = s_teacher.copy()
    np.fill_diagonal(s, -np.inf)
    topk = np.argsort(-s, axis=1)[:, :k]                       # (n, k)
    sim = (z1 @ z2.T) / tau
    pos = sim.diagonal()                                        # (n,)
    neg = sim.gather(1, torch.tensor(topk, device=z1.device))   # (n, k)
    logits = torch.cat([pos[:, None], neg], dim=1)
    return torch.nn.functional.cross_entropy(
        logits, torch.zeros(n, dtype=torch.long, device=z1.device))


def rc_hpm_loss(z1: torch.Tensor, z2: torch.Tensor, mined: core.MinedBatch,
                rho_hat, rho_plus, rho_amb, s_aug: np.ndarray,
                tau: float = 0.5):
    """L_out multi-positive InfoNCE per founding spec section 4.

    Positives P_i = mined certified positives + own aug view (true by
    construction). Denominator D_i = certified negatives (v-) + ambiguous
    (omega, debiased soft) over OTHER samples' view-2 embeddings.
    All weights teacher-side, stop-grad by construction (numpy constants).
    """
    dev = z1.device
    s, q = mined.s, mined.q
    v_neg = torch.tensor((1.0 - rho_hat(s)) * mined.neg_mask, dtype=torch.float32,
                         device=dev)
    omega = torch.tensor((1.0 - rho_amb(q)) * core.PINNED["c_amb"] * mined.amb_mask,
                         dtype=torch.float32, device=dev)
    v_pos = torch.tensor((1.0 - rho_plus(s)) * mined.pos_mask, dtype=torch.float32,
                         device=dev)
    v_aug = torch.tensor(1.0 - rho_plus(s_aug), dtype=torch.float32, device=dev)

    sim_cross = (z1 @ z2.T) / tau          # anchor i (view1) vs sample j (view2)
    exp_sim = torch.exp(sim_cross - sim_cross.max().detach())
    D = (v_neg * exp_sim).sum(1) + (omega * exp_sim).sum(1)    # (n,)

    # log p for each positive pair (i, p): log(exp(s_ip)/(exp(s_ip)+D_i)),
    # stable form; the max-shift cancels in the ratio
    log_frac = -torch.log1p(D[:, None] / (exp_sim + 1e-12))

    aug_diag = log_frac.diagonal()                              # aug positive
    pos_terms = (v_pos * log_frac).sum(1) + v_aug * aug_diag
    pos_count = v_pos.bool().sum(1).float() + 1.0               # + aug view
    return -(pos_terms / pos_count).mean()


def certified_random_k(z1, z2, q: np.ndarray, beta_plus: float, beta_minus: float,
                       rho_hat, rho_plus, rho_amb, s_teacher: np.ndarray,
                       s_aug: np.ndarray, k_plus: int, k_minus: int,
                       rng: np.random.Generator, tau: float = 0.5):
    """Certification WITHOUT hardness: random k among certified sets (F15)."""
    n = q.shape[0]
    s = s_teacher.copy()
    np.fill_diagonal(s, -np.inf)
    cert_pos = (q >= beta_plus) & np.isfinite(s)
    cert_neg = (q <= beta_minus) & np.isfinite(s)
    amb = (~cert_pos) & (~cert_neg) & np.isfinite(s)
    neg_mask = np.zeros((n, n), bool)
    pos_mask = np.zeros((n, n), bool)
    for i in range(n):
        cn = np.where(cert_neg[i])[0]
        if cn.size:
            neg_mask[i, rng.choice(cn, min(k_minus, cn.size), replace=False)] = True
        cp = np.where(cert_pos[i])[0]
        if cp.size:
            pos_mask[i, rng.choice(cp, min(k_plus, cp.size), replace=False)] = True
    s_safe = np.where(np.isfinite(s), s, 0.0)
    mined = core.MinedBatch(s_safe, q, neg_mask, pos_mask, amb, 0.0,
                            s_safe[neg_mask | pos_mask], 0.0)
    return rc_hpm_loss(z1, z2, mined, rho_hat, rho_plus, rho_amb, s_aug, tau)


def rince(z1: torch.Tensor, z2: torch.Tensor, q_exp: float = 0.5,
          lam: float = 0.025, tau: float = 0.5):
    """Robust InfoNCE (Chuang et al. 2022): -e^{q s+}/q + (lam (e^{s+} + sum e^{s-}))^q / q."""
    n = z1.shape[0]
    sim = (z1 @ z2.T) / tau
    pos = torch.exp(sim.diagonal())
    mask = ~torch.eye(n, dtype=torch.bool, device=z1.device)
    neg_sum = (torch.exp(sim) * mask).sum(1)
    loss = -(pos ** q_exp) / q_exp + ((lam * (pos + neg_sum)) ** q_exp) / q_exp
    return loss.mean()


def supcon(z1: torch.Tensor, z2: torch.Tensor, labels: np.ndarray,
           tau: float = 0.5):
    """Supervised contrastive (L_out), oracle ceiling arm."""
    dev = z1.device
    n = z1.shape[0]
    lab = torch.tensor(labels, device=dev)
    z = torch.cat([z1, z2], 0)
    labs = torch.cat([lab, lab], 0)
    sim = (z @ z.T) / tau
    eye = torch.eye(2 * n, dtype=torch.bool, device=dev)
    sim = sim.masked_fill(eye, -1e9)
    pos_mask = (labs[:, None] == labs[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos_mask.sum(1).clamp_min(1)
    return -((log_prob * pos_mask).sum(1) / pos_cnt).mean()
