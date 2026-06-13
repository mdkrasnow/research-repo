"""D3 certified-selection variants (no hardness) + curriculum train loop.

All selection is teacher-driven, theta-free, stop-gradiented. Reuses
rc_hpm.core (gate/calibration/risk) and rc_hpm.ladder (generator/student/
probe). Selectors differ ONLY in WHICH certified negatives enter the per-anchor
top-k-; weights v- = (1-rho_hat(s)) and the loss form are unchanged from RC.
Positives = aug view only (D2 B3: positives channel not load-bearing).
"""
from __future__ import annotations

import numpy as np
import torch

from . import core
from .ladder import (Rung, Student, draw, teacher_embed, linear_probe,
                     calibrate_rung)
from .toy import aug_view

TAU = 0.5
N_BATCH = 64
K_MINUS = 8
STEPS = 1500


# ----------------------------------------------------------------------------
# Certified-negative selectors -> per-anchor boolean mask (n, n)
# All operate ONLY on the certified-negative set (q <= beta_minus).
# ----------------------------------------------------------------------------

def _certified_neg(emb, q, beta_minus):
    s = emb @ emb.T
    np.fill_diagonal(s, -np.inf)
    cert = (q <= beta_minus) & np.isfinite(s)
    return s, cert


def _topk_by(score, cert, k):
    n = cert.shape[0]
    u = np.where(cert, score, -np.inf)
    mask = np.zeros((n, n), bool)
    kk = min(k, n)
    idx = np.argpartition(-u, kk - 1, axis=1)[:, :kk]
    rows = np.repeat(np.arange(n), kk)
    mask[rows, idx.ravel()] = True
    return mask & cert


def sel_random_k(emb, q, beta_minus, rho_hat, k, rng):
    s, cert = _certified_neg(emb, q, beta_minus)
    return _topk_by(rng.random(cert.shape), cert, k), s


def sel_conf_easy(emb, q, beta_minus, rho_hat, k, rng):
    """Most-confidently-clean: highest (1 - rho_hat(s)) = lowest same-class p."""
    s, cert = _certified_neg(emb, q, beta_minus)
    return _topk_by(1.0 - rho_hat(np.where(np.isfinite(s), s, 0.0)), cert, k), s


def sel_mid_band(emb, q, beta_minus, rho_hat, k, rng, band=None):
    """Middle similarity tercile of the certified set (drop top+bottom).
    band overrides for the curriculum: {'low','mid','lowmid'}."""
    s, cert = _certified_neg(emb, q, beta_minus)
    cs = s[cert]
    if cs.size < 3:
        return _topk_by(rng.random(cert.shape), cert, k), s
    lo, hi = np.quantile(cs, [1 / 3, 2 / 3])
    if band == "low":
        keep = (s <= lo)
    elif band == "lowmid":
        keep = (s <= hi)
    else:                                   # mid (default)
        keep = (s >= lo) & (s <= hi)
    band_cert = cert & keep
    # diverse within band (farthest-point), fallback random if empty
    return _diverse_topk(emb, band_cert, cert, k, rng), s


def sel_diverse(emb, q, beta_minus, rho_hat, k, rng):
    s, cert = _certified_neg(emb, q, beta_minus)
    return _diverse_topk(emb, cert, cert, k, rng), s


def _diverse_topk(emb, pool, fallback, k, rng):
    """Greedy farthest-point selection per anchor among `pool`; if an anchor's
    pool < k, fill from fallback randomly."""
    n = emb.shape[0]
    mask = np.zeros((n, n), bool)
    for i in range(n):
        cand = np.where(pool[i])[0]
        if cand.size == 0:
            cand = np.where(fallback[i])[0]
            if cand.size == 0:
                continue
        if cand.size <= k:
            mask[i, cand] = True
            continue
        # farthest-point: start random, add the candidate maximizing min dist
        chosen = [int(rng.choice(cand))]
        sims = emb[cand] @ emb[cand].T
        cand_idx = {c: p for p, c in enumerate(cand)}
        while len(chosen) < k:
            ch_pos = [cand_idx[c] for c in chosen]
            mind = sims[:, ch_pos].max(1)        # high sim = close
            mind[ch_pos] = np.inf
            nxt = cand[int(np.argmin(mind))]
            chosen.append(int(nxt))
        mask[i, chosen] = True
    return mask & fallback


SELECTORS = dict(cert_random_k=sel_random_k, cert_conf_easy=sel_conf_easy,
                 cert_mid_band=sel_mid_band, cert_diverse=sel_diverse)


# ----------------------------------------------------------------------------
# Loss (negatives-only debiased; positives = aug view) reusing core weights
# ----------------------------------------------------------------------------

def _neg_loss(z1, z2, s, q, neg_mask, amb_mask, rho_hat, rho_amb, tau=TAU):
    n = z1.shape[0]
    v_neg = torch.tensor((1.0 - rho_hat(s)) * neg_mask, dtype=torch.float32)
    omega = torch.tensor((1.0 - rho_amb(q)) * core.PINNED["c_amb"] * amb_mask,
                         dtype=torch.float32)
    sim = (z1 @ z2.T) / tau
    exps = torch.exp(sim - sim.max().detach())
    D = (v_neg * exps).sum(1) + (omega * exps).sum(1)
    log_frac = -torch.log1p(D / (exps.diagonal() + 1e-12))
    return -log_frac.mean()


# ----------------------------------------------------------------------------
# Train one D3 arm
# ----------------------------------------------------------------------------

def train_d3_arm(rung: Rung, arm: str, seed: int, q_fn, calib,
                 alpha: float = 0.40, curriculum_C: int = 150):
    torch.manual_seed(seed * 37 + 5)
    rng = np.random.default_rng(seed + 4000)
    if calib is None or calib.aborted:
        return dict(arm=arm, seed=seed, probe_acc=None, aborted=True,
                    rung=rung.tag())
    bm, bp = calib.lam[1], calib.lam[0]
    student = Student(d_in=rung.dim)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)

    band = "low"                              # curriculum start
    risk_window, traj, abst = [], [], []
    for step in range(STEPS):
        xb, lb = draw(rung, N_BATCH, rng)
        x1 = aug_view(xb, rng, 0.25)
        x2 = aug_view(xb, rng, 0.25)
        z1 = student(torch.tensor(x1, dtype=torch.float32))
        z2 = student(torch.tensor(x2, dtype=torch.float32))
        eb = teacher_embed(rung, xb)
        qb = core.q_matrix(q_fn, eb, lb)
        s_full = eb @ eb.T
        np.fill_diagonal(s_full, -np.inf)
        amb = (qb < bp) & (qb > bm) & np.isfinite(s_full)

        if arm == "cert_curriculum":
            neg, s = sel_mid_band(eb, qb, bm, calib.rho_hat, K_MINUS, rng,
                                  band=band)
        else:
            neg, s = SELECTORS[arm](eb, qb, bm, calib.rho_hat, K_MINUS, rng)
        s_safe = np.where(np.isfinite(s), s, 0.0)

        loss = _neg_loss(z1, z2, s_safe, qb, neg, amb, calib.rho_hat,
                         calib.rho_amb)
        opt.zero_grad(); loss.backward(); opt.step()

        # realized batch risk (teacher labels; synthetic) for curriculum + log
        ys = (lb[:, None] == lb[None, :]).astype(float)
        w = core.w_repulsive(s_safe)
        vneg = 1.0 - calib.rho_hat(s_safe)
        num = (vneg * w * ys)[neg].sum()
        den = (vneg * w)[neg].sum()
        risk = float(num / den) if den > 0 else 0.0
        risk_window.append(risk)
        abst.append(float(amb.sum()) / max(N_BATCH * N_BATCH - N_BATCH, 1))

        if arm == "cert_curriculum" and (step + 1) % curriculum_C == 0:
            wr = float(np.mean(risk_window[-curriculum_C:]))
            order = ["low", "lowmid", "mid"]
            i = order.index(band)
            if wr <= alpha and i < len(order) - 1:
                band = order[i + 1]
            elif wr > alpha and i > 0:
                band = order[i - 1]
            traj.append((step + 1, band, round(wr, 3)))

    acc = linear_probe(student, rung, rng)
    return dict(arm=arm, seed=seed, probe_acc=acc, aborted=False,
                rung=rung.tag(), realized_risk=float(np.mean(risk_window)),
                realized_risk_max=float(np.max(risk_window)),
                abstention=float(np.mean(abst)),
                trajectory=traj if arm == "cert_curriculum" else None)
