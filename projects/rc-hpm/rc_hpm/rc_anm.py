"""RC-ANM — Risk-Controlled Adversarial Negative Mining for Equilibrium
Matching. EqM-native: certifies the MINED ENDPOINT (not a contrastive pair).

Reuses rc_hpm.core (LTT/HB) and rc_hpm.eqm2d (Field, pgd_mine, basin_certify,
get_ct, make_triplet, voronoi_basin, MEANS). Pinned per
documentation/preregistration-rc-anm.md: certified functional = r_basin
(analytic Voronoi labeler, eta=0 in 2D); r_field/r_target/r_inflate/r_return
are reported diagnostics + a soft accept/reject nuisance knob.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import core
from .eqm2d import (Field, MEANS, get_ct, gamma_bin, make_triplet,
                    pgd_mine, voronoi_basin)

GAMMA_BINS = [(0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0001)]
EPS_GRID = (0.1, 0.25, 0.5, 0.8, 1.2)
INFL_OK, INFL_MAX = 3.0, 20.0


# ----------------------------------------------------------------------------
# Teacher safety scores (frozen EMA teacher field). Each in [0,1], higher=worse.
# ----------------------------------------------------------------------------

def _residual(teacher: Field, x1, eps, t):
    tt = torch.tensor(t, dtype=torch.float32)[:, None]
    ct = get_ct(tt)
    xt = tt * torch.tensor(x1, dtype=torch.float32) + (1 - tt) * \
        torch.tensor(eps, dtype=torch.float32)
    target = (torch.tensor(x1, dtype=torch.float32) -
              torch.tensor(eps, dtype=torch.float32)) * ct
    with torch.no_grad():
        out = teacher(xt)
    return ((out - target) ** 2).sum(1).numpy(), xt.numpy()


def _descend_basin(teacher, xt, steps, eta):
    x = torch.tensor(xt, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(steps):
            x = x + eta * teacher(x)
    return voronoi_basin(x.numpy())


# v2: short-rollout proxy basin (low descent-noise) + analytic-oracle agreement
def proxy_basin(teacher, xt, steps=15, eta=0.1):
    return _descend_basin(teacher, xt, steps, eta)


def proxy_oracle_agreement(teacher, x1, lab, eps_adv, t, steps=15, eta=0.1):
    """Balanced accuracy of the teacher proxy's accept decision
    (proxy_basin(x_t_adv)==lab) vs the analytic oracle (voronoi_basin(
    x_t_adv)==lab). v2 validation gate uses this BEFORE calibration."""
    xt_adv = t[:, None] * x1 + (1 - t[:, None]) * eps_adv
    proxy_ok = proxy_basin(teacher, xt_adv, steps, eta) == lab
    oracle_ok = voronoi_basin(xt_adv) == lab
    tp = (proxy_ok & oracle_ok).sum()
    tn = (~proxy_ok & ~oracle_ok).sum()
    p = oracle_ok.sum()
    n = (~oracle_ok).sum()
    sens = tp / p if p else 0.0
    spec = tn / n if n else 0.0
    return dict(balanced_acc=float((sens + spec) / 2),
                agree=float((proxy_ok == oracle_ok).mean()),
                sens=float(sens), spec=float(spec))


def safety_scores(teacher: Field, x1, lab, eps_orig, eps_adv, t,
                  descend_steps=15, eta=0.1):
    """Returns dict of arrays (per sample): r_basin (certified functional) +
    diagnostics r_field, r_target, r_inflate, r_return.

    r_basin = FLIP risk (D2 lesson): mining-INDUCED basin error only —
    1 iff the mined endpoint flows to the wrong mode AND the un-mined endpoint
    flowed to the right one. Absolute basin rate has a teacher/geometry floor
    (~0.6: a pure-noise endpoint has no canonical basin); flip-risk isolates
    the damage mining adds and is ~0 for un-mined by construction.
    """
    tt = t[:, None]
    xt_adv = tt * x1 + (1 - tt) * eps_adv
    xt_orig = tt * x1 + (1 - tt) * eps_orig

    att_adv = _descend_basin(teacher, xt_adv, descend_steps, eta)
    att_org = _descend_basin(teacher, xt_orig, descend_steps, eta)
    r_basin = ((att_adv != lab) & (att_org == lab)).astype(float)

    # r_field: teacher-field disagreement induced by mining
    with torch.no_grad():
        f_adv = teacher(torch.tensor(xt_adv, dtype=torch.float32)).numpy()
        f_org = teacher(torch.tensor(xt_orig, dtype=torch.float32)).numpy()
    cos = (f_adv * f_org).sum(1) / (np.linalg.norm(f_adv, axis=1) *
                                    np.linalg.norm(f_org, axis=1) + 1e-12)
    r_field = np.clip((1 - cos) / 2, 0, 1)

    # r_target: mined-target direction corruption
    tg_adv = x1 - eps_adv
    tg_org = x1 - eps_orig
    cost = (tg_adv * tg_org).sum(1) / (np.linalg.norm(tg_adv, axis=1) *
                                       np.linalg.norm(tg_org, axis=1) + 1e-12)
    r_target = np.clip((1 - cost) / 2, 0, 1)

    # r_inflate: local loss inflation
    res_adv, _ = _residual(teacher, x1, eps_adv, t)
    res_org, _ = _residual(teacher, x1, eps_orig, t)
    ratio = res_adv / np.clip(res_org, 1e-9, None)
    r_inflate = np.clip((ratio - INFL_OK) / (INFL_MAX - INFL_OK), 0, 1)

    # r_return: short trajectory contraction toward nearest mode
    x = torch.tensor(xt_adv, dtype=torch.float32)
    d0 = np.sqrt(((xt_adv[:, None] - MEANS[None]) ** 2).sum(-1)).min(1)
    with torch.no_grad():
        for _ in range(20):
            x = x + eta * teacher(x)
    d1 = np.sqrt(((x.numpy()[:, None] - MEANS[None]) ** 2).sum(-1)).min(1)
    r_return = np.clip((d1 - d0) / (np.abs(d0) + 1e-9), 0, 1)

    return dict(r_basin=r_basin, r_field=r_field, r_target=r_target,
                r_inflate=r_inflate, r_return=r_return)


def r_soft(scores):
    return np.mean([scores["r_field"], scores["r_target"],
                    scores["r_inflate"], scores["r_return"]], axis=0)


# ----------------------------------------------------------------------------
# Calibration: per gamma-bin largest certified eps_ball (LTT/HB on r_basin)
# ----------------------------------------------------------------------------

@dataclass
class RCANMCalib:
    eps_by_bin: dict          # bin -> certified eps_ball (or None=abort)
    soft_thresh: float
    diag: dict


def calibrate_rc_anm(teacher: Field, alpha: float, delta_r: float, rng,
                     n_batch=128, m=250, m_fit=30, pgd_steps=3, rel_step=0.15,
                     soft_quantile=0.8):
    """For each gamma-bin (manifold-adjacent t), fixed-sequence over EPS_GRID:
    certify the largest eps_ball with HB p(mean r_basin among accepted) <=
    delta_r. soft_thresh = soft_quantile of r_soft on the calibration set."""
    soft_vals = []

    def risk_for(eb, m_, bin_lo, bin_hi):
        rb_means = []
        for _ in range(m_):
            x1, lab, eps, t, _, _ = make_triplet(n_batch, rng)
            sel = (t >= bin_lo) & (t < bin_hi)
            if sel.sum() < 4:
                rb_means.append(0.0)
                continue
            adv = pgd_mine(teacher, x1, eps, t, pgd_steps, rel_step, eb)
            sc = safety_scores(teacher, x1[sel], lab[sel], eps[sel],
                               adv[sel], t[sel])
            rb_means.append(float(sc["r_basin"].mean()))
            soft_vals.append(r_soft(sc))
        return np.array(rb_means)

    eps_by_bin = {}
    diag = {}
    for b, (lo, hi) in enumerate(GAMMA_BINS):
        certified = None
        # largest-first: try eps descending, accept first that certifies
        for eb in sorted(EPS_GRID, reverse=True):
            rb = risk_for(eb, m, lo, hi)
            p = core.hb_pvalue(float(rb.mean()), m, alpha)
            diag[f"bin{b}_eps{eb}"] = dict(mean_r_basin=float(rb.mean()),
                                           p=float(p))
            if p <= delta_r:
                certified = eb
                break
        eps_by_bin[b] = certified
    soft_thresh = (float(np.quantile(np.concatenate(soft_vals), soft_quantile))
                   if soft_vals else 1.0)
    return RCANMCalib(eps_by_bin=eps_by_bin, soft_thresh=soft_thresh, diag=diag)
