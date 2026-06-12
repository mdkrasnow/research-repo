"""E1.1 (ARM A disp-pair) + E1.2 (ARM B endpoint-cert) on the 2D EqM toy.
Gate G2.

Arms (5 seeds each, matched 4000 steps bs 128):
  vanilla            — shared baseline / negative-control floor
  A: disp_uniform    — upstream Dispersive Loss (uniform pairs), coeff 0.5
  A: rc_hpm          — certified mined InfoNCE on penultimate activations
  A: rc_repulsive    — certified repulsive-only (F13 decomposition)
  A: oracle_pairs    — SupCon on activations with true labels (+)
  B: anm_cert        — frozen-teacher PGD endpoint mining, LTT eps_ball,
                       basin-certified (P5)
  B: anm_live_neg    — live-student PGD, eps x10, uncertified (-) damage arm

Primary metrics (P3a): ARM A = field MSE vs pooled MC reference field;
ARM B = attractor purity. Margins = 2 x vanilla seed-SD per metric.

G2: (a) loss finite; (b) aux/base ratio in [0.05, 20] and non-saturating
(last-quarter mean > 0.01); (c) arm beats vanilla on its primary by margin;
(d) damage arm worse than vanilla by margin.

Writes results/e1_12_results.json + results/e1_12_verdict.json.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm import core, eqm2d                                    # noqa: E402
from rc_hpm.eqm2d import (Field, GAMMA_BINS, MINE_BINS, RFFTeacher, attractor_purity,
                          mode_recall_distance,
                          basin_certify, calibrate_arm_a, calibrate_arm_b,
                          field_mse, gd_sample, get_ct, gamma_bin,
                          make_triplet, mine_binned, mode_coverage,
                          pgd_mine, reference_field, train_binned_gate,
                          voronoi_basin)                          # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

ALPHA = 0.10
DELTA_R = 0.05
STEPS = 4000
N_BATCH = 128
COEFF = 0.5
TEACHER_SNAP_STEP = 1000
EPS_GRID = (0.1, 0.25, 0.5, 0.8, 1.2)
LAM_GRID = [(bp, bm) for bp in (0.55, 0.65, 0.75, 0.85, 0.92, 0.97)
            for bm in (0.03, 0.08, 0.15, 0.25, 0.35, 0.45) if bm < bp]
N_SEEDS = 5
ARMS = ["vanilla", "disp_uniform", "rc_hpm", "rc_repulsive", "oracle_pairs",
        "anm_cert", "anm_live_neg"]

_REF = None


def get_ref():
    global _REF
    if _REF is None:
        _REF = reference_field()
    return _REF


def disp_loss_torch(act: torch.Tensor) -> torch.Tensor:
    """Upstream InfoNCE-L2 dispersive loss port (transport.py disp_loss)."""
    z = act.reshape(act.shape[0], -1)
    diff = torch.nn.functional.pdist(z).pow(2) / z.shape[1]
    diff = torch.cat([diff, diff, torch.zeros(z.shape[0])])
    return torch.log(torch.exp(-diff).mean())


def supcon_act(z: torch.Tensor, labels: np.ndarray, tau=0.5) -> torch.Tensor:
    lab = torch.tensor(labels)
    sim = (z @ z.T) / tau
    eye = torch.eye(len(lab), dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    pos = (lab[:, None] == lab[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    cnt = pos.sum(1).clamp_min(1)
    return -((logp * pos).sum(1) / cnt)[pos.any(1)].mean()


def rc_infonce_act(z, s, q, neg, pos_m, amb, bins, rhos, tau=0.5):
    """L_out InfoNCE on activations with certified per-bin weights (stopgrad)."""
    n = z.shape[0]
    v_neg = np.zeros((n, n)); omega = np.zeros((n, n)); v_pos = np.zeros((n, n))
    for b in MINE_BINS:                   # A4 gamma-window
        rho_hat, rho_plus, rho_amb = rhos[b]
        in_b = np.where(bins == b)[0]
        blk = np.ix_(in_b, in_b)
        v_neg[blk] = 1.0 - rho_hat(s[blk])
        omega[blk] = (1.0 - rho_amb(q[blk])) * core.PINNED["c_amb"]
        v_pos[blk] = 1.0 - rho_plus(s[blk])
    v_neg = torch.tensor(v_neg * neg, dtype=torch.float32)
    omega = torch.tensor(omega * amb, dtype=torch.float32)
    v_pos = torch.tensor(v_pos * pos_m, dtype=torch.float32)
    sim = (z @ z.T) / tau
    sim = sim.masked_fill(torch.eye(n, dtype=torch.bool), -1e9)
    exps = torch.exp(sim - sim.max().detach())
    D = (v_neg * exps).sum(1) + (omega * exps).sum(1)
    log_frac = -torch.log1p(D[:, None] / (exps + 1e-12))
    has_pos = v_pos.bool().any(1)
    if not has_pos.any():
        return torch.zeros(())
    cnt = v_pos.bool().sum(1).clamp_min(1)
    return -(((v_pos * log_frac).sum(1) / cnt)[has_pos]).mean()


def rc_repulsive_act(z, s, q, neg, amb, bins, rhos, tau=0.5):
    """Certified repulsive-only (F13): weighted dispersion over certified
    negatives + debiased ambiguous; no attraction term."""
    n = z.shape[0]
    v = np.zeros((n, n))
    for b in MINE_BINS:                   # A4 gamma-window
        rho_hat, _, rho_amb = rhos[b]
        in_b = np.where(bins == b)[0]
        blk = np.ix_(in_b, in_b)
        v[blk] = (1.0 - rho_hat(s[blk])) * neg[blk] + \
            (1.0 - rho_amb(q[blk])) * core.PINNED["c_amb"] * amb[blk]
    vt = torch.tensor(v, dtype=torch.float32)
    if vt.sum() <= 0:
        return torch.zeros(())
    sim = (z @ z.T) / tau
    return torch.log((vt * torch.exp(sim)).sum() / vt.sum() + 1e-12)


def run_arm(args):
    arm, seed = args
    torch.manual_seed(seed * 101 + 3)
    rng = np.random.default_rng(seed + 9000)
    field = Field()
    opt = torch.optim.Adam(field.parameters(), lr=1e-3)
    ema = deepcopy(field)
    for p in ema.parameters():
        p.requires_grad_(False)

    teacher = calA = None
    eps_ball = None
    teacher_field = None
    aux_ratio_log, risk_log = [], []
    aborted_to_vanilla = False

    alpha_used = None
    if arm in ("rc_hpm", "rc_repulsive"):
        teacher = RFFTeacher(seed=7)        # frozen, shared structure
        q_fn = train_binned_gate(teacher, seed, rng)
        # P2: full pre-registered alpha grid runs at Stages 0-1; use the
        # tightest endpoint that certifies, label the arm with it.
        for a_try in (0.10, 0.20):
            calA = calibrate_arm_a(teacher, q_fn, LAM_GRID, a_try, DELTA_R,
                                   rng, n_batch=N_BATCH)
            if not calA.aborted:
                alpha_used = a_try
                break
        if calA.aborted:
            aborted_to_vanilla = True       # P7

    t_start = time.time()
    for step in range(STEPS):
        x1, lab, eps, t, xt, target = make_triplet(N_BATCH, rng)

        if arm == "anm_cert" and step >= TEACHER_SNAP_STEP:
            if teacher_field is None:       # snapshot + calibrate ONCE
                teacher_field = deepcopy(ema)
                for p in teacher_field.parameters():
                    p.requires_grad_(False)
                eps_ball = calibrate_arm_b(teacher_field, EPS_GRID, ALPHA,
                                           DELTA_R, rng, n_batch=N_BATCH,
                                           m=250, m_fit=30)
                if eps_ball is None:
                    aborted_to_vanilla = True   # P7
            if eps_ball is not None:
                adv = pgd_mine(teacher_field, x1, eps, t, eps_ball=eps_ball)
                cert = basin_certify(teacher_field, x1, lab, adv, t)
                eps_used = np.where(cert[:, None], adv, eps)
                xt = t[:, None] * x1 + (1 - t[:, None]) * eps_used
                target = (x1 - eps_used) * get_ct(t)[:, None]
                if step % 100 == 0:
                    sel = cert & (t >= 2 / 3)
                    xt_orig = t[:, None] * x1 + (1 - t[:, None]) * eps
                    flip = (voronoi_basin(xt) != lab) & \
                        (voronoi_basin(xt_orig) == lab)
                    risk_log.append(float(flip[sel].mean())
                                    if sel.sum() else 0.0)
        elif arm == "anm_live_neg" and step >= TEACHER_SNAP_STEP:
            adv = pgd_mine(None, x1, eps, t, eps_ball=5.0, live_field=field)
            xt = t[:, None] * x1 + (1 - t[:, None]) * adv
            target = (x1 - adv) * get_ct(t)[:, None]

        xt_t = torch.tensor(xt, dtype=torch.float32)
        tg_t = torch.tensor(target, dtype=torch.float32)
        out, act = field(xt_t, return_act=True)
        base = ((out - tg_t) ** 2).mean()
        aux = torch.zeros(())
        if not aborted_to_vanilla:
            z = act / act.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if arm == "disp_uniform":
                aux = disp_loss_torch(act)
            elif arm == "oracle_pairs":
                aux = supcon_act(z, lab)
            elif arm in ("rc_hpm", "rc_repulsive"):
                emb = teacher(xt)
                s, q, neg, pos_m, amb, bins, _ = mine_binned(
                    emb, t, lab, q_fn, calA.lam, calA.rhos, 2, 8)
                if arm == "rc_hpm":
                    aux = rc_infonce_act(z, s, q, neg, pos_m, amb, bins,
                                         calA.rhos)
                else:
                    aux = rc_repulsive_act(z, s, q, neg, amb, bins, calA.rhos)
                if step % 100 == 0:
                    x1b, labb = x1, lab
                    eps2 = rng.normal(0, 1, (N_BATCH, 2))
                    xt2 = t[:, None] * x1b + (1 - t[:, None]) * eps2
                    s_aug = (emb * teacher(xt2)).sum(1)
                    y_same = (labb[:, None] == labb[None, :]).astype(float)
                    from rc_hpm.eqm2d import binned_batch_risks
                    risk_log.append(binned_batch_risks(
                        s, q, neg, pos_m, amb, bins, y_same, calA.rhos, s_aug))
        loss = base + COEFF * aux
        if not torch.isfinite(loss):
            return dict(arm=arm, seed=seed, finite=False)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), field.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
        if step % 50 == 0:
            denom = float(base.detach())
            aux_ratio_log.append(abs(float(aux.detach() if
                                 torch.is_tensor(aux) else aux)) /
                                 max(denom, 1e-9))

    grid, f_star, _ = get_ref()
    samples = gd_sample(field, 2048, np.random.default_rng(seed + 12345))
    out = dict(
        arm=arm, seed=seed, finite=True,
        aborted_to_vanilla=aborted_to_vanilla,
        field_mse=field_mse(field, grid, f_star),
        purity=attractor_purity(samples),
        coverage=mode_coverage(samples),
        recall_dist=mode_recall_distance(samples),
        wall=round(time.time() - t_start, 1))
    if aux_ratio_log:
        arl = np.array(aux_ratio_log)
        q4 = arl[3 * len(arl) // 4:]
        out.update(aux_base_mean=float(arl.mean()),
                   aux_base_last_quarter=float(q4.mean()),
                   aux_base_max=float(arl.max()))
    if risk_log:
        rl = np.array(risk_log)
        if rl.ndim == 2:
            out.update(train_risk_minus=float(rl[:, 0].mean()),
                       train_risk_plus=float(rl[:, 1].mean()))
        else:
            out.update(train_basin_risk=float(rl.mean()))
    if arm == "anm_cert":
        out["eps_ball"] = eps_ball
    if calA is not None and not calA.aborted:
        out["lam"] = list(calA.lam)
        out["alpha_used"] = alpha_used
    return out


def main():
    t0 = time.time()
    get_ref()                                # build once before forking
    jobs = [(arm, s) for arm in ARMS for s in range(N_SEEDS)]
    with ProcessPoolExecutor(7) as ex:
        rows = list(ex.map(run_arm, jobs))
    with open(os.path.join(RESULTS, "e1_12_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    def vals(arm, key):
        return np.array([r[key] for r in rows if r["arm"] == arm
                         and r.get("finite")])

    van_mse = vals("vanilla", "field_mse")
    van_rec = vals("vanilla", "recall_dist")
    mar_mse = 2 * van_mse.std(ddof=1)
    mar_rec = 2 * van_rec.std(ddof=1)

    verdict = {"gate": "G2", "vanilla": dict(
        field_mse=float(van_mse.mean()), recall_dist=float(van_rec.mean()),
        coverage=float(vals("vanilla", "coverage").mean()),
        margin_mse=float(mar_mse), margin_recall=float(mar_rec)), "arms": {}}
    for arm in ARMS[1:]:
        mse, rec = vals(arm, "field_mse"), vals(arm, "recall_dist")
        if len(mse) == 0:
            verdict["arms"][arm] = {"finite": False}
            continue
        d = dict(field_mse=float(mse.mean()), recall_dist=float(rec.mean()),
                 purity=float(vals(arm, "purity").mean()),
                 coverage=float(vals(arm, "coverage").mean()))
        rats = [r.get("aux_base_last_quarter") for r in rows
                if r["arm"] == arm and r.get("aux_base_last_quarter") is not None]
        if rats:
            d["aux_base_last_quarter"] = float(np.mean(rats))
            d["b_nonsaturating"] = bool(0.01 < np.mean(rats) < 20)
        if arm in ("disp_uniform", "rc_hpm", "rc_repulsive", "oracle_pairs"):
            d["primary"] = "field_mse"
            d["c_beats_vanilla"] = bool(mse.mean() < van_mse.mean() - mar_mse)
        elif arm == "anm_cert":
            d["primary"] = "recall_dist (Amendment A3)"
            d["c_beats_vanilla"] = bool(rec.mean() < van_rec.mean() - mar_rec)
        elif arm == "anm_live_neg":
            d["primary"] = "recall_dist (damage control)"
            d["d_damages"] = bool(rec.mean() > van_rec.mean() + mar_rec)
        verdict["arms"][arm] = d

    a_pass = any(verdict["arms"][a].get("c_beats_vanilla") for a in
                 ("rc_hpm", "rc_repulsive"))
    b_pass = verdict["arms"]["anm_cert"].get("c_beats_vanilla", False)
    d_ok = verdict["arms"]["anm_live_neg"].get("d_damages", False)
    verdict["summary"] = dict(armA_pass=bool(a_pass), armB_pass=bool(b_pass),
                              damage_visible=bool(d_ok))
    verdict["branch"] = (
        "both arms pass -> carry both to E1.3" if a_pass and b_pass else
        "one arm passes -> carry winner to E1.3" if a_pass or b_pass else
        "harm-bounding only" if d_ok else "EqM bridge weak -> see tree")
    verdict["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "e1_12_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
