"""RC-ANM ladder on the 2D EqM toy (replaces failed E1.1/E1.2 contrastive
bridge). Pre-registered gate G-RCANM (preregistration-rc-anm.md) -> R1..R4.

Arms (5 seeds): vanilla / fixed_anm / aggressive_anm / rc_anm /
oracle_safe_anm. Certified object = the mined ENDPOINT.

Writes results/rcanm_results.json + results/rcanm_verdict.json.
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
from rc_hpm import rc_anm                                          # noqa: E402
from rc_hpm.eqm2d import (Field, get_ct, gamma_bin, make_triplet, pgd_mine,
                          gd_sample, field_mse, mode_coverage,
                          mode_recall_distance, reference_field,
                          voronoi_basin)                           # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
ALPHA, DELTA_R = 0.10, 0.05
STEPS, N_BATCH = 4000, 128
SNAP = 1000
N_SEEDS = 5
ARMS = ["vanilla", "fixed_anm", "aggressive_anm", "rc_anm", "oracle_safe_anm"]
_REF = None


def get_ref():
    global _REF
    if _REF is None:
        _REF = reference_field()
    return _REF


def run_arm(args):
    arm, seed = args
    torch.manual_seed(seed * 101 + 3)
    rng = np.random.default_rng(seed + 9000)
    field = Field()
    opt = torch.optim.Adam(field.parameters(), lr=1e-3)
    ema = deepcopy(field)
    for p in ema.parameters():
        p.requires_grad_(False)
    teacher = None
    calib = None
    risk_log = []
    accept_frac = []
    t0 = time.time()

    for step in range(STEPS):
        x1, lab, eps, t, xt, target = make_triplet(N_BATCH, rng)

        if arm != "vanilla" and step >= SNAP:
            if teacher is None:
                teacher = deepcopy(ema)
                for p in teacher.parameters():
                    p.requires_grad_(False)
                if arm == "rc_anm":
                    calib = rc_anm.calibrate_rc_anm(teacher, ALPHA, DELTA_R,
                                                    rng, n_batch=N_BATCH)
            eps_used = eps.copy()
            if arm == "fixed_anm":
                eps_used = pgd_mine(teacher, x1, eps, t, eps_ball=0.5)
            elif arm == "aggressive_anm":
                eps_used = pgd_mine(teacher, x1, eps, t, eps_ball=1.5)
            elif arm == "oracle_safe_anm":
                adv = pgd_mine(teacher, x1, eps, t, eps_ball=0.8)
                xt_adv = t[:, None] * x1 + (1 - t[:, None]) * adv
                safe = voronoi_basin(xt_adv) == lab        # true basin
                eps_used = np.where(safe[:, None], adv, eps)
                accept_frac.append(float(safe.mean()))
            elif arm == "rc_anm":
                bins = gamma_bin(t)
                adv = eps.copy()
                # per-bin certified eps_ball; ABORT bin -> un-mined
                for b, (lo, hi) in enumerate(rc_anm.GAMMA_BINS):
                    eb = calib.eps_by_bin.get(b)
                    if eb is None:
                        continue
                    sel = (t >= lo) & (t < hi)
                    if sel.sum() == 0:
                        continue
                    mined = pgd_mine(teacher, x1[sel], eps[sel], t[sel],
                                     eps_ball=eb)
                    adv[sel] = mined
                # accept/reject on r_basin (+ soft nuisance)
                sc = rc_anm.safety_scores(teacher, x1, lab, eps, adv, t)
                accept = (sc["r_basin"] < 0.5) & \
                    (rc_anm.r_soft(sc) <= calib.soft_thresh)
                eps_used = np.where(accept[:, None], adv, eps)
                accept_frac.append(float(accept.mean()))
                if step % 100 == 0:
                    # realized basin risk among ACCEPTED (manifold-adjacent)
                    hi_t = t >= 2 / 3
                    m = accept & hi_t
                    if m.sum():
                        risk_log.append(float(sc["r_basin"][m].mean()))

            xt = t[:, None] * x1 + (1 - t[:, None]) * eps_used
            target = (x1 - eps_used) * get_ct(t)[:, None]

        out = field(torch.tensor(xt, dtype=torch.float32))
        loss = ((out - torch.tensor(target, dtype=torch.float32)) ** 2).mean()
        if not torch.isfinite(loss):
            return dict(arm=arm, seed=seed, finite=False)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), field.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)

    grid, f_star, _ = get_ref()
    samples = gd_sample(field, 2048, np.random.default_rng(seed + 12345))
    out = dict(arm=arm, seed=seed, finite=True,
               field_mse=field_mse(field, grid, f_star),
               coverage=mode_coverage(samples),
               recall_dist=mode_recall_distance(samples),
               accept_frac=float(np.mean(accept_frac)) if accept_frac else None,
               realized_basin_risk=float(np.mean(risk_log)) if risk_log
               else None,
               eps_by_bin=(calib.eps_by_bin if calib else None),
               wall=round(time.time() - t0, 1))
    return out


def main():
    t0 = time.time()
    get_ref()
    jobs = [(arm, s) for arm in ARMS for s in range(N_SEEDS)]
    with ProcessPoolExecutor(min(8, len(jobs))) as ex:
        rows = list(ex.map(run_arm, jobs))
    with open(os.path.join(RESULTS, "rcanm_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    def vals(arm, key):
        return np.array([r[key] for r in rows if r["arm"] == arm
                         and r.get("finite") and r.get(key) is not None])

    van = vals("vanilla", "field_mse")
    margin = 2 * van.std(ddof=1)
    van_cov = vals("vanilla", "coverage")
    cov_margin = 2 * van_cov.std(ddof=1)
    tbl = {}
    for arm in ARMS:
        mse = vals(arm, "field_mse")
        tbl[arm] = dict(
            field_mse=float(mse.mean()) if len(mse) else None,
            field_mse_se=float(mse.std(ddof=1) / np.sqrt(len(mse)))
            if len(mse) > 1 else None,
            coverage=float(vals(arm, "coverage").mean()),
            recall_dist=float(vals(arm, "recall_dist").mean()),
            accept_frac=(float(vals(arm, "accept_frac").mean())
                         if len(vals(arm, "accept_frac")) else None),
            realized_basin_risk=(float(vals(arm, "realized_basin_risk").mean())
                                 if len(vals(arm, "realized_basin_risk"))
                                 else None))

    vmse = tbl["vanilla"]["field_mse"]
    a = bool(tbl["aggressive_anm"]["field_mse"] > vmse + margin)
    b = bool(tbl["rc_anm"]["field_mse"] <= vmse + margin)
    c = bool(tbl["rc_anm"]["field_mse"] <= tbl["fixed_anm"]["field_mse"] + margin
             and tbl["rc_anm"]["coverage"] >=
             tbl["fixed_anm"]["coverage"] - cov_margin)
    d = bool(tbl["rc_anm"]["realized_basin_risk"] is not None and
             tbl["rc_anm"]["realized_basin_risk"] <= ALPHA)

    if a and b and c and d:
        branch = "R1 BEST: RC-ANM works -> CIFAR mini"
    elif a and b and d and not c:
        branch = "R2 SAFETY-ONLY: RC-ANM safe but loses utility vs fixed ANM"
    elif not a:
        branch = "R3 NULL: aggressive ANM did not damage -> toy can't exercise premise"
    elif not b:
        branch = "R4 BROKEN: RC-ANM damages despite certification -> postmortem"
    else:
        branch = "NO-BRANCH -> STOP + postmortem"

    verdict = dict(gate="G-RCANM", alpha=ALPHA, vanilla_mse=vmse,
                   margin=float(margin), cov_margin=float(cov_margin),
                   table=tbl,
                   criteria=dict(a_premise_aggressive_damages=a,
                                 b_safety_rc_no_damage=b,
                                 c_utility_parity_vs_fixed=c,
                                 d_certification_realized=d),
                   branch=branch, wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "rcanm_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict["criteria"], indent=2))
    print("BRANCH:", branch)
    for arm in ARMS:
        t_ = tbl[arm]
        print(f"{arm:18} mse={t_['field_mse']:.3f} cov={t_['coverage']:.3f} "
              f"recall={t_['recall_dist']:.3f} acc={t_['accept_frac']} "
              f"risk={t_['realized_basin_risk']}")


if __name__ == "__main__":
    main()
