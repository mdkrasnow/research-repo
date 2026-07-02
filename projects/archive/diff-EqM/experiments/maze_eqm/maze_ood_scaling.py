"""Maze-EqM OOD-generalization study for trajectory-metacognition.

Question (user, overnight): does metacognition help MORE as you go OOD — i.e. does
probe-restart's advantage over vanilla GROW with distribution shift, and does an
in-distribution-trained probe still transfer to harder OOD tiers?

Design (sharp, two claims in one run):
  - One EqM trained on c7 (15x15). Eval on tiers {c5,c7,c9,c11,c13} = increasing OOD.
  - FIXED descent budget across all tiers (equal NFE everywhere). Vanilla valid-rate
    degrades naturally with OOD.
  - Train the metacognition probe ONCE on the in-dist tier (c7) draws, FREEZE it,
    apply the frozen probe to every tier for the action (probe-restart). This tests
    whether the in-dist probe generalizes, not just a per-tier refit.
  - Arms per tier (equal NFE, same R draws): vanilla / random-restart (neg) /
    probe-restart (treatment, frozen probe) / oracle (pos ceiling). Exact BFS labels.

Hypotheses:
  H1 vanilla degrades with OOD (sanity: task gets harder).
  H2 probe-restart − vanilla advantage GROWS with OOD (more instability-failures to catch).
  H3 frozen in-dist probe still flags OOD failures (AUROC stays > chance OOD).

Run: python maze_ood_scaling.py --ckpt runs/maze_gpu_s0/model.pt \
       --tiers 5 7 9 11 13 --steps 150 --eta 0.02 --R 4 --n 600 --out runs/maze_ood
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eqm_maze import MazeEqM, load, gd_sample
from maze_metacog import valid_mask
from probe_validate import feature_groups
from learned_probe import fit_logreg, auc, within_norm_auc


def draws_for(m, cond, eta, steps, R, seed):
    """R independent EqM-GD draws; return per-draw (norm,dot,valid,paths)."""
    dn, dd, dv = [], [], []
    for r in range(R):
        torch.manual_seed(seed * 100 + r)
        xt, norm, dot = gd_sample(m, cond, eta, steps, log=True)
        dn.append(norm); dd.append(dot)
        dv.append(valid_mask(xt.cpu().numpy(), cond))
    return dn, dd, np.stack(dv)


def feats(dn, dd, R):
    X = [feature_groups(dn[r], dd[r])["ALL-shape"] for r in range(R)]
    return [np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in X]


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    W = int(ck.get("args", {}).get("width", 64))
    m = MazeEqM(C=W).to(dev); m.load_state_dict(ck["model"]); m.eval()

    # ---- 1) train + FREEZE the probe on the in-dist tier (first tier == train tier) ----
    train_tier = args.train_tier
    dpath = lambda c: f"{args.data_dir}/maze_c{c}_ood.npz"
    cond_tr, _ = load(dpath(train_tier)); cond_tr = cond_tr[:args.n].to(dev)
    dn, dd, dv = draws_for(m, cond_tr, args.eta, args.steps, args.R, args.seed)
    Xtr = np.concatenate(feats(dn, dd, args.R))
    ytr = np.concatenate([(~dv[r]).astype(float) for r in range(args.R)])
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    w, b = fit_logreg((Xtr - mu) / sd, ytr, l2=1.0)
    print(f"[probe trained on c{train_tier}] inv_rate={ytr.mean():.3f}", flush=True)

    def p_invalid(norm, dot):
        X = np.nan_to_num(feature_groups(norm, dot)["ALL-shape"], nan=0.0, posinf=0.0, neginf=0.0)
        return 1 / (1 + np.exp(-np.clip((X - mu) / sd @ w + b, -30, 30)))

    # ---- 2) apply frozen probe to every tier; equal NFE, exact labels ----
    rows = []
    for c in args.tiers:
        cond, _ = load(dpath(c)); cond = cond[:args.n].to(dev); M = cond.shape[0]
        dn, dd, dv = draws_for(m, cond, args.eta, args.steps, args.R, args.seed)
        bi = np.arange(M)
        pinv = np.stack([p_invalid(dn[r], dd[r]) for r in range(args.R)])
        vanilla = float(dv[0].mean())
        probe = float(dv[np.argmin(pinv, 0), bi].mean())
        rng = np.random.default_rng(args.seed)
        random_ = float(dv[rng.integers(0, args.R, M), bi].mean())
        oracle = float(dv.any(0).mean())
        # frozen-probe detection AUROC on this tier (de-confounded within norm bins)
        yc = np.concatenate([(~dv[r]).astype(float) for r in range(args.R)])
        ne = np.concatenate([dn[r][:, -1] for r in range(args.R)])
        # score with FROZEN probe, then AUROC vs labels (raw + within-norm)
        sc = np.concatenate([p_invalid(dn[r], dd[r]) for r in range(args.R)])
        raw = float(auc(yc, sc)) if 0 < yc.mean() < 1 else float("nan")
        wn = float(within_norm_auc(sc, yc, ne)) if 0 < yc.mean() < 1 else float("nan")
        row = {"tier": int(c), "grid": int(cond.shape[-1]), "ood_dist": int(c - train_tier),
               "invalid_rate": round(float(yc.mean()), 4),
               "frozen_probe_auc_raw": round(raw, 4), "frozen_probe_auc_deconf": round(wn, 4),
               "vanilla": round(vanilla, 4), "random_restart": round(random_, 4),
               "probe_restart": round(probe, 4), "oracle_restart": round(oracle, 4),
               "probe_minus_vanilla": round(probe - vanilla, 4),
               "probe_minus_random": round(probe - random_, 4)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    # ---- 3) trend summary ----
    work = [r for r in rows if 0.05 < r["vanilla"] < 0.95]  # tiers with a workable band
    summary = {"ckpt": args.ckpt, "train_tier": train_tier, "budget": args.steps * args.eta,
               "steps": args.steps, "eta": args.eta, "R": args.R, "n": args.n,
               "rows": rows,
               "workable_tiers": [r["tier"] for r in work],
               "H1_vanilla_degrades_with_ood": None,
               "H2_advantage_grows_with_ood": None,
               "H3_frozen_probe_transfers_ood": None}
    if len(work) >= 2:
        ood = np.array([r["ood_dist"] for r in work], float)
        adv = np.array([r["probe_minus_vanilla"] for r in work], float)
        van = np.array([r["vanilla"] for r in work], float)
        auc_d = np.array([r["frozen_probe_auc_deconf"] for r in work], float)
        summary["H1_vanilla_degrades_with_ood"] = bool(np.corrcoef(ood, van)[0, 1] < 0)
        summary["H2_advantage_grows_with_ood"] = bool(np.corrcoef(ood, adv)[0, 1] > 0)
        summary["H2_corr_ood_vs_advantage"] = round(float(np.corrcoef(ood, adv)[0, 1]), 3)
        summary["H3_frozen_probe_transfers_ood"] = bool(np.nanmin(auc_d) > 0.55)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "ood_scaling.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tiers", type=int, nargs="+", default=[5, 7, 9, 11, 13])
    ap.add_argument("--train-tier", type=int, default=7)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/maze_ood")
    main(ap.parse_args())
