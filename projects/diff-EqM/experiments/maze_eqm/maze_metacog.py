"""Maze-EqM Step 3 — trajectory-metacognition on the trained EqM (CPU).

Does the descent-trajectory SHAPE predict maze-solve failure, and does probe-guided
restart beat random at equal NFE — on a REAL trained EqM, with EXACT BFS labels?

For each maze, draw R independent EqM-GD solutions (logging per-step norm/dot), label
each valid/invalid by BFS (exact oracle — no inception noise). Then:
  1. detection: train a trajectory-shape probe (reuse the image-probe feature builder)
     to predict invalid from descent dynamics; held-out de-confounded AUROC.
  2. action (best-of-R, equal NFE, same R draws for all arms):
       vanilla        = draw 0                       (floor)
       random-restart = keep a random draw            (NEG, compute-matched)
       probe-restart  = keep argmin P(invalid)        (TREATMENT)
       oracle-restart = keep any valid draw            (POS ceiling)
     metric = valid-path-rate. Success = probe-restart > random at equal NFE.

Run: python maze_metacog.py --ckpt runs/maze_c5/model.pt --data data/maze_c10_ood.npz \
     --steps 40 --eta 0.02 --R 4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eqm_maze import MazeEqM, load, gd_sample
from gen_maze_data import path_valid, decode_path

# reuse the image trajectory-shape feature builder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "separability_diagnostic"))
from probe_validate import feature_groups          # noqa: E402
from learned_probe import fit_logreg, auc, within_norm_auc  # noqa: E402


def valid_mask(paths, cond, thr=0.0):
    """paths:(M,1,H,W) -> bool array valid per maze."""
    out = np.zeros(len(paths), bool)
    for i in range(len(paths)):
        wall = cond[i, 0].cpu().numpy().astype(np.int8)
        s = tuple(np.argwhere(cond[i, 1].cpu().numpy() > 0)[0])
        g = tuple(np.argwhere(cond[i, 2].cpu().numpy() > 0)[0])
        out[i] = path_valid(decode_path(paths[i, 0], thr), wall, s, g)
    return out


def heldout_auc(X, y, norm_end, frac=0.3, seeds=5, l2=1.0):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    raws, wns = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(y)); nte = int(len(y) * frac)
        te, tr = idx[:nte], idx[nte:]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        w, b = fit_logreg((X[tr] - mu) / sd, y[tr], l2=l2)
        p = 1 / (1 + np.exp(-np.clip((X[te] - mu) / sd @ w + b, -30, 30)))
        raws.append(auc(y[te], p)); wns.append(within_norm_auc(p, y[te], norm_end[te]))
    return (float(np.mean(raws)) if raws else float("nan"),
            float(np.mean(wns)) if wns else float("nan"))


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    m = MazeEqM().to(dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    m.load_state_dict(ck["model"]); m.eval()
    cond, _ = load(args.data); cond = cond[:args.n].to(dev)
    M = cond.shape[0]

    # R draws per maze, logging descent dynamics + validity
    draw_norm, draw_dot, draw_valid, draw_paths = [], [], [], []
    for r in range(args.R):
        torch.manual_seed(args.seed * 100 + r)
        xt, norm, dot = gd_sample(m, cond, args.eta, args.steps, log=True)
        paths = xt.cpu().numpy()
        draw_norm.append(norm); draw_dot.append(dot)
        draw_valid.append(valid_mask(paths, cond, args.thr))
        draw_paths.append(paths)
        print(f"  draw {r}: valid={draw_valid[-1].mean():.3f}", flush=True)
    draw_valid = np.stack(draw_valid)             # (R, M) bool

    # ---- detection: probe over descent shape predicts INVALID (label=1) ----
    Xall, yall, norm_end_all = [], [], []
    for r in range(args.R):
        X = feature_groups(draw_norm[r], draw_dot[r])["ALL-shape"]
        Xall.append(X); yall.append((~draw_valid[r]).astype(float))
        norm_end_all.append(draw_norm[r][:, -1])
    Xall = np.concatenate(Xall); yall = np.concatenate(yall)
    norm_end_all = np.concatenate(norm_end_all)
    inv_rate = float(yall.mean())
    raw_auc, wn_auc = heldout_auc(Xall, yall, norm_end_all)

    # train a deployable probe on ALL draws for the action stage
    Xn = np.nan_to_num(Xall, nan=0.0, posinf=0.0, neginf=0.0)
    mu = Xn.mean(0); sd = Xn.std(0) + 1e-8
    w, b = fit_logreg((Xn - mu) / sd, yall, l2=1.0)

    def p_invalid(norm, dot):
        X = np.nan_to_num(feature_groups(norm, dot)["ALL-shape"], nan=0.0, posinf=0.0, neginf=0.0)
        return 1 / (1 + np.exp(-np.clip((X - mu) / sd @ w + b, -30, 30)))

    # ---- action: best-of-R at equal NFE ----
    pinv = np.stack([p_invalid(draw_norm[r], draw_dot[r]) for r in range(args.R)])  # (R,M)
    bi = np.arange(M)
    vanilla = draw_valid[0].mean()
    probe_pick = np.argmin(pinv, 0)
    probe = draw_valid[probe_pick, bi].mean()
    rng = np.random.default_rng(args.seed)
    rnd_pick = rng.integers(0, args.R, M)
    random_ = draw_valid[rnd_pick, bi].mean()
    oracle = (draw_valid.any(0)).mean()

    gap = float(probe - random_)
    band = float(oracle - random_)
    rec = gap / band if band > 1e-9 else float("nan")
    if gap > 0.01 and probe <= oracle + 1e-9:
        verdict = "PROBE>RANDOM at equal NFE — trajectory-metacognition works on real EqM maze planning"
    elif abs(gap) <= 0.01:
        verdict = "PROBE≈RANDOM — dynamics probe not actionable here"
    else:
        verdict = "PROBE<RANDOM — anti-correlated"

    res = {"data": Path(args.data).name, "grid": int(cond.shape[-1]), "M": M, "R": args.R,
           "steps": args.steps, "eta": args.eta, "invalid_rate": round(inv_rate, 4),
           "probe_raw_auc": round(raw_auc, 4), "probe_within_norm_auc": round(wn_auc, 4),
           "valid_rate": {"vanilla": round(float(vanilla), 4), "random_restart": round(float(random_), 4),
                          "probe_restart": round(float(probe), 4), "oracle_restart": round(float(oracle), 4)},
           "probe_minus_random": round(gap, 4),
           "fraction_oracle_recovered": None if not np.isfinite(rec) else round(float(rec), 3),
           "nfe_per_output": f"R*steps = {args.R}*{args.steps} (all arms share R draws)",
           "verdict": verdict}
    out = Path(args.out) if args.out else Path(__file__).parent / "runs" / "maze_metacog"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metacog_{Path(args.data).stem}_s{args.steps}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/maze_c5/model.pt")
    ap.add_argument("--data", default="data/maze_c10_ood.npz")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    main(args)
