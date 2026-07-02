"""Stage 5b: held-out validation + feature ablation, and SAVE the probe artifact.

CV said SHAPE-only de-confounded AUROC ~0.81. Two things remain before building
the sampler:
  1. Confirm it generalizes on a TRUE held-out split (70/30, not just CV folds).
  2. Ablate which shape feature groups carry the signal.
  3. Train on ALL good/garbage and SAVE (weights + standardizer + feature spec) so
     the probe-gated sampler can load and apply it during generation.

CPU-only, reuses the cached Stage-1 shard. numpy LR + Mann-Whitney AUROC.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

from learned_probe import (auc, within_norm_auc, fit_logreg, _down)  # reuse


# ---- feature groups (all magnitude-removed / de-confoundable) -------------- #
def feature_groups(norm, dot):
    eps = 1e-8
    nn = norm / (norm[:, :1] + eps)
    dd = dot / (np.abs(dot[:, :1]) + eps)
    T = norm.shape[1]
    steps = np.arange(T, dtype=np.float64)
    dnorm = np.diff(norm, axis=1); ddot = np.diff(dot, axis=1)
    osc_n = (np.sign(dnorm[:, 1:]) != np.sign(dnorm[:, :-1])).mean(1)
    osc_d = (np.sign(ddot[:, 1:]) != np.sign(ddot[:, :-1])).mean(1)
    rel_jitter = np.std(dnorm, axis=1) / (norm.mean(1) + eps)
    logn = np.log(norm + eps)
    slope = ((steps - steps.mean()) @ (logn - logn.mean(1, keepdims=True)).T) / ((steps - steps.mean()) ** 2).sum()
    q = max(2, T // 4)
    late = nn[:, -q:].mean(1) - nn[:, :q].mean(1)
    curv = np.abs(np.diff(nn, axis=1, n=2)).mean(1)
    groups = {
        "oscillation": np.stack([osc_n, osc_d, rel_jitter], axis=1),
        "slopes":      np.stack([slope, late, curv], axis=1),
        "norm_curve":  _down(nn, 16),
        "dot_curve":   _down(dd, 8),
    }
    groups["ALL-shape"] = np.concatenate([groups[k] for k in
                          ["oscillation", "slopes", "norm_curve", "dot_curve"]], axis=1)
    return groups


def load(folder):
    shards = sorted(glob.glob(str(Path(folder) / "logs" / "traj_rank*.npz")))
    P = {k: [] for k in ["sample_id", "norm", "dot"]}
    for s in shards:
        d = np.load(s)
        if d["sample_id"].shape[0] == 0:
            continue
        for k in P:
            P[k].append(d[k])
    out = {k: np.concatenate(v, 0) for k, v in P.items()}
    labels = {}
    with open(Path(folder) / "labels.csv") as fh:
        for r in csv.DictReader(fh):
            labels[int(r["sample_id"])] = r["label"]
    y = np.array([1.0 if labels.get(int(i)) == "garbage"
                  else (0.0 if labels.get(int(i)) == "good" else -1) for i in out["sample_id"]])
    keep = y >= 0
    return out["norm"][keep], out["dot"][keep], y[keep]


def holdout_eval(X, y, norm_end, frac=0.30, seed=0, l2=1.0):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); ntest = int(len(y) * frac)
    te, tr = idx[:ntest], idx[ntest:]
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
    w, b = fit_logreg((X[tr] - mu) / sd, y[tr], l2=l2)
    pte = 1.0 / (1.0 + np.exp(-np.clip((X[te] - mu) / sd @ w + b, -30, 30)))
    return auc(y[te], pte), within_norm_auc(pte, y[te], norm_end[te])


def main(args):
    norm, dot, y = load(args.folder)
    norm_end = norm[:, -1]
    n_good = int((y == 0).sum()); n_garb = int((y == 1).sum())
    print(f"[validate] good={n_good} garbage={n_garb} T={norm.shape[1]}", flush=True)
    groups = feature_groups(norm, dot)

    lines = ["EqM PROBE — held-out (30%) validation + ablation", "=" * 50,
             f"good={n_good} garbage={n_garb}; 5-seed held-out test AUROC (raw / within-norm)"]
    print(lines[-1], flush=True)
    summary = {}
    for gname, X in groups.items():
        raws, wns = [], []
        for s in range(5):
            r, wn = holdout_eval(X, y, norm_end, frac=0.30, seed=s, l2=args.l2)
            raws.append(r); wns.append(wn)
        raws, wns = np.array(raws), np.array(wns)
        summary[gname] = (raws.mean(), raws.std(), wns.mean(), wns.std())
        line = (f"  {gname:12s} dim={X.shape[1]:3d}  raw={raws.mean():.3f}±{raws.std():.3f}  "
                f"within-norm={wns.mean():.3f}±{wns.std():.3f}")
        lines.append(line); print(line, flush=True)

    # ---- train ALL-shape on ALL data, SAVE artifact for the sampler ---------
    Xall = np.nan_to_num(groups["ALL-shape"], nan=0.0, posinf=0.0, neginf=0.0)
    mu = Xall.mean(0); sd = Xall.std(0) + 1e-8
    w, b = fit_logreg((Xall - mu) / sd, y, l2=args.l2)
    art = Path(args.folder) / "probe_artifact.npz"
    np.savez(art, w=w, b=np.float64(b), mu=mu, sd=sd,
             feature_spec=json.dumps({"groups": ["oscillation", "slopes", "norm_curve", "dot_curve"],
                                      "down_nn": 16, "down_dd": 8}))
    lines.append(f"\nsaved probe artifact -> {art.name} (dim={Xall.shape[1]})")
    print(lines[-1], flush=True)

    best = max(summary.items(), key=lambda kv: kv[1][2])
    verdict = ("GENERALIZES" if summary["ALL-shape"][2] >= 0.78 else "DOES NOT hold out")
    lines.append(f"\nVERDICT: held-out ALL-shape within-norm = {summary['ALL-shape'][2]:.3f} -> {verdict}. "
                 f"Top group: {best[0]} ({best[1][2]:.3f}).")
    print(lines[-1], flush=True)
    (Path(args.folder) / "results").mkdir(exist_ok=True, parents=True)
    (Path(args.folder) / "results" / "PROBE_VALIDATION.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()
    main(args)
