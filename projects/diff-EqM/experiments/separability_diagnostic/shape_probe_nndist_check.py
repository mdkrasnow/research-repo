"""Skepticism control: is the SHAPE probe (p_bad from descent dynamics) just
another nn_dist proxy, like the endpoint E_psi head turned out to be?

Same held-out protocol as energy_ood_head.py: 70/30 split, fit logistic probe
on shape features on train, score held-out only, then check correlation with
nn_dist and AUROC after residualizing p_bad against nn_dist. Also within-
nn_dist-decile AUROC (matched-distance control, same spirit as the existing
within-norm-bin control in learned_probe.py / dynamics_probe.py).

Run: python shape_probe_nndist_check.py --folder runs/b2_vanilla --seed 0
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np

from learned_probe import auc, shape_feats, fit_logreg
from energy_ood_head import pearson, spearman


def load(folder):
    folder = Path(folder)
    shards = sorted(glob.glob(str(folder / "logs" / "traj_rank*.npz")))
    P = {k: [] for k in ["sample_id", "norm", "dot"]}
    for s in shards:
        d = np.load(s)
        if d["sample_id"].shape[0] == 0:
            continue
        for k in P:
            P[k].append(d[k])
    out = {k: np.concatenate(v, 0) for k, v in P.items()}
    lab, nn_dist = {}, {}
    with open(folder / "labels.csv") as fh:
        for r in csv.DictReader(fh):
            lab[int(r["sample_id"])] = r["label"]
            nn_dist[int(r["sample_id"])] = float(r["nn_dist"])
    y = np.array([1.0 if lab.get(int(i)) == "garbage"
                  else (0.0 if lab.get(int(i)) == "good" else -1)
                  for i in out["sample_id"]])
    nnd = np.array([nn_dist.get(int(i), np.nan) for i in out["sample_id"]])
    out["y"] = y
    out["nn_dist"] = nnd
    return out


def residual_auroc(score, nnd, y):
    m = np.isfinite(score) & np.isfinite(nnd) & np.isfinite(y)
    s, n, yy = score[m], nnd[m], y[m]
    A = np.stack([n, np.ones_like(n)], 1)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    resid = s - A @ coef
    return auc(yy, resid)


def within_decile_auroc(score, nnd, y, n_bins=10):
    m = np.isfinite(score) & np.isfinite(nnd) & np.isfinite(y)
    s, n, yy = score[m], nnd[m], y[m]
    edges = np.quantile(n, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-6
    aucs, used = [], 0
    for b in range(n_bins):
        sel = (n >= edges[b]) & (n < edges[b + 1])
        if sel.sum() < 10 or len(np.unique(yy[sel])) < 2:
            continue
        a = auc(yy[sel], s[sel])
        if np.isfinite(a):
            aucs.append(a); used += 1
    return (float(np.mean(aucs)) if aucs else float("nan")), used


def main(args):
    rng = np.random.default_rng(args.seed)
    folder = Path(args.folder)
    d = load(folder)
    clean = np.where(d["y"] >= 0)[0]
    perm = rng.permutation(len(clean))
    n_te = int(0.3 * len(clean))
    te_idx, tr_idx = clean[perm[:n_te]], clean[perm[n_te:]]

    norm, dot, y = d["norm"], d["dot"], d["y"]
    shp = np.nan_to_num(shape_feats(norm, dot), nan=0.0, posinf=0.0, neginf=0.0)

    mu = shp[tr_idx].mean(0); sd = shp[tr_idx].std(0) + 1e-8
    w, b = fit_logreg((shp[tr_idx] - mu) / sd, y[tr_idx], l2=1.0)
    p_bad_te = 1.0 / (1.0 + np.exp(-np.clip(((shp[te_idx] - mu) / sd) @ w + b, -30, 30)))
    y_te = y[te_idx]; nnd_te = d["nn_dist"][te_idx]

    raw_auroc = auc(y_te, p_bad_te)
    pr = pearson(p_bad_te, nnd_te)
    sr = spearman(p_bad_te, nnd_te)
    resid_auroc = residual_auroc(p_bad_te, nnd_te, y_te)
    decile_auroc, n_bins_used = within_decile_auroc(p_bad_te, nnd_te, y_te, n_bins=10)

    lines = ["# SHAPE probe -- nn_dist skepticism control", "",
             f"held-out n={len(te_idx)} (70/30 split, seed={args.seed})", "",
             f"1. raw shape-probe AUROC (held-out): {raw_auroc:.4f}",
             f"2. Pearson corr(p_bad, nn_dist): {pr:.4f}",
             f"   Spearman corr(p_bad, nn_dist): {sr:.4f}",
             f"3. AUROC after residualizing p_bad against nn_dist: {resid_auroc:.4f}",
             f"4. within-nn_dist-decile AUROC: {decile_auroc:.4f} ({n_bins_used}/10 usable bins)",
             ""]

    if not np.isfinite(resid_auroc) or resid_auroc < 0.65:
        verdict = (f"COLLAPSE: residual AUROC {resid_auroc:.3f} falls toward the baseline floor, "
                   "same failure mode as endpoint E_psi. DOWNGRADE the whole claim to "
                   "'early trajectory predicts nn_dist-defined failure', not "
                   "'trajectory dynamics reveal semantic OOD'.")
    elif resid_auroc >= raw_auroc - 0.05:
        verdict = (f"SURVIVES: residual AUROC {resid_auroc:.3f} stays close to raw {raw_auroc:.3f} "
                   "(and/or decile-matched AUROC holds). The shape probe carries "
                   "distance-independent signal -- main result survives this control.")
    else:
        verdict = (f"PARTIAL DROP: residual AUROC {resid_auroc:.3f} vs raw {raw_auroc:.3f} -- some "
                   "of the shape-probe signal is nn_dist-correlated, but not all of it. "
                   "Report both numbers; do not cite raw alone.")
    lines.append(f"## VERDICT: {verdict}")

    out = folder / "results" / "shape_probe_nndist_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "SHAPE_NNDIST_CHECK.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
