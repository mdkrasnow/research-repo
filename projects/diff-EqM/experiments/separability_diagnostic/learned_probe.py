"""Stage 5 (new direction): does the COMBINED trajectory signal cross 0.80?

The single hand-crafted scalars capped ~0.67 (best: norm oscillation s8=0.674,
the first to beat the dumb latent-NN baseline). But each was tested alone. Here
we ask whether a LEARNED probe over the whole descent curve — the full norm /
dot / l2 / step_dot trajectories — separates good from garbage better, and
crucially whether the *shape* (magnitude-removed) signal is actionable.

Three logistic-regression probes (numpy, L2, 5-fold CV, out-of-fold preds):
  - MAG-only   : norm-magnitude features only -> the "is it just the norm?" floor
  - SHAPE-only : magnitude-normalized curves + dynamics (oscillation, slopes,
                 curvature) -> the DE-CONFOUNDED test. This is the decisive number.
  - FULL       : mag + shape -> upper bound.

Each probe's OOF predictions get a raw AUROC and a within-norm-bin AUROC (the
same matched-norm control used for the single scores). Verdict keys off SHAPE-only:
>=0.80 GREEN (combined dynamics is actionable; metacognition sampler reopens via a
learned dynamics probe), 0.674-0.80 WEAK (improves on best scalar but not
actionable), <=0.674 CAPPED (no lift from combining -> direction dead).

CPU-only, reuses the cached Stage-1 shard. No sklearn (numpy LR + Mann-Whitney AUROC).
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# AUROC (Mann-Whitney, tie-corrected) + within-norm-bin control
# --------------------------------------------------------------------------- #
def auc(labels, score):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels == 1
    npos = int(pos.sum()); nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    a = (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return max(a, 1 - a)


def within_norm_auc(score, y, norm, n_bins=5):
    edges = np.quantile(norm, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-6
    aucs = []
    for b in range(n_bins):
        sel = (norm >= edges[b]) & (norm < edges[b + 1])
        if sel.sum() < 20 or len(np.unique(y[sel])) < 2:
            continue
        a = auc(y[sel], score[sel])
        if np.isfinite(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


# --------------------------------------------------------------------------- #
# numpy L2 logistic regression
# --------------------------------------------------------------------------- #
def fit_logreg(X, y, l2=1.0, iters=800, lr=0.5):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        gw = X.T @ g / n + l2 * w / n
        gb = g.mean()
        w -= lr * gw; b -= lr * gb
    return w, b


def cv_oof_preds(X, y, k=5, seed=0, l2=1.0):
    """5-fold out-of-fold predicted probabilities (standardized per train fold)."""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    oof = np.zeros(n)
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(k) if g != f])
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        w, b = fit_logreg(Xtr, y[tr], l2=l2)
        oof[te] = 1.0 / (1.0 + np.exp(-np.clip(Xte @ w + b, -30, 30)))
    return oof


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def _down(curve, m):
    """Downsample a (N,T) curve to (N,m) by averaging contiguous chunks."""
    T = curve.shape[1]
    idx = np.linspace(0, T, m + 1).astype(int)
    return np.stack([curve[:, idx[j]:max(idx[j] + 1, idx[j + 1])].mean(1) for j in range(m)], axis=1)


def magnitude_feats(norm, dot, l2, step_dot):
    """Norm-magnitude features -> the 'is it just the norm?' control."""
    cum = np.cumsum(step_dot, axis=1)
    return np.stack([
        norm[:, -1], norm.mean(1), norm.max(1), norm.min(1),
        l2[:, -1], dot[:, -1], cum[:, -1],
    ], axis=1)


def shape_feats(norm, dot):
    """Magnitude-removed curve shape + dynamics -> the de-confounded signal."""
    eps = 1e-8
    nn = norm / (norm[:, :1] + eps)                      # normalized norm curve
    dd = dot / (np.abs(dot[:, :1]) + eps)
    T = norm.shape[1]
    steps = np.arange(T, dtype=np.float64)
    dnorm = np.diff(norm, axis=1)
    # oscillation (sign-flip fraction) for norm and dot
    osc_n = (np.sign(dnorm[:, 1:]) != np.sign(dnorm[:, :-1])).mean(1)
    ddot = np.diff(dot, axis=1)
    osc_d = (np.sign(ddot[:, 1:]) != np.sign(ddot[:, :-1])).mean(1)
    # log-decay slope, late/early slope of normalized curve
    logn = np.log(norm + eps)
    slope = ((steps - steps.mean()) @ (logn - logn.mean(1, keepdims=True)).T) / ((steps - steps.mean()) ** 2).sum()
    q = max(2, T // 4)
    late = (nn[:, -q:].mean(1) - nn[:, :q].mean(1))      # settle amount
    rel_jitter = np.std(dnorm, axis=1) / (norm.mean(1) + eps)
    curv = np.abs(np.diff(nn, axis=1, n=2)).mean(1)      # curvature of normalized curve
    feats = [osc_n, osc_d, slope, late, rel_jitter, curv]
    block = np.stack(feats, axis=1)
    # plus the magnitude-normalized curves, downsampled
    return np.concatenate([block, _down(nn, 16), _down(dd, 8)], axis=1)


# --------------------------------------------------------------------------- #
def load(folder):
    shards = sorted(glob.glob(str(Path(folder) / "logs" / "traj_rank*.npz")))
    P = {k: [] for k in ["sample_id", "norm", "dot", "l2", "step_dot"]}
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
    for k in out:
        out[k] = out[k][keep]
    out["y"] = y[keep]
    return out


def main(args):
    d = load(args.folder)
    norm, dot, l2, sd = d["norm"], d["dot"], d["l2"], d["step_dot"]
    y = d["y"]; norm_end = norm[:, -1]
    n_good = int((y == 0).sum()); n_garb = int((y == 1).sum())
    print(f"[probe] good={n_good} garbage={n_garb} T={norm.shape[1]}", flush=True)

    mag = magnitude_feats(norm, dot, l2, sd)
    shp = shape_feats(norm, dot)
    full = np.concatenate([mag, shp], axis=1)

    res = {}
    for name, X in [("MAG-only", mag), ("SHAPE-only", shp), ("FULL", full)]:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        oof = cv_oof_preds(X, y, k=args.folds, seed=args.seed, l2=args.l2)
        raw = auc(y, oof)
        wn = within_norm_auc(oof, y, norm_end, args.n_bins)
        res[name] = (raw, wn, X.shape[1])
        print(f"[probe] {name:11s} dim={X.shape[1]:3d}  raw_AUROC={raw:.3f}  within-norm={wn:.3f}", flush=True)

    shape_wn = res["SHAPE-only"][1]
    best_scalar = 0.674   # s8 norm-oscillation within-norm AUROC (the bar to beat)
    lines = ["EqM LEARNED-PROBE (combined dynamics) -- VERDICT", "=" * 52,
             f"good={n_good} garbage={n_garb}",
             f"MAG-only   raw={res['MAG-only'][0]:.3f} within-norm={res['MAG-only'][1]:.3f}  (norm floor)",
             f"SHAPE-only raw={res['SHAPE-only'][0]:.3f} within-norm={res['SHAPE-only'][1]:.3f}  (DE-CONFOUNDED, decisive)",
             f"FULL       raw={res['FULL'][0]:.3f} within-norm={res['FULL'][1]:.3f}  (upper bound)",
             f"best single scalar (s8 oscillation) within-norm = {best_scalar:.3f}", ""]
    if not np.isfinite(shape_wn):
        v = "INCONCLUSIVE: shape probe within-norm AUROC is NaN."
    elif shape_wn >= 0.80:
        v = (f"GREEN: combined SHAPE (de-confounded) probe reaches {shape_wn:.3f} >= 0.80. "
             f"The full descent-dynamics signal IS actionable -> metacognition sampler "
             f"reopens via a learned dynamics probe (not the energy scalar).")
    elif shape_wn > best_scalar + 0.02:
        v = (f"WEAK+: combining lifts the shape signal to {shape_wn:.3f} (> best scalar "
             f"{best_scalar:.3f}) but still < 0.80. Real, growing, not yet actionable. "
             f"Target quantified; richer dynamics features / temporal model is the lever.")
    else:
        v = (f"CAPPED: combining the full trajectory gives {shape_wn:.3f}, no real lift over "
             f"the best single scalar ({best_scalar:.3f}). The descent-dynamics signal is "
             f"genuinely capped ~0.67 -> metacognition-via-inference-signal direction is dead.")
    lines.append("VERDICT: " + v)
    out = Path(args.folder) / "results"; out.mkdir(parents=True, exist_ok=True)
    (out / "PROBE_VERDICT.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--n-bins", type=int, default=5)
    args = ap.parse_args()
    main(args)
