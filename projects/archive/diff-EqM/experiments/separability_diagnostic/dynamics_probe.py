"""PART 2 — learned dynamics probe.

Tests whether the SHAPE of the relaxation trajectory predicts garbage even when
endpoint scalar energy fails. Builds per-sample features from the cached logged
trajectories (norm/dot/l2/step_dot), trains simple logistic probes (sklearn if
available, else a numpy L2-logistic fallback), and compares a FULL-dynamics probe
against endpoint-only / norm-only / scalar-only baselines under a matched-final-
norm control.

Reads only: logs/*.npz, labels.csv (or quality_labels.csv). No regeneration.

Decision (FULL within-norm OOF AUROC): >=0.80 GREEN, 0.70-0.80 PROMISING, <0.65 weak.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

MIN_FOR_BINNING = 50
MIN_USABLE_BINS = 3   # within-norm AUROC needs >=this many mixed bins to de-confound


# --------------------------- AUROC + control ------------------------------- #
def auc(y, score):
    m = np.isfinite(score) & np.isfinite(y)
    y, score = y[m].astype(int), score[m]
    if len(y) < 10 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = y == 1; npos = int(pos.sum()); nneg = len(y) - npos
    a = (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return max(a, 1 - a)


def within_norm_auc(y, score, norm, n_bins=5):
    m = np.isfinite(score) & np.isfinite(norm) & np.isfinite(y)
    y, score, norm = y[m], score[m], norm[m]
    if len(y) < MIN_FOR_BINNING:
        return float("nan"), 0
    edges = np.quantile(norm, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-6
    aucs = []
    for b in range(n_bins):
        sel = (norm >= edges[b]) & (norm < edges[b + 1])
        if sel.sum() < 20 or len(np.unique(y[sel])) < 2:
            continue
        a = auc(y[sel], score[sel])
        if np.isfinite(a):
            aucs.append(a)
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


# --------------------------- logistic probe -------------------------------- #
def _np_logreg(X, y, l2=1.0, iters=800, lr=0.5):
    n, d = X.shape; w = np.zeros(d); b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w / n); b -= lr * g.mean()
    return w, b


def _fit_predict(Xtr, ytr, Xte, l2=1.0):
    """Try sklearn; fall back to numpy. Returns (test_probs, weights)."""
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0 / max(l2, 1e-6), max_iter=2000)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1], clf.coef_[0]
    except Exception:
        w, b = _np_logreg(Xtr, ytr, l2=l2)
        return 1.0 / (1.0 + np.exp(-np.clip(Xte @ w + b, -30, 30))), w


def cv_oof(X, y, k=5, seed=0, l2=1.0):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(y); idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k); oof = np.zeros(n); coefs = []
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(k) if g != f])
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        p, w = _fit_predict((X[tr] - mu) / sd, y[tr], (X[te] - mu) / sd, l2)
        oof[te] = p; coefs.append(w)
    return oof, np.mean(coefs, 0)


# --------------------------- features -------------------------------------- #
def _down(curve, m):
    T = curve.shape[1]; idx = np.linspace(0, T, m + 1).astype(int)
    return np.stack([curve[:, idx[j]:max(idx[j] + 1, idx[j + 1])].mean(1) for j in range(m)], 1)


def _slope(curve, lo, hi):
    seg = curve[:, lo:hi]; t = np.arange(seg.shape[1], dtype=np.float64)
    tc = t - t.mean()
    return (seg - seg.mean(1, keepdims=True)) @ tc / (tc @ tc + 1e-12)


def build_features(norm, dot, l2, step_dot):
    eps = 1e-8; T = norm.shape[1]; q = max(2, T // 4)
    dnorm = np.diff(norm, 1)
    feats, names = [], []

    def add(v, nm):
        feats.append(v); names.append(nm)

    add(norm[:, -1], "final_norm"); add(dot[:, -1], "final_dot"); add(l2[:, -1], "final_l2")
    add(step_dot.sum(1), "path_integral")
    add(_slope(norm, 0, q), "early_norm_slope"); add(_slope(norm, T - q, T), "late_norm_slope")
    add(norm[:, -1] / (norm[:, 0] + eps), "norm_drop_ratio")
    add(dot[:, -1] - dot[:, 0], "dot_change"); add(l2[:, -1] - l2[:, 0], "l2_change")
    add(norm.mean(1), "norm_auc"); add(dot.mean(1), "dot_auc"); add(l2.mean(1), "l2_auc")
    add((np.sign(dnorm[:, 1:]) != np.sign(dnorm[:, :-1])).mean(1), "norm_oscillation")
    cols = {}
    full = list(zip(names, feats))
    dn = _down(norm, 16); dd = _down(dot, 16); dl = _down(l2, 16)
    for j in range(16):
        full.append((f"norm_ds{j}", dn[:, j]))
    for j in range(16):
        full.append((f"dot_ds{j}", dd[:, j]))
    for j in range(16):
        full.append((f"l2_ds{j}", dl[:, j]))
    names = [n for n, _ in full]; X = np.stack([v for _, v in full], 1)
    # named groups for baselines
    cols["norm_only"] = [names.index("final_norm")]
    cols["endpoint_only"] = [names.index(n) for n in ["final_norm", "final_dot", "final_l2"]]
    cols["scalar_only"] = [names.index(n) for n in
                           ["final_norm", "final_dot", "final_l2", "path_integral",
                            "early_norm_slope", "late_norm_slope", "norm_drop_ratio",
                            "dot_change", "l2_change", "norm_auc", "dot_auc", "l2_auc",
                            "norm_oscillation"]]
    cols["FULL"] = list(range(len(names)))
    return X, names, cols


# --------------------------- io -------------------------------------------- #
def load(folder):
    import glob
    P = {k: [] for k in ["sample_id", "norm", "dot", "l2", "step_dot"]}
    for s in sorted(glob.glob(str(folder / "logs" / "traj_rank*.npz"))):
        d = np.load(s)
        if d["sample_id"].shape[0] == 0:
            continue
        for k in P:
            P[k].append(d[k])
    out = {k: np.concatenate(v, 0) for k, v in P.items()}
    lp = folder / "labels.csv"
    if not lp.exists():
        lp = folder / "quality_labels.csv"
    lab = {}
    with open(lp) as fh:
        for r in csv.DictReader(fh):
            lab[int(r["sample_id"])] = r["label"]
    y = np.array([1.0 if lab.get(int(i)) == "garbage"
                  else (0.0 if lab.get(int(i)) == "good" else -1) for i in out["sample_id"]])
    keep = y >= 0
    for k in out:
        out[k] = out[k][keep]
    out["y"] = y[keep]
    return out


def main(args):
    folder = Path(args.folder)
    out = Path(args.out) if args.out else folder / "results" / "dynamics_probe"
    out.mkdir(parents=True, exist_ok=True)
    d = load(folder)
    X, names, cols = build_features(d["norm"], d["dot"], d["l2"], d["step_dot"])
    y = d["y"]; norm_end = d["norm"][:, -1]
    n_good = int((y == 0).sum()); n_garb = int((y == 1).sum())

    table = []
    full_oof = None; full_coef = None
    for grp in ["norm_only", "endpoint_only", "scalar_only", "FULL"]:
        Xg = X[:, cols[grp]]
        oof, coef = cv_oof(Xg, y, k=args.n_folds, seed=args.seed, l2=args.l2)
        raw = auc(y, oof); wn, nb = within_norm_auc(y, oof, norm_end, args.n_bins)
        table.append({"feature_set": grp, "dim": len(cols[grp]),
                      "oof_auroc": raw, "within_norm_auroc": wn, "n_usable_bins": nb})
        if grp == "FULL":
            full_oof, full_coef = oof, coef

    with open(out / "probe_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature_set", "dim", "oof_auroc",
                                           "within_norm_auroc", "n_usable_bins"])
        w.writeheader()
        for r in table:
            w.writerow({**r, "oof_auroc": f"{r['oof_auroc']:.4f}",
                        "within_norm_auroc": f"{r['within_norm_auroc']:.4f}"})

    # feature importance (standardized logistic coef magnitude on FULL)
    order = np.argsort(-np.abs(full_coef))
    with open(out / "feature_importance.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["feature", "coef", "abs_coef"])
        for i in order:
            w.writerow([names[i], f"{full_coef[i]:.4f}", f"{abs(full_coef[i]):.4f}"])

    with open(out / "oof_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["sample_id", "y", "oof_prob_garbage"])
        for sid, yy, pp in zip(d["sample_id"], y, full_oof):
            w.writerow([int(sid), int(yy), f"{pp:.4f}"])

    full_row = next(r for r in table if r["feature_set"] == "FULL")
    full_wn = full_row["within_norm_auroc"]; full_nb = full_row["n_usable_bins"]
    base = {r["feature_set"]: r["within_norm_auroc"] for r in table}
    if not np.isfinite(full_wn):
        verdict, why = "INCONCLUSIVE", "FULL within-norm AUROC is NaN (too few samples / pure bins)."
    elif full_nb < MIN_USABLE_BINS:
        verdict, why = "INCONCLUSIVE", (f"FULL within-norm rests on only {full_nb} usable norm bins "
                                        f"(<{MIN_USABLE_BINS}): the norm determines the label in most bins "
                                        "(norm-collapse), so dynamics cannot be de-confounded here.")
    elif full_wn >= 0.80:
        verdict, why = "GREEN", ("Failure IS detectable in the relaxation dynamics (FULL within-norm "
                                 f"{full_wn:.3f} >= 0.80), even where endpoint scalars fail "
                                 f"(endpoint_only {base['endpoint_only']:.3f}, norm_only {base['norm_only']:.3f}).")
    elif full_wn >= 0.70:
        verdict, why = "PROMISING", (f"Dynamics within-norm {full_wn:.3f} in 0.70-0.80 — worth a small "
                                     "metacognitive sampler pilot (see METACOGNITIVE_RESCUE_SPEC.md).")
    elif full_wn < 0.65:
        verdict, why = "WEAK", f"Dynamics within-norm {full_wn:.3f} < 0.65 — do not chase the sampler yet."
    else:
        verdict, why = "BORDERLINE", f"Dynamics within-norm {full_wn:.3f} in 0.65-0.70 — ambiguous."

    md = ["# Dynamics probe summary", "",
          f"good={n_good} garbage={n_garb}; {args.n_folds}-fold OOF, seed={args.seed}", "",
          "| feature set | dim | OOF AUROC | within-norm AUROC |",
          "|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['feature_set']} | {r['dim']} | {r['oof_auroc']:.3f} | {r['within_norm_auroc']:.3f} |")
    md += ["",
           f"- FULL beats endpoint_only by {full_wn - base['endpoint_only']:+.3f} (within-norm)",
           f"- FULL beats norm_only by {full_wn - base['norm_only']:+.3f}",
           f"- FULL beats scalar_only by {full_wn - base['scalar_only']:+.3f}",
           "",
           "top features:"]
    for i in order[:8]:
        md.append(f"  - {names[i]}: {full_coef[i]:+.3f}")
    md += ["", f"## VERDICT: {verdict}", why]
    (out / "DYNAMICS_PROBE_SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    return verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()
    main(args)
