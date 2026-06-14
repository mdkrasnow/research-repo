"""PART 1 — cached robustness analysis (label-threshold sweep).

Re-derives good/garbage labels at several quantile choices FROM THE CACHED
nn_dist column (no feature/sample/label regeneration) and recomputes, per score:
raw AUROC, within-norm-bin AUROC, n usable norm bins, orientation. Classifies the
separability verdict as GREEN-ROBUST / GREEN-FRAGILE / WEAK / KILL / LABEL-BROKEN.

Reads only: labels.csv (or quality_labels.csv), scores.csv, (logs/*.npz unused here).
numpy only — no sklearn.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

INDEPENDENT = ["s1", "s3"]      # carry signal not in endpoint norm
NORM_COUPLED = ["s2", "s5"]     # norm-in-disguise
SANITY = "s4"                   # latent-NN baseline (no f) -> label sanity
ALL_SCORES = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
MIN_FOR_BINNING = 50
MIN_USABLE_BINS = 3             # within-norm AUROC needs >=this many mixed bins to
#                                trust as a de-confound; fewer => norm determines the
#                                label in (most) bins => norm-collapse.


def auc(y, score):
    """Mann-Whitney AUROC for detecting y==1, tie-corrected. NaN if degenerate."""
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
    return (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def oriented_auc(y, score):
    a = auc(y, score)
    if not np.isfinite(a):
        return float("nan"), 0
    return (a, +1) if a >= 0.5 else (1 - a, -1)


def within_norm_auc(y, score, norm, n_bins=5):
    m = np.isfinite(score) & np.isfinite(norm) & np.isfinite(y)
    y, score, norm = y[m], score[m], norm[m]
    if len(y) < MIN_FOR_BINNING:
        return float("nan"), 0, len(y)
    edges = np.quantile(norm, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-6
    aucs = []
    for b in range(n_bins):
        sel = (norm >= edges[b]) & (norm < edges[b + 1])
        if sel.sum() < 20 or len(np.unique(y[sel])) < 2:
            continue
        a, _ = oriented_auc(y[sel], score[sel])
        if np.isfinite(a):
            aucs.append(a)
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs), len(y)


def load_labels(folder):
    p = folder / "labels.csv"
    if not p.exists():
        p = folder / "quality_labels.csv"
    assert p.exists(), f"no labels.csv / quality_labels.csv in {folder}"
    sid, nn, ms = [], [], []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            sid.append(int(r["sample_id"]))
            nn.append(float(r.get("nn_dist", "nan")))
            ms.append(float(r.get("max_softmax", "nan")))
    return np.array(sid), np.array(nn), np.array(ms)


def load_scores(folder):
    """fixed-regime scores per sample_id -> {score: val, norm_at_kstar}."""
    out = {}
    with open(folder / "scores.csv") as fh:
        for r in csv.DictReader(fh):
            if r["regime"] != "fixed":
                continue
            d = {"norm": float(r["norm_at_kstar"])}
            for s in ALL_SCORES:
                if s in r and r[s] != "":
                    d[s] = float(r[s])
            out[int(r["sample_id"])] = d
    return out


def main(args):
    folder = Path(args.folder)
    out = Path(args.out) if args.out else folder / "results" / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    sid, nn, ms = load_labels(folder)
    sc = load_scores(folder)
    present = [s for s in ALL_SCORES if any(s in sc[i] for i in sc)]
    qs = [float(x) for x in args.label_quantiles.split(",")]

    rows = []
    per_q = {}   # q -> dict(score-> (raw, within, nbins))
    for q in qs:
        tau_lo = np.quantile(nn[np.isfinite(nn)], q)
        tau_hi = np.quantile(nn[np.isfinite(nn)], 1 - q)
        # align to scored samples
        ids = [i for i in sid if i in sc]
        idx = {i: k for k, i in enumerate(sid)}
        nn_a = np.array([nn[idx[i]] for i in ids])
        y = np.where(nn_a < tau_lo, 0.0, np.where(nn_a > tau_hi, 1.0, np.nan))
        keep = np.isfinite(y)
        ids_k = [i for i, kk in zip(ids, keep) if kk]
        y_k = y[keep]
        norm_k = np.array([sc[i]["norm"] for i in ids_k])
        ms_k = np.array([ms[idx[i]] for i in ids_k])
        per_q[q] = {}
        for s in present:
            sv = np.array([sc[i].get(s, np.nan) for i in ids_k])
            raw, orient = oriented_auc(y_k, sv)
            wn, nb, _ = within_norm_auc(y_k, orient * sv if np.isfinite(orient) else sv, norm_k, args.n_bins)
            per_q[q][s] = (raw, wn, nb)
            rows.append({"label_q": q, "n_good": int((y_k == 0).sum()),
                         "n_garbage": int((y_k == 1).sum()), "score": s,
                         "raw_auroc": raw, "within_norm_auroc": wn,
                         "n_usable_bins": nb, "orientation": orient,
                         "category": ("independent" if s in INDEPENDENT else
                                      "norm-coupled" if s in NORM_COUPLED else
                                      "sanity" if s == SANITY else "dynamics")})
        # max-softmax sanity (secondary): good should have higher max-softmax
        per_q[q]["_ms_sanity"] = oriented_auc(y_k, ms_k)[0]

    with open(out / "robustness_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label_q", "n_good", "n_garbage", "score",
                                           "raw_auroc", "within_norm_auroc", "n_usable_bins",
                                           "orientation", "category"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "raw_auroc": f"{r['raw_auroc']:.4f}",
                        "within_norm_auroc": f"{r['within_norm_auroc']:.4f}"})

    # ---- classify ----
    # only count a within-norm AUROC if it rests on >=MIN_USABLE_BINS mixed bins;
    # otherwise the norm determines the label in (most) bins -> not de-confounded.
    def best_indep(q):
        vals = [per_q[q][s][1] for s in INDEPENDENT
                if s in per_q[q] and np.isfinite(per_q[q][s][1]) and per_q[q][s][2] >= MIN_USABLE_BINS]
        return max(vals) if vals else float("nan")

    def best_normcoupled(q):
        vals = [per_q[q][s][1] for s in NORM_COUPLED if s in per_q[q] and np.isfinite(per_q[q][s][1])]
        return max(vals) if vals else float("nan")

    def best_raw_indep(q):
        vals = [per_q[q][s][0] for s in INDEPENDENT if s in per_q[q] and np.isfinite(per_q[q][s][0])]
        return max(vals) if vals else float("nan")

    indep = np.array([best_indep(q) for q in qs])
    s4san = np.array([per_q[q].get(SANITY, (np.nan, np.nan, 0))[1] for q in qs])
    nbins_indep = np.array([max([per_q[q][s][2] for s in INDEPENDENT if s in per_q[q]] + [0]) for q in qs])
    raw_indep = np.array([best_raw_indep(q) for q in qs])
    normcp = np.array([best_normcoupled(q) for q in qs])
    n_ge80 = int(np.nansum(indep >= 0.80))
    frac_ge80 = n_ge80 / len(qs)

    label_broken = np.nanmean(s4san) < 0.55
    # norm-collapse: the de-confounded (>=3-bin) independent signal is unavailable
    # everywhere, yet the RAW independent score separates -> the split is carried by
    # the norm itself (most bins single-class).
    norm_collapse = np.all(~np.isfinite(indep)) and np.nanmax(raw_indep) >= 0.60
    if label_broken:
        verdict = "LABEL-BROKEN"
    elif norm_collapse or np.nanmax(indep) < 0.60:
        verdict = "KILL"
    elif frac_ge80 >= 0.75:
        verdict = "GREEN-ROBUST"
    elif n_ge80 >= 1:
        verdict = "GREEN-FRAGILE"
    elif np.nanmax(indep) >= 0.60:
        verdict = "WEAK"
    else:
        verdict = "KILL"

    md = ["# Robustness summary — label-threshold sweep", "",
          f"label quantiles swept: {qs}", f"scores present: {present}", "",
          "| label_q | best independent (within-norm) | best norm-coupled | s4 sanity | usable bins |",
          "|---|---|---|---|---|"]
    for k, q in enumerate(qs):
        md.append(f"| {q} | {indep[k]:.3f} | {normcp[k]:.3f} | {s4san[k]:.3f} | {int(nbins_indep[k])} |")
    md += ["",
           f"- independent ≥0.80 at {n_ge80}/{len(qs)} thresholds ({100*frac_ge80:.0f}%)",
           f"- mean s4 label-sanity = {np.nanmean(s4san):.3f} (<0.55 => LABEL-BROKEN)",
           f"- max independent across thresholds = {np.nanmax(indep):.3f}",
           "",
           f"## VERDICT: {verdict}",
           {"GREEN-ROBUST": "Independent signal ≥0.80 across most thresholds — robust separability.",
            "GREEN-FRAGILE": "Independent ≥0.80 only under narrow threshold choices — fragile; treat with caution.",
            "WEAK": "Independent 0.60–0.80 — real but sub-actionable; see dynamics_probe.py for combined signal.",
            "KILL": "Independent <0.60 or norm-collapse — no usable norm-independent endpoint signal.",
            "LABEL-BROKEN": "Label-sanity baseline ~0.5 — labels uninformative; fix labels before trusting any AUROC."}[verdict]]
    (out / "ROBUSTNESS_SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    return verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--label-quantiles", default="0.2,0.25,0.3,0.35")
    ap.add_argument("--n-bins", type=int, default=5)
    args = ap.parse_args()
    main(args)
