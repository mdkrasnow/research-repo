"""Stage 4 of the EqM Separability Diagnostic -- the actual experiment.

For each candidate score, in each regime, does it separate good from garbage,
and does that separation SURVIVE de-confounding from the gradient norm?

Outputs:
  results/auroc_table.csv  : raw + within-norm-bin AUROC for every (score, regime, tau)
  results/plots/*.png      : good-vs-garbage histograms + AUROC bar chart
  results/VERDICT.txt      : GREEN / WEAK / KILL with the reasoning

Verdict keys off best_independent_auroc = max over {s1, s3} of within-norm-bin
AUROC, because s1 (dot) and s3 (path integral) are the only candidates that can
carry information NOT already in the gradient norm. s2 and s5 are norm-coupled
by construction; if only they separate, the low-norm-but-garbage quadrant
(spurious minimum) is undetectable and the metacognition matrix collapses to a
single row distinguished by norm alone.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

SCORES = ["s1", "s2", "s3", "s4", "s5"]
SCORE_DESC = {
    "s1": "dot energy  -<f,x>  (de-confoundable)",
    "s2": "l2 energy  0.5||f||^2  (norm-coupled)",
    "s3": "path integral  sum<f,dx>  (de-confoundable)",
    "s4": "latent NN dist  (no f; label sanity)",
    "s5": "post-step residual ||f||  (norm-coupled)",
}
INDEPENDENT = ["s1", "s3"]   # scores that can de-confound from norm


def _auc(labels, score):
    """AUROC via the Mann-Whitney rank statistic (no sklearn dependency).
    labels in {0,1}; returns P(score[pos] > score[neg]) with tie correction."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels == 1
    n_pos = int(pos.sum()); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auroc(score, is_garbage):
    """AUROC for detecting garbage. Orientation-free: returns max(a, 1-a) and
    the chosen orientation (+1 if higher score => more garbage)."""
    m = np.isfinite(score) & np.isfinite(is_garbage)
    if m.sum() < 10 or len(np.unique(is_garbage[m])) < 2:
        return float("nan"), 0
    a = _auc(is_garbage[m].astype(int), score[m])
    if not np.isfinite(a):
        return float("nan"), 0
    if a >= 0.5:
        return float(a), +1
    return float(1.0 - a), -1


def within_norm_bin_auroc(score, is_garbage, norm, n_bins=5):
    """Mean AUROC within quantile bins of the gradient norm -> strips out any
    separation that is merely norm-in-disguise.

    Returns (mean_auc, n_usable_bins, n_total). n_usable_bins==0 with adequate
    n_total means every norm-bin is single-class: the label is FULLY determined
    by the norm in these bins, so no norm-independent signal can be established
    (a collapse signal, NOT a sample-size problem)."""
    m = np.isfinite(score) & np.isfinite(norm)
    score, is_garbage, norm = score[m], is_garbage[m], norm[m]
    n_total = len(score)
    if n_total < 50:
        return float("nan"), 0, n_total
    edges = np.quantile(norm, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    aucs = []
    for b in range(n_bins):
        sel = (norm >= edges[b]) & (norm < edges[b + 1])
        if sel.sum() < 20 or len(np.unique(is_garbage[sel])) < 2:
            continue
        a, _ = auroc(score[sel], is_garbage[sel])
        if np.isfinite(a):
            aucs.append(a)
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs), n_total


def load_merged(folder):
    labels = {}
    with open(folder / "labels.csv") as fh:
        for r in csv.DictReader(fh):
            labels[int(r["sample_id"])] = r["label"]
    rows = []
    with open(folder / "scores.csv") as fh:
        for r in csv.DictReader(fh):
            sid = int(r["sample_id"])
            lab = labels.get(sid)
            if lab not in ("good", "garbage"):
                continue
            rows.append({
                "sample_id": sid, "regime": r["regime"], "tau_norm": float(r["tau_norm"]),
                "is_garbage": 1.0 if lab == "garbage" else 0.0,
                "norm_at_kstar": float(r["norm_at_kstar"]),
                **{s: float(r[s]) for s in SCORES},
            })
    return rows


def subset(rows, regime, tau=None):
    out = [r for r in rows if r["regime"] == regime and (tau is None or r["tau_norm"] == tau)]
    arr = {k: np.asarray([r[k] for r in out], dtype=np.float64) for k in
           SCORES + ["is_garbage", "norm_at_kstar"]}
    return arr


def main(args):
    folder = Path(args.folder)
    res = folder / "results"
    (res / "plots").mkdir(parents=True, exist_ok=True)
    rows = load_merged(folder)
    n_good = sum(1 for r in rows if r["is_garbage"] == 0 and r["regime"] == "fixed")
    n_garb = sum(1 for r in rows if r["is_garbage"] == 1 and r["regime"] == "fixed")
    print(f"[sep-diag/analyze] merged good={n_good} garbage={n_garb}", flush=True)

    # regimes to report: fixed (de-confound) + each threshold tau
    taus = sorted({r["tau_norm"] for r in rows if r["regime"] == "threshold"})
    regimes = [("fixed", None)] + [("threshold", t) for t in taus]

    table = []
    for regime, tau in regimes:
        arr = subset(rows, regime, tau)
        for s in SCORES:
            raw, orient = auroc(arr[s], arr["is_garbage"])
            binned, n_bins_used, n_tot = within_norm_bin_auroc(
                orient * arr[s], arr["is_garbage"], arr["norm_at_kstar"], args.n_bins)
            table.append({"regime": regime, "tau_norm": tau, "score": s,
                          "raw_auroc": raw, "within_norm_auroc": binned,
                          "n_bins_used": n_bins_used, "n_total": n_tot,
                          "orientation": orient, "desc": SCORE_DESC[s]})

    with open(res / "auroc_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["regime", "tau_norm", "score", "raw_auroc",
                                           "within_norm_auroc", "n_bins_used", "n_total",
                                           "orientation", "desc"])
        w.writeheader()
        for t in table:
            w.writerow({**t, "raw_auroc": f"{t['raw_auroc']:.4f}",
                        "within_norm_auroc": f"{t['within_norm_auroc']:.4f}"})

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arr_fixed = subset(rows, "fixed", None)
        fig, axes = plt.subplots(1, len(SCORES), figsize=(4 * len(SCORES), 3.2))
        for ax, s in zip(axes, SCORES):
            g = arr_fixed[s][arr_fixed["is_garbage"] == 0]
            b = arr_fixed[s][arr_fixed["is_garbage"] == 1]
            g, b = g[np.isfinite(g)], b[np.isfinite(b)]
            lo = np.nanpercentile(np.concatenate([g, b]), 1) if len(g) + len(b) else 0
            hi = np.nanpercentile(np.concatenate([g, b]), 99) if len(g) + len(b) else 1
            bins = np.linspace(lo, hi, 40)
            ax.hist(g, bins=bins, alpha=0.6, label="good", density=True)
            ax.hist(b, bins=bins, alpha=0.6, label="garbage", density=True)
            ax.set_title(f"{s} ({SCORE_DESC[s].split('(')[0].strip()})", fontsize=8)
            ax.legend(fontsize=7)
        fig.suptitle("good vs garbage, fixed regime", fontsize=10)
        fig.tight_layout()
        fig.savefig(res / "plots" / "hist_scores.png", dpi=110)
        plt.close(fig)

        # AUROC bar: within-norm AUROC, fixed regime
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fx = [t for t in table if t["regime"] == "fixed"]
        x = np.arange(len(SCORES))
        raw = [next(t["raw_auroc"] for t in fx if t["score"] == s) for s in SCORES]
        wb = [next(t["within_norm_auroc"] for t in fx if t["score"] == s) for s in SCORES]
        ax.bar(x - 0.2, raw, 0.4, label="raw AUROC")
        ax.bar(x + 0.2, wb, 0.4, label="within-norm-bin AUROC")
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.axhline(0.8, color="g", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(SCORES); ax.set_ylim(0.4, 1.0)
        ax.set_title("AUROC (fixed regime): raw vs de-confounded"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(res / "plots" / "auroc_bars.png", dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"[sep-diag/analyze] plotting skipped: {e}", flush=True)

    # ---- verdict ----
    fixed = {t["score"]: t for t in table if t["regime"] == "fixed"}
    best_ind = max((fixed[s]["within_norm_auroc"] for s in INDEPENDENT
                    if np.isfinite(fixed[s]["within_norm_auroc"])), default=float("nan"))
    best_ind_score = max(INDEPENDENT,
                         key=lambda s: (fixed[s]["within_norm_auroc"]
                                        if np.isfinite(fixed[s]["within_norm_auroc"]) else -1))
    s4_raw = fixed["s4"]["raw_auroc"]
    norm_coupled_sep = max(fixed["s2"]["within_norm_auroc"], fixed["s5"]["within_norm_auroc"])

    lines = []
    lines.append("EqM SEPARABILITY DIAGNOSTIC -- VERDICT")
    lines.append("=" * 50)
    lines.append(f"samples: good={n_good} garbage={n_garb}")
    lines.append(f"label sanity (s4 latent-NN raw AUROC, should be >~0.6): {s4_raw:.3f}")
    if np.isfinite(s4_raw) and s4_raw < 0.6:
        lines.append("  !! s4 weak -> LABEL PIPELINE SUSPECT; treat all below with caution.")
    lines.append("")
    lines.append("de-confounded (within-norm-bin) AUROC, fixed regime:")
    for s in SCORES:
        lines.append(f"  {s}: raw={fixed[s]['raw_auroc']:.3f}  "
                     f"within-norm={fixed[s]['within_norm_auroc']:.3f}   {SCORE_DESC[s]}")
    lines.append("")
    lines.append(f"best_independent_auroc = {best_ind:.3f}  (score {best_ind_score})")
    lines.append("")
    # distinguish "nan because too few samples" from "nan because every norm-bin
    # is single-class" (label fully norm-determined -> collapse, a KILL signal).
    ind_bins_used = max(fixed[s]["n_bins_used"] for s in INDEPENDENT)
    ind_n_total = max(fixed[s]["n_total"] for s in INDEPENDENT)
    if not np.isfinite(best_ind):
        if ind_n_total >= 40 and ind_bins_used == 0:
            verdict = ("KILL (norm-collapse): every gradient-norm bin is single-class, "
                       "so the good/garbage split is FULLY determined by the norm and no "
                       "norm-INDEPENDENT signal can be established. The spurious-minimum "
                       "quadrant is undetectable; the 2x2 collapses to one row (norm). "
                       "Metacognition sampler is not justified.")
        else:
            verdict = (f"INCONCLUSIVE: independent within-norm AUROC is NaN with only "
                       f"n={ind_n_total} good+garbage and {ind_bins_used} usable bins. "
                       f"Re-run with more samples (raise --num-samples / widen quantiles).")
    elif best_ind >= 0.80:
        verdict = (f"GREEN LIGHT: a norm-INDEPENDENT energy signal exists "
                   f"({best_ind_score} within-norm AUROC {best_ind:.3f} >= 0.80). "
                   f"The spurious-minimum quadrant is detectable. Build the "
                   f"metacognition sampler next; it is now properly de-confounded.")
    elif best_ind < 0.60:
        verdict = (f"KILL: no local signal separates true from spurious minima "
                   f"independent of gradient norm (best {best_ind:.3f} < 0.60). "
                   f"Both downstream ideas (metacognition sampler, conservativity "
                   f"fix) are high-risk. Stop or rethink.")
        if norm_coupled_sep >= 0.70:
            verdict += (f" NOTE: norm-coupled scores DO separate "
                        f"(s2/s5 within-norm {norm_coupled_sep:.3f}) -> bottom-right "
                        f"quadrant NOT detectable; matrix collapses to one row (norm).")
    else:
        verdict = (f"WEAK: separation exists but below the 0.80 action threshold "
                   f"(best {best_ind:.3f}). Quantified target: lift {best_ind_score} "
                   f">= 0.80. This is how much a 'more reliable scalar' must improve "
                   f"before the metacognition sampler is worth building.")
    lines.append("VERDICT: " + verdict)
    (res / "VERDICT.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Stage 1 output dir (has scores.csv, labels.csv)")
    ap.add_argument("--n-bins", type=int, default=5, help="quantile bins for matched-norm control")
    args = ap.parse_args()
    main(args)
