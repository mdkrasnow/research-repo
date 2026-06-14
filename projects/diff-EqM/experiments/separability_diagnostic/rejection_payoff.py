"""Task 1 — rejection-payoff analysis (label-enrichment, cached-only).

Consumes a completed run dir. Ranks samples by each available score, rejects the
bottom-k% (predicted-garbage) for k in {10,20,30,40}, and measures the PAYOFF as
label-enrichment in the kept set:

    enrichment(k) = good_fraction(kept) - good_fraction(all)
    mean_nn_kept  = mean nn_dist over kept   (lower = better; secondary payoff)

Every score's curve is bracketed by controls so a number is interpretable:
  - random        : reject a random k% (floor)
  - norm-only     : reject by endpoint ||f|| only (the confounder)
  - shuffled-score: same score, rows permuted (kills real ranking -> ~random)
  - shuffled-label: labels permuted (kills payoff -> ~0 enrichment; metric sanity)

FID payoff is a separate, GPU-bound script (fid_payoff.py); hook left below.
numpy-only, fixed seed, fails gracefully naming the missing path + next command.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

KS = [10, 20, 30, 40]
SEED = 0


def die(path, nxt):
    print(f"[rejection_payoff] MISSING: {path}\n[next] {nxt}")
    sys.exit(2)


def load(folder):
    folder = Path(folder)
    labels = folder / "labels.csv"
    scores = folder / "scores.csv"
    if not labels.exists():
        die(labels, "run compute_quality_labels.py on this run dir first.")
    if not scores.exists():
        die(scores, "run compute_scores.py on this run dir first.")
    lab = {}
    with open(labels) as fh:
        for r in csv.DictReader(fh):
            lab[int(r["sample_id"])] = (r["label"], float(r["nn_dist"]))
    rows = []
    with open(scores) as fh:
        for r in csv.DictReader(fh):
            if r.get("regime", "fixed") != "fixed":
                continue
            sid = int(r["sample_id"])
            if sid not in lab:
                continue
            rows.append((sid, r))
    return lab, rows


def orient_map(folder):
    """Read per-score orientation (+1 higher=garbage, -1 higher=good) from auroc_table."""
    t = Path(folder) / "results" / "auroc_table.csv"
    o = {}
    if t.exists():
        with open(t) as fh:
            for r in csv.DictReader(fh):
                if r.get("regime") == "fixed":
                    o[r["score"]] = int(float(r["orientation"]))
    return o


def good_frac(labels_arr):
    m = labels_arr >= 0
    return float(labels_arr[m].mean()) if m.any() else float("nan")


def reject_curve(badness, y, nn, k):
    """Reject top-k% badness; return enrichment + mean nn over kept good/garbage set."""
    n = len(badness)
    n_rej = int(round(n * k / 100.0))
    order = np.argsort(-badness)  # most-garbage first
    kept = np.ones(n, bool)
    kept[order[:n_rej]] = False
    base = good_frac(y)
    enr = good_frac(y[kept]) - base
    mnn = float(nn[kept].mean()) if kept.any() else float("nan")
    return enr, mnn, n_rej


def main(a):
    rng = np.random.default_rng(SEED)
    lab, rows = load(a.folder)
    orient = orient_map(a.folder)
    sids = np.array([s for s, _ in rows])
    y = np.array([1 if lab[s][0] == "good" else (0 if lab[s][0] == "garbage" else -1) for s in sids])
    nn = np.array([lab[s][1] for s in sids])
    keep = y >= 0
    dicts = [d for (_, d), k in zip(rows, keep) if k]
    sids, y, nn = sids[keep], y[keep], nn[keep]
    n_good, n_garb = int((y == 1).sum()), int((y == 0).sum())
    if n_good < 10 or n_garb < 10:
        die("enough good+garbage", f"only n_good={n_good} n_garb={n_garb}; re-run with more --num-samples.")

    score_cols = [c for c in dicts[0].keys() if c.startswith("s") and c[1:].isdigit()]
    scores = {c: np.array([float(r[c]) for r in dicts]) for c in score_cols}
    norm_end = np.array([float(r.get("norm_at_kstar", "nan")) for r in dicts])

    out = []  # (method, k, enrichment, mean_nn_kept, n_rejected)

    def add(name, badness, yy=y, nnv=nn):
        for k in KS:
            enr, mnn, nr = reject_curve(badness, yy, nnv, k)
            out.append((name, k, enr, mnn, nr))

    # real scores (badness = orient * score; default +1 if unknown)
    for c, v in scores.items():
        if c == "s4":  # latent-NN baseline (no f) -> report as its own row
            continue
        add(c, orient.get(c, 1) * v)
    # baselines / controls
    add("norm_only", norm_end)
    add("latent_nn_s4", orient.get("s4", 1) * scores["s4"]) if "s4" in scores else None
    rand = rng.random(len(y))
    add("random", rand)
    # shuffled-score control: take best real score, permute its values
    best = max((c for c in scores if c != "s4"),
               key=lambda c: abs(np.corrcoef(orient.get(c, 1) * scores[c], y)[0, 1]))
    add(f"shuf_score[{best}]", orient.get(best, 1) * rng.permutation(scores[best]))
    # shuffled-label control: real best score, permute labels (payoff must vanish)
    y_shuf = rng.permutation(y)
    nn_shuf = nn[rng.permutation(len(nn))]
    for k in KS:
        enr, mnn, nr = reject_curve(orient.get(best, 1) * scores[best], y_shuf, nn_shuf, k)
        out.append((f"shuf_label[{best}]", k, enr, mnn, nr))

    res = Path(a.folder) / "results"
    res.mkdir(parents=True, exist_ok=True)
    csv_p = res / "rejection_payoff.csv"
    with open(csv_p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "k_pct", "enrichment", "mean_nn_kept", "n_rejected"])
        for row in out:
            w.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}", row[4]])

    # markdown: enrichment @k table
    methods = []
    for m, *_ in out:
        if m not in methods:
            methods.append(m)
    lines = ["# Rejection-payoff (label-enrichment)", "",
             f"Run: `{a.folder}`  seed={SEED}  n_good={n_good} n_garb={n_garb}  base_good_frac={good_frac(y):.3f}",
             "",
             "Payoff = good-fraction(kept) − base. Higher = score concentrates garbage in the rejected tail.",
             "Controls: random (floor), norm_only (confounder), shuf_score/shuf_label (must ≈0).",
             "",
             "| method | " + " | ".join(f"enr@{k}" for k in KS) + " | mean_nn@40 |",
             "|" + "---|" * (len(KS) + 2)]
    by = {}
    for m, k, enr, mnn, _ in out:
        by.setdefault(m, {})[k] = (enr, mnn)
    for m in methods:
        cells = " | ".join(f"{by[m][k][0]:+.3f}" for k in KS)
        lines.append(f"| {m} | {cells} | {by[m][40][1]:.3f} |")
    # verdict on best real score vs controls @30
    real = [m for m in methods if m.startswith("s") and not m.startswith("shuf")]
    if real:
        best_real = max(real, key=lambda m: by[m][30][0])
        gain = by[best_real][30][0] - by["random"][30][0]
        shuf = next((by[m][30][0] for m in methods if m.startswith("shuf_label")), 0.0)
        lines += ["", f"**Best real score @30%:** {best_real} enr={by[best_real][30][0]:+.3f} "
                      f"(random {by['random'][30][0]:+.3f}, norm_only {by['norm_only'][30][0]:+.3f}, "
                      f"shuf_label {shuf:+.3f}). Gain over random = {gain:+.3f}."]
    lines += ["", "## FID-payoff hook",
              "Label-enrichment is the cheap payoff. For FID payoff (rank→keep→FID vs random/oracle), "
              "run `fid_payoff.py` (GPU; Inception). This script leaves that to the GPU path."]
    md = res / "REJECTION_PAYOFF.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"[rejection_payoff] wrote {csv_p} and {md}")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="completed run dir (has labels.csv, scores.csv)")
    main(ap.parse_args())
