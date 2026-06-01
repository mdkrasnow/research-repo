"""Plots for Experiment 3. Reads the CSVs written by eval_fidelity_diversity.py.

All plots write to <out>/plots/. Matplotlib only (Agg backend, no display).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path):
    return list(csv.DictReader(open(path)))


def _agg_by_arm(out):
    rows = _read_csv(Path(out) / "aggregate_metrics.csv")
    return {r["checkpoint_type"]: r for r in rows}


def make_all(out):
    out = Path(out)
    pdir = out / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    agg = _agg_by_arm(out)

    _bars_fid_kid(agg, pdir / "aggregate_fid_kid_bars_with_ci.png")
    _scatter_pr(agg, pdir / "precision_vs_recall.png")
    _scatter_dc(agg, pdir / "density_vs_coverage.png")
    _classifier_hist(out, pdir / "classifier_histogram_vanilla_vs_anm.png")
    _per_class_feature_scatter(out, pdir / "per_class_feature_distance_scatter.png")
    _per_class_fid_scatter(out, pdir / "per_class_fid_scatter.png")
    _bottom_quartile(out, pdir / "bottom_quartile_improvement.png")
    _class_delta_sorted(out, pdir / "class_delta_sorted_by_vanilla_weakness.png")
    print(f"[plots] wrote -> {pdir}")


def _bars_fid_kid(agg, path):
    arms = ["vanilla", "anm"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fid = [float(agg[a]["fid"]) for a in arms]
    lo = [float(agg[a]["fid"]) - float(agg[a]["fid_ci_low"]) for a in arms]
    hi = [float(agg[a]["fid_ci_high"]) - float(agg[a]["fid"]) for a in arms]
    ax[0].bar(arms, fid, yerr=[np.abs(lo), np.abs(hi)], capsize=6,
              color=["#888", "#2a7"])
    ax[0].set_title("FID (↓)")
    kid = [float(agg[a]["kid_mean"]) for a in arms]
    kerr = [float(agg[a]["kid_std"]) for a in arms]
    ax[1].bar(arms, kid, yerr=kerr, capsize=6, color=["#888", "#2a7"])
    ax[1].set_title("KID (↓)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _scatter_pr(agg, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for a, c in [("vanilla", "#888"), ("anm", "#2a7")]:
        ax.scatter(float(agg[a]["recall"]), float(agg[a]["precision"]),
                   s=120, label=a, color=c)
    ax.set_xlabel("recall (diversity ↑)")
    ax.set_ylabel("precision (fidelity ↑)")
    ax.set_title("Precision vs Recall")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _scatter_dc(agg, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for a, c in [("vanilla", "#888"), ("anm", "#2a7")]:
        ax.scatter(float(agg[a]["coverage"]), float(agg[a]["density"]),
                   s=120, label=a, color=c)
    ax.set_xlabel("coverage (mode coverage ↑)")
    ax.set_ylabel("density (↑)")
    ax.set_title("Density vs Coverage")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _classifier_hist(out, path):
    rows = _read_csv(Path(out) / "classifier_histogram.csv")
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["checkpoint_type"]].append((int(r["predicted_class_id"]),
                                             float(r["probability"])))
    fig, ax = plt.subplots(figsize=(9, 4))
    for a, c in [("vanilla", "#888"), ("anm", "#2a7")]:
        probs = np.array([p for _, p in sorted(by_arm[a])])
        ax.plot(np.sort(probs)[::-1], label=a, color=c)
    ax.set_xlabel("class rank")
    ax.set_ylabel("predicted probability (sorted)")
    ax.set_title("Generated class histogram (sorted) — flatter = more balanced")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _per_class_feature_scatter(out, path):
    rows = _read_csv(Path(out) / "class_metrics.csv")
    v = {int(r["class_id"]): float(r["feature_distance_class"])
         for r in rows if r["checkpoint_type"] == "vanilla"}
    a = {int(r["class_id"]): float(r["feature_distance_class"])
         for r in rows if r["checkpoint_type"] == "anm"}
    cids = sorted(set(v) & set(a))
    vx = [v[c] for c in cids]; ay = [a[c] for c in cids]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(vx, ay, s=8, alpha=0.4)
    lim = [0, max(max(vx), max(ay)) * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("vanilla per-class feature distance")
    ax.set_ylabel("anm per-class feature distance")
    ax.set_title("Per-class feature distance (below diagonal = ANM better)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _per_class_fid_scatter(out, path):
    rows = _read_csv(Path(out) / "class_metrics.csv")
    def fid_map(arm):
        return {int(r["class_id"]): float(r["fid_class"])
                for r in rows if r["checkpoint_type"] == arm and r["fid_class"] != ""}
    v, a = fid_map("vanilla"), fid_map("anm")
    cids = sorted(set(v) & set(a))
    if len(cids) < 10:
        return  # not reliable enough to plot
    vx = [v[c] for c in cids]; ay = [a[c] for c in cids]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(vx, ay, s=8, alpha=0.4)
    lim = [0, max(max(vx), max(ay)) * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("vanilla per-class FID (NOISY @50/class)")
    ax.set_ylabel("anm per-class FID (NOISY)")
    ax.set_title("Per-class FID — interpret with caution")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _bottom_quartile(out, path):
    import json
    s = json.loads((Path(out) / "aggregate_metrics.json").read_text())
    bq = s.get("bottom_quartile_fid", {})
    if bq.get("vanilla") is None or bq.get("anm") is None:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["vanilla", "anm"], [bq["vanilla"], bq["anm"]], color=["#888", "#2a7"])
    ax.set_title("Bottom-quartile pooled FID (↓)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _class_delta_sorted(out, path):
    rows = _read_csv(Path(out) / "delta_class_metrics.csv")
    pairs = [(r["class_quartile"], float(r["delta_feature_mean_distance"]))
             for r in rows]
    # sort by vanilla weakness proxy: bottom first
    order = {"bottom": 0, "middle": 1, "top": 2, "": 3}
    pairs.sort(key=lambda x: order.get(x[0], 3))
    deltas = [p[1] for p in pairs]
    colors = {"bottom": "#c33", "middle": "#888", "top": "#39c", "": "#bbb"}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(deltas)), deltas,
           color=[colors[p[0]] for p in pairs])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("class (sorted: weak→strong by vanilla quartile)")
    ax.set_ylabel("Δ feature distance (anm − vanilla; <0 = ANM better)")
    ax.set_title("Per-class improvement sorted by vanilla weakness")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
