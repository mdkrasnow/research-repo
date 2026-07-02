"""Plots for Experiment 4. Headless (Agg). All take the per-seed DataFrame-like
list of row dicts and write PNGs. Individual seed points are always shown;
vanilla<->anm are paired with lines per seed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_COLORS = {"vanilla": "#4477AA", "anm": "#EE6677"}


def _by(rows, regime, ctype, key):
    vals = [r[key] for r in rows if r["regime"] == regime and r["checkpoint_type"] == ctype and r.get(key) is not None]
    return np.array([v for v in vals if np.isfinite(v)], dtype=float)


def _paired(rows, regime, key):
    """Return {seed: {vanilla:v, anm:v}} for paired lines."""
    out = {}
    for r in rows:
        if r["regime"] != regime:
            continue
        out.setdefault(r["seed"], {})[r["checkpoint_type"]] = r.get(key)
    return out


def seed_error_bars(rows, key, ylabel, out_path, ci_label="std"):
    regimes = sorted({r["regime"] for r in rows})
    fig, axes = plt.subplots(1, max(1, len(regimes)), figsize=(6 * max(1, len(regimes)), 5), squeeze=False)
    for ax, regime in zip(axes[0], regimes):
        for xi, ctype in enumerate(["vanilla", "anm"]):
            v = _by(rows, regime, ctype, key)
            if v.size:
                ax.errorbar(xi, v.mean(), yerr=v.std() if v.size > 1 else 0,
                            fmt="o", color=_COLORS[ctype], capsize=6, markersize=10, label=ctype)
                ax.scatter(np.full(v.size, xi) + 0.05, v, color=_COLORS[ctype], alpha=0.5, zorder=3)
        # paired lines
        for seed, d in _paired(rows, regime, key).items():
            if d.get("vanilla") is not None and d.get("anm") is not None:
                ax.plot([0, 1], [d["vanilla"], d["anm"]], color="gray", alpha=0.4, zorder=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["vanilla", "anm"])
        ax.set_title(f"{regime}  (err={ci_label})"); ax.set_ylabel(ylabel)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def train_val_gap(rows, fid_key="fid_gap_val_minus_train", out_path="gap.png", ylabel="FID gap (val - train)"):
    regimes = sorted({r["regime"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 5))
    x = 0
    ticks, labels = [], []
    for regime in regimes:
        for ctype in ["vanilla", "anm"]:
            v = _by(rows, regime, ctype, fid_key)
            if v.size:
                ax.bar(x, v.mean(), yerr=v.std() if v.size > 1 else 0, color=_COLORS[ctype], capsize=5)
                ax.scatter(np.full(v.size, x), v, color="k", alpha=0.5, zorder=3)
            ticks.append(x); labels.append(f"{regime}\n{ctype}"); x += 1
        x += 0.5
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel(ylabel)
    ax.set_title("Train-val generalization gap (larger positive = closer to train)")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def nn_ratio_histogram(ratio_by_group: dict, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, arr in ratio_by_group.items():
        arr = np.asarray(arr); arr = arr[np.isfinite(arr)]
        if arr.size:
            ax.hist(arr, bins=60, alpha=0.5, label=label, density=True)
    ax.axvline(1.0, color="k", ls="--", lw=1, label="ratio=1")
    ax.axvline(0.9, color="r", ls=":", lw=1, label="0.9 (suspicious)")
    ax.set_xlabel("NN distance ratio  d_train / d_val"); ax.set_ylabel("density")
    ax.set_title("Memorization: lower ratio = closer to train"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def duplicate_rate_by_seed(rows, out_path, key="near_duplicate_rate"):
    fig, ax = plt.subplots(figsize=(7, 5))
    regimes = sorted({r["regime"] for r in rows})
    width = 0.35
    for ri, regime in enumerate(regimes):
        for ci, ctype in enumerate(["vanilla", "anm"]):
            v = _by(rows, regime, ctype, key)
            if v.size:
                ax.bar(ri + (ci - 0.5) * width, v.mean(), width,
                       yerr=v.std() if v.size > 1 else 0, color=_COLORS[ctype], capsize=4,
                       label=ctype if ri == 0 else None)
    ax.set_xticks(range(len(regimes))); ax.set_xticklabels(regimes)
    ax.set_ylabel(key); ax.set_title("Duplicate / near-duplicate rate"); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def step_vs_compute_summary(rows, out_path, key="val_fid", ylabel="Val FID"):
    fig, ax = plt.subplots(figsize=(7, 5))
    regimes = ["step_matched", "compute_matched"]
    for ctype in ["vanilla", "anm"]:
        xs, ys, es = [], [], []
        for ri, regime in enumerate(regimes):
            v = _by(rows, regime, ctype, key)
            if v.size:
                xs.append(ri); ys.append(v.mean()); es.append(v.std() if v.size > 1 else 0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt="o-", color=_COLORS[ctype], capsize=5, label=ctype, markersize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(regimes)
    ax.set_ylabel(ylabel); ax.set_title("Step-matched vs compute-matched"); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def render_nn_panel(gen_imgs, train_nn_imgs, val_nn_imgs, dists_train, dists_val, out_path, max_rows=8):
    """gen_imgs: (R,H,W,3) uint8; *_nn_imgs: (R,k,H,W,3). One row per suspicious sample."""
    R = min(max_rows, gen_imgs.shape[0])
    k = train_nn_imgs.shape[1] if train_nn_imgs.ndim == 5 else 0
    cols = 1 + 2 * k
    fig, axes = plt.subplots(R, cols, figsize=(2 * cols, 2 * R), squeeze=False)
    for r in range(R):
        axes[r][0].imshow(gen_imgs[r]); axes[r][0].set_title("gen", fontsize=7); axes[r][0].axis("off")
        for c in range(k):
            ax = axes[r][1 + c]; ax.imshow(train_nn_imgs[r, c]); ax.axis("off")
            ax.set_title(f"train d={dists_train[r, c]:.2f}", fontsize=7)
            ax = axes[r][1 + k + c]; ax.imshow(val_nn_imgs[r, c]); ax.axis("off")
            ax.set_title(f"val d={dists_val[r, c]:.2f}", fontsize=7)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
