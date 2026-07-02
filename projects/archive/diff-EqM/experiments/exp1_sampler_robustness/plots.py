#!/usr/bin/env python3
"""
Plotting for Experiment 1. Reads results.csv (+ aux CSVs) and writes PNGs.
Headless (Agg). No GPU. Safe to rerun via --plots-only.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SAMPLERS = ["gd", "ngd"]
STEP_MULTS = [0.5, 1.0, 1.5, 2.0]
NFES = [10, 25, 50, 100, 250]
COLORS = {"vanilla": "tab:blue", "anm": "tab:red"}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def metric_vs_nfe(df, plots_dir, metric="fid"):
    fig, axes = plt.subplots(len(SAMPLERS), len(STEP_MULTS),
                             figsize=(4 * len(STEP_MULTS), 3.2 * len(SAMPLERS)),
                             squeeze=False, sharex=True)
    for r, sampler in enumerate(SAMPLERS):
        for c, sm in enumerate(STEP_MULTS):
            ax = axes[r][c]
            for ct in ["vanilla", "anm"]:
                sub = df[(df.checkpoint_type == ct) & (df.sampler == sampler)
                         & (df.step_mult == sm)].dropna(subset=[metric]).sort_values("nfe")
                if sub.empty:
                    continue
                ax.plot(sub.nfe, sub[metric], "o-", color=COLORS[ct], label=ct)
            ax.set_xscale("log"); ax.set_xticks(NFES)
            ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
            ax.set_title(f"{sampler}  step×{sm}", fontsize=9)
            if c == 0:
                ax.set_ylabel(metric.upper())
            if r == len(SAMPLERS) - 1:
                ax.set_xlabel("NFE (steps)")
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"{metric.upper()} vs NFE  (lower = better)")
    _save(fig, f"{plots_dir}/{metric}_vs_nfe_by_sampler.png")


def delta_heatmap(df, plots_dir, metric="fid"):
    col = "fid" if metric == "fid" else "kid_mean"
    for sampler in SAMPLERS:
        M = np.full((len(STEP_MULTS), len(NFES)), np.nan)
        for i, sm in enumerate(STEP_MULTS):
            for j, nfe in enumerate(NFES):
                def g(ct):
                    c = df[(df.checkpoint_type == ct) & (df.sampler == sampler)
                           & (df.nfe == nfe) & (df.step_mult == sm)]
                    return c.iloc[0][col] if len(c) else np.nan
                M[i, j] = g("anm") - g("vanilla")
        fig, ax = plt.subplots(figsize=(6, 4))
        vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(NFES))); ax.set_xticklabels(NFES)
        ax.set_yticks(range(len(STEP_MULTS))); ax.set_yticklabels(STEP_MULTS)
        ax.set_xlabel("NFE"); ax.set_ylabel("step+time mult")
        ax.set_title(f"ANM - vanilla {metric.upper()}  ({sampler})  (blue=ANM better)")
        for i in range(len(STEP_MULTS)):
            for j in range(len(NFES)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
        _save(fig, f"{plots_dir}/{metric}_delta_heatmap_{sampler}.png")


def auc_bar(auc_df, plots_dir):
    glob = auc_df[auc_df.step_mult == "ALL"]
    if glob.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(SAMPLERS)); w = 0.35
    for k, ct in enumerate(["vanilla", "anm"]):
        vals = [glob[(glob.sampler == s) & (glob.checkpoint_type == ct)].auc_log_fid.mean()
                for s in SAMPLERS]
        ax.bar(x + (k - 0.5) * w, vals, w, label=ct, color=COLORS[ct])
    ax.set_xticks(x); ax.set_xticklabels(SAMPLERS)
    ax.set_ylabel("log-NFE FID AUC (lower=better)")
    ax.set_title("Global FID-vs-NFE AUC"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{plots_dir}/fid_auc_barplot.png")


def low_nfe_stress_bars(auc_df, plots_dir):
    glob = auc_df[auc_df.step_mult == "ALL"]
    for field, fname, title in [("low_nfe_mean_fid", "low_nfe_score_barplot.png", "Low-NFE mean FID"),
                                ("stress_mean_fid", "stress_score_barplot.png", "Stress mean FID")]:
        if glob.empty or field not in glob:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(SAMPLERS)); w = 0.35
        for k, ct in enumerate(["vanilla", "anm"]):
            vals = [glob[(glob.sampler == s) & (glob.checkpoint_type == ct)][field].mean()
                    for s in SAMPLERS]
            ax.bar(x + (k - 0.5) * w, vals, w, label=ct, color=COLORS[ct])
        ax.set_xticks(x); ax.set_xticklabels(SAMPLERS)
        ax.set_ylabel(field); ax.set_title(title + " (lower=better)")
        ax.legend(); ax.grid(alpha=0.3)
        _save(fig, f"{plots_dir}/{fname}")


def nfe_to_match_plot(ntm_df, plots_dir):
    sub = ntm_df[ntm_df.vanilla_target_type == "vanilla_250_default"].copy()
    if sub.empty:
        return
    sub["nfe_num"] = pd.to_numeric(sub.anm_min_nfe_to_match, errors="coerce").fillna(300)
    fig, ax = plt.subplots(figsize=(6, 4))
    for sampler in SAMPLERS:
        s = sub[sub.sampler == sampler].sort_values("step_mult")
        ax.plot(s.step_mult, s.nfe_num, "o-", label=sampler)
    ax.axhline(250, ls="--", color="gray", label="vanilla 250")
    ax.set_xlabel("step+time mult"); ax.set_ylabel("ANM NFE to match vanilla 250-step FID")
    ax.set_title("NFE-to-match (lower=better; 300=did not match)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{plots_dir}/nfe_to_match.png")


def wallclock_pareto(df, plots_dir):
    if "wall_clock_sec" not in df or df.wall_clock_sec.isna().all():
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for ct in ["vanilla", "anm"]:
        sub = df[df.checkpoint_type == ct].dropna(subset=["fid", "wall_clock_sec"])
        ax.scatter(sub.wall_clock_sec, sub.fid, s=18, color=COLORS[ct], label=ct, alpha=0.7)
    ax.set_xlabel("wall clock (s)"); ax.set_ylabel("FID")
    ax.set_title("Wall-clock vs FID (Pareto)"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{plots_dir}/wallclock_vs_fid.png")


def plot_all(results_csv, metrics_dir, plots_dir):
    df = pd.read_csv(results_csv)
    metric_vs_nfe(df, plots_dir, "fid")
    metric_vs_nfe(df, plots_dir, "kid_mean")
    delta_heatmap(df, plots_dir, "fid")
    delta_heatmap(df, plots_dir, "kid")
    wallclock_pareto(df, plots_dir)
    for name, fn in [("auc_summary.csv", auc_bar), ("auc_summary.csv", low_nfe_stress_bars)]:
        p = f"{metrics_dir}/{name}"
        if os.path.exists(p):
            fn(pd.read_csv(p), plots_dir)
    ntm = f"{metrics_dir}/nfe_to_match.csv"
    if os.path.exists(ntm):
        nfe_to_match_plot(pd.read_csv(ntm), plots_dir)
