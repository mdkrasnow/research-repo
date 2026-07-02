#!/usr/bin/env python3
"""
Aggregate Experiment 1 results.csv into AUC / low-NFE / stress / nfe-to-match /
pairwise-delta tables. Pure pandas/numpy; no GPU, no sampling. Safe to rerun.
"""
import numpy as np
import pandas as pd

CHECKPOINT_TYPES = ["vanilla", "anm"]
SAMPLERS = ["gd", "ngd"]
STEP_MULTS = [0.5, 1.0, 1.5, 2.0]
LOW_NFE = [10, 25, 50]
STRESS_MULTS = [1.5, 2.0]


def _auc(nfe, vals, log=True):
    """Mean-height trapezoidal AUC over the NFE grid (lower is better)."""
    order = np.argsort(nfe)
    nfe, vals = np.asarray(nfe)[order], np.asarray(vals)[order]
    x = np.log(nfe) if log else nfe
    if len(x) < 2 or x[-1] == x[0]:
        return float("nan")
    return float(np.trapz(vals, x=x) / (x[-1] - x[0]))


def compute_auc(df, out_csv):
    rows = []
    for ct in CHECKPOINT_TYPES:
        for sampler in SAMPLERS:
            for sm in STEP_MULTS:
                sub = df[(df.checkpoint_type == ct) & (df.sampler == sampler)
                         & (df.step_mult == sm)].dropna(subset=["fid"]).sort_values("nfe")
                if sub.empty:
                    continue
                low = sub[sub.nfe.isin(LOW_NFE)]
                default = sub[(sub.nfe == 250) & (sub.step_mult == 1.0)]
                rows.append({
                    "checkpoint_type": ct, "sampler": sampler, "step_mult": sm,
                    "auc_log_fid": _auc(sub.nfe, sub.fid, log=True),
                    "auc_linear_fid": _auc(sub.nfe, sub.fid, log=False),
                    "auc_log_kid": _auc(sub.nfe, sub.kid_mean, log=True),
                    "low_nfe_mean_fid": low.fid.mean(),
                    "low_nfe_mean_kid": low.kid_mean.mean(),
                    "fid_at_250_default": float(default.fid.iloc[0]) if len(default) else np.nan,
                    "best_fid_over_grid": sub.fid.min(),
                    "best_condition": sub.loc[sub.fid.idxmin(), "run_id"],
                })
    # global (mean over step_mults) + stress, per checkpoint x sampler
    glob = []
    for ct in CHECKPOINT_TYPES:
        for sampler in SAMPLERS:
            sub = df[(df.checkpoint_type == ct) & (df.sampler == sampler)].dropna(subset=["fid"])
            if sub.empty:
                continue
            stress = sub[sub.nfe.isin(LOW_NFE) & sub.step_mult.isin(STRESS_MULTS)]
            per_sm = [r for r in rows if r["checkpoint_type"] == ct and r["sampler"] == sampler]
            glob.append({
                "checkpoint_type": ct, "sampler": sampler, "step_mult": "ALL",
                "auc_log_fid": np.nanmean([r["auc_log_fid"] for r in per_sm]),
                "auc_linear_fid": np.nanmean([r["auc_linear_fid"] for r in per_sm]),
                "auc_log_kid": np.nanmean([r["auc_log_kid"] for r in per_sm]),
                "low_nfe_mean_fid": sub[sub.nfe.isin(LOW_NFE)].fid.mean(),
                "low_nfe_mean_kid": sub[sub.nfe.isin(LOW_NFE)].kid_mean.mean(),
                "stress_mean_fid": stress.fid.mean() if len(stress) else np.nan,
                "stress_mean_kid": stress.kid_mean.mean() if len(stress) else np.nan,
                "best_fid_over_grid": sub.fid.min(),
                "best_condition": sub.loc[sub.fid.idxmin(), "run_id"],
            })
    out = pd.concat([pd.DataFrame(rows), pd.DataFrame(glob)], ignore_index=True)
    out.to_csv(out_csv, index=False)
    return out


def pairwise_delta(df, out_csv):
    rows = []
    for sampler in SAMPLERS:
        for nfe in sorted(df.nfe.unique()):
            for sm in STEP_MULTS:
                def cell(ct):
                    c = df[(df.checkpoint_type == ct) & (df.sampler == sampler)
                           & (df.nfe == nfe) & (df.step_mult == sm)]
                    return c.iloc[0] if len(c) else None
                v, a = cell("vanilla"), cell("anm")
                if v is None or a is None:
                    continue
                rows.append({
                    "sampler": sampler, "nfe": nfe, "step_mult": sm,
                    "fid_vanilla": v.fid, "fid_anm": a.fid,
                    "delta_fid": a.fid - v.fid,
                    "relative_delta_fid": (a.fid - v.fid) / v.fid if v.fid else np.nan,
                    "kid_vanilla": v.kid_mean, "kid_anm": a.kid_mean,
                    "delta_kid": a.kid_mean - v.kid_mean,
                    "anm_wins_fid": bool(a.fid < v.fid),
                    "anm_wins_kid": bool(a.kid_mean < v.kid_mean),
                })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def nfe_to_match(df, out_csv):
    rows = []
    for sampler in SAMPLERS:
        van = df[(df.checkpoint_type == "vanilla") & (df.sampler == sampler)].dropna(subset=["fid"])
        if van.empty:
            continue
        default = van[(van.nfe == 250) & (van.step_mult == 1.0)]
        targets = []
        if len(default):
            targets.append(("vanilla_250_default", float(default.fid.iloc[0])))
        targets.append(("vanilla_best_grid", float(van.fid.min())))
        for tname, tfid in targets:
            for sm in STEP_MULTS:
                anm = df[(df.checkpoint_type == "anm") & (df.sampler == sampler)
                         & (df.step_mult == sm)].dropna(subset=["fid"]).sort_values("nfe")
                hit = anm[anm.fid <= tfid]
                rows.append({
                    "sampler": sampler, "step_mult": sm,
                    "vanilla_target_type": tname, "vanilla_target_fid": tfid,
                    "anm_min_nfe_to_match": int(hit.iloc[0].nfe) if len(hit) else ">250",
                    "anm_condition_used": hit.iloc[0].run_id if len(hit) else None,
                    "matched": bool(len(hit)),
                })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def run_all(results_csv, metrics_dir):
    df = pd.read_csv(results_csv)
    compute_auc(df, f"{metrics_dir}/auc_summary.csv")
    pairwise_delta(df, f"{metrics_dir}/pairwise_delta.csv")
    nfe_to_match(df, f"{metrics_dir}/nfe_to_match.csv")
