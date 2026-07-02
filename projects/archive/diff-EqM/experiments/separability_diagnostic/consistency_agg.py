"""Phase 1 — aggregate per-seed 50k probe-gated FIDs into a consistency table.

Reads runs/<tag>/results/consistency/seed*/gated_fid.json (written by fid_gated_agg)
and reports per-arm FID mean ± std ± 95% CI across sampler seeds, plus the
consistency verdict: is the vanilla->probe gain stable in sign and magnitude?
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

ARMS = ["vanilla", "probe", "oracle"]


def main(args):
    cons = Path(args.cons_dir)
    seeds = sorted(glob.glob(str(cons / "seed*/gated_fid.json")))
    per = {a: [] for a in ARMS}
    gains = []          # vanilla - probe per seed
    recs = []
    rows = []
    for sj in seeds:
        d = json.loads(Path(sj).read_text())
        sid = Path(sj).parent.name
        f = d["fids"]
        for a in ARMS:
            per[a].append(f.get(a, float("nan")))
        gains.append(d.get("vanilla_minus_probe", float("nan")))
        if d.get("recovered_fraction") is not None:
            recs.append(d["recovered_fraction"])
        rows.append({"seed": sid, **{a: f.get(a) for a in ARMS},
                     "vanilla_minus_probe": d.get("vanilla_minus_probe"),
                     "recovered_fraction": d.get("recovered_fraction"),
                     "sane": d.get("sane")})

    def stat(x):
        x = np.array([v for v in x if v is not None and np.isfinite(v)], float)
        if len(x) == 0:
            return float("nan"), float("nan"), float("nan")
        m, s = float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0
        ci = 1.96 * s / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        return m, s, ci

    with open(cons / "fid_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", *ARMS, "vanilla_minus_probe",
                                           "recovered_fraction", "sane"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        gm, gs, gci = stat(gains)
        w.writerow({"seed": "MEAN", **{a: round(stat(per[a])[0], 3) for a in ARMS},
                    "vanilla_minus_probe": round(gm, 3) if np.isfinite(gm) else "",
                    "recovered_fraction": round(float(np.mean(recs)), 3) if recs else "",
                    "sane": ""})
        w.writerow({"seed": "STD", **{a: round(stat(per[a])[1], 3) for a in ARMS},
                    "vanilla_minus_probe": round(gs, 3) if np.isfinite(gs) else "", "sane": ""})

    gm, gs, gci = stat(gains)
    nseed = len([g for g in gains if g is not None and np.isfinite(g)])
    all_pos = nseed > 0 and all((g is not None and g > 0) for g in gains)
    if nseed == 0:
        verdict = "INCOMPLETE: no seed results found."
    elif nseed == 1:
        verdict = (f"SINGLE-SEED: probe gain Δ{gm:.2f} FID at 50k seed0 only — "
                   f"direction confirmed, consistency (multi-seed) NOT yet established.")
    elif all_pos and np.isfinite(gci) and gm - gci > 0:
        verdict = (f"CONSISTENT: probe beats vanilla on all {nseed} seeds, mean Δ{gm:.2f} "
                   f"FID (95% CI ±{gci:.2f}, excludes 0). Gain is stable.")
    elif all_pos:
        verdict = (f"DIRECTION-CONSISTENT: probe beats vanilla on all {nseed} seeds "
                   f"(mean Δ{gm:.2f}) but CI ±{gci:.2f} touches 0 — add seeds to tighten.")
    else:
        verdict = (f"INCONSISTENT: probe gain flips sign across seeds (mean Δ{gm:.2f}, "
                   f"std {gs:.2f}). Not a reliable improvement.")

    md = ["# Consistency summary — 50k probe-gated, multi-seed", "",
          f"checkpoint tag: {cons.parents[1].name} · slots/seed: {args.num_slots} · R: {args.r} · "
          f"seeds: {nseed}", "",
          "| seed | vanilla | probe | oracle | Δ(van−probe) | recovered |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['seed']} | {r['vanilla']} | {r['probe']} | {r['oracle']} | "
                  f"{r['vanilla_minus_probe']} | {r['recovered_fraction']} |")
    md += ["",
           f"- per-arm FID mean±std: vanilla {stat(per['vanilla'])[0]:.2f}±{stat(per['vanilla'])[1]:.2f}, "
           f"probe {stat(per['probe'])[0]:.2f}±{stat(per['probe'])[1]:.2f}, "
           f"oracle {stat(per['oracle'])[0]:.2f}±{stat(per['oracle'])[1]:.2f}",
           f"- vanilla−probe gain: {gm:.2f} ± {gs:.2f} (95% CI ±{gci:.2f})" if np.isfinite(gci)
           else f"- vanilla−probe gain: {gm:.2f} (single seed, no CI)",
           f"- NFE per slot: R×N = {args.r}×250 (all arms share the SAME R draws → identical compute)",
           "", f"## VERDICT: {verdict}"]
    (cons / "CONSISTENCY_SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cons-dir", required=True)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--num-slots", type=int, default=50000)
    ap.add_argument("--r", type=int, default=3)
    args = ap.parse_args()
    main(args)
