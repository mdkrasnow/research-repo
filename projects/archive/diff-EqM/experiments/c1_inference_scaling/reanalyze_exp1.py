#!/usr/bin/env python3
"""
C1 minimal test (ZERO compute): re-analyze the existing Exp 1 80-cell sampler
sweep through the inference-compute-scaling lens.

Question (pre-registered, c1-inference-compute-scaling-proposal.md):
  Does ANM keep improving as we spend more optimization (NFE / larger steps)
  where vanilla plateaus then turns up (overshoot)?

Input : documentation/exp1_data/results.csv  (cols: checkpoint_type in {vanilla,anm},
        sampler in {gd,ngd}, nfe in {10,25,50,100,250}, step_mult in {0.5,1,1.5,2}, fid, ...)
Output: prints tables + a verdict; pure pandas/numpy, safe to rerun.

Note: this data has only 2 arms (vanilla, anm=v10 lambda=0.1) and caps nfe=250,
step_mult=2.0. Dose-ordering (lambda=0.3) and clean high-NFE turn-up require the
GPU extension run. This script measures only what the existing CSVs can answer.
"""
import numpy as np
import pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "documentation/exp1_data/results.csv"
# under-converged cells: huge FID where neither arm has reached the data manifold.
# A cell is "converged" if BOTH arms' FID < this (keeps the comparison meaningful).
CONVERGED_FID = 100.0


def load():
    d = pd.read_csv(CSV)
    return d[["checkpoint_type", "sampler", "nfe", "step_mult", "fid",
              "divergence_count", "nan_count"]].copy()


def converged_delta(d):
    """anm - vanilla FID on cells where both arms converged (neg = anm better)."""
    rows = []
    for samp in ["gd", "ngd"]:
        for nfe in sorted(d.nfe.unique()):
            for sm in sorted(d.step_mult.unique()):
                cell = d[(d.sampler == samp) & (d.nfe == nfe) & (d.step_mult == sm)]
                v = cell[cell.checkpoint_type == "vanilla"].fid
                a = cell[cell.checkpoint_type == "anm"].fid
                if v.empty or a.empty:
                    continue
                v, a = float(v.iloc[0]), float(a.iloc[0])
                if max(v, a) >= CONVERGED_FID:
                    continue
                rows.append({"sampler": samp, "nfe": nfe, "step_mult": sm,
                             "vanilla": v, "anm": a, "delta": a - v,
                             "eff_compute": nfe * sm})
    return pd.DataFrame(rows)


def turn_up(d):
    """For each (arm, sampler) sweep the step_mult overshoot axis at the most-
    converged nfe and report whether FID rises after its minimum (= overshoot
    degradation). Returns min-FID, FID at max step_mult, and the rise."""
    rows = []
    for samp in ["gd", "ngd"]:
        for arm in ["vanilla", "anm"]:
            sub = d[(d.sampler == samp) & (d.checkpoint_type == arm)]
            # use nfe=100 (and 250): the converged band where step_mult sweeps fully
            for nfe in [100, 250]:
                s = sub[sub.nfe == nfe].sort_values("step_mult")
                s = s[s.fid < CONVERGED_FID]
                if len(s) < 2:
                    continue
                fmin = s.fid.min()
                sm_min = float(s.loc[s.fid.idxmin(), "step_mult"])
                f_hi = float(s.loc[s.step_mult.idxmax(), "fid"])
                sm_hi = float(s.step_mult.max())
                rows.append({"sampler": samp, "arm": arm, "nfe": nfe,
                             "fid_min": round(fmin, 2), "sm_at_min": sm_min,
                             "fid_at_max_sm": round(f_hi, 2), "max_sm": sm_hi,
                             "rise_after_min": round(f_hi - fmin, 2)})
    return pd.DataFrame(rows)


def main():
    d = load()
    print(f"loaded {len(d)} cells from {CSV.name}\n")

    dd = converged_delta(d)
    print("=== Converged-cell delta (anm - vanilla); neg = ANM better ===")
    print(dd.sort_values(["sampler", "eff_compute"]).to_string(index=False))
    print(f"\nconverged cells: {len(dd)}  |  ANM-better: {(dd.delta < 0).sum()}"
          f"  vanilla-better: {(dd.delta > 0).sum()}")
    print(f"mean delta (converged): {dd.delta.mean():.3f}  "
          f"median: {dd.delta.median():.3f}")

    # does ANM advantage GROW with step aggressiveness? corr(delta, step_mult)
    big = dd[dd.nfe >= 100]  # restrict to clearly-converged band
    if len(big) > 2:
        c = np.corrcoef(big.step_mult, big.delta)[0, 1]
        print(f"\ncorr(step_mult, delta) on nfe>=100 cells: {c:+.3f}  "
              f"(neg = ANM edge grows as steps get more aggressive)")

    print("\n=== Overshoot turn-up: FID rise after its min along step_mult axis ===")
    tu = turn_up(d)
    print(tu.to_string(index=False))

    # verdict logic
    van = tu[tu.arm == "vanilla"].rise_after_min
    anm = tu[tu.arm == "anm"].rise_after_min
    print("\n=== VERDICT (existing-data, partial) ===")
    print(f"vanilla mean rise-after-min: {van.mean():+.2f} FID")
    print(f"anm     mean rise-after-min: {anm.mean():+.2f} FID")
    anm_wins_converged = (dd[dd.nfe >= 100].delta < 0).mean()
    print(f"ANM wins {anm_wins_converged*100:.0f}% of converged nfe>=100 cells")
    print("Signature present if: vanilla rises more than ANM after min AND "
          "ANM edge grows with step aggressiveness (neg corr).")


if __name__ == "__main__":
    main()
