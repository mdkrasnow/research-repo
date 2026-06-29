"""Aggregate the metacog policy sweep: sum each arm's rank shards -> FID, read
exact NFE from meta.json, build a table + promotion verdict.

Promotion rule (matched-NFE screen): an arm is PROMOTABLE iff
  (a) |nfe_per_img - 750| / 750 < args.nfe_tol   (compute matched), AND
  (b) FID <= FID(old_r3probe50) - args.margin     (beats the locked selector).
Off-budget arms (segmented heun/etc.) are REPORTED with their true NFE, never
silently treated as matched.

Repro check: if --pareto-energy <dir> is given, recompute its FID with the same
math and compare to the sweep's energy_path arm (should match the original
pareto seed0 within rounding) — guards the incremental-FID pipeline.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import linalg


def fid(mu1, c1, mu2, c2):
    cov, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    d = mu1 - mu2
    return float(d @ d + np.trace(c1 + c2 - 2 * cov))


def arm_fid(d, mu_r, c_r):
    shards = sorted(glob.glob(str(Path(d) / "stats_rank*.npz")))
    if not shards:
        return None, 0
    s1 = np.zeros(2048); s2 = np.zeros((2048, 2048)); n = 0
    for s in shards:
        z = np.load(s); s1 += z["s1"]; s2 += z["s2"]; n += int(z["n"])
    if n < 100:
        return None, n
    mu = s1 / n; cov = (s2 - n * np.outer(mu, mu)) / (n - 1)
    return fid(mu, cov, mu_r, c_r), n


def arm_nfe(d):
    metas = glob.glob(str(Path(d) / "meta_rank*.json"))
    if not metas:
        return None
    vals = [json.load(open(m)) for m in metas]
    tot = sum(v["nfe_per_img"] * v["n"] for v in vals); n = sum(v["n"] for v in vals)
    return tot / max(1, n), vals[0].get("policy", "?"), vals[0].get("engine", "?")


def main(a):
    ref = np.load(a.ref_stats); mu_r, c_r = ref["mu"], ref["sigma"]
    root = Path(a.root)
    arms = sorted([p.name for p in root.iterdir() if p.is_dir() and (p / "meta_rank0.json").exists()
                   or glob.glob(str(p / "stats_rank*.npz"))]) if not a.arms else a.arms.split(",")
    rows = {}
    for arm in arms:
        d = root / arm
        f, n = arm_fid(d, mu_r, c_r)
        nf = arm_nfe(d)
        if f is None:
            continue
        nfe, pol, eng = (nf if nf else (float("nan"), arm, "?"))
        rows[arm] = {"fid": f, "n": n, "nfe": nfe, "policy": pol, "engine": eng}

    base_name = a.baseline
    base_fid = rows.get(base_name, {}).get("fid")
    L = [f"METACOG POLICY SWEEP — FID @ NFE (baseline = {base_name})", "=" * 64,
         f"{'arm':22s} {'engine':10s} {'FID':>8s} {'nfe/img':>9s} {'n':>7s}  verdict"]
    for arm, r in sorted(rows.items(), key=lambda kv: kv[1]["fid"]):
        matched = abs(r["nfe"] - a.target) / a.target < a.nfe_tol
        verdict = ""
        if base_fid is not None and arm != base_name:
            d_fid = base_fid - r["fid"]
            if not matched:
                verdict = f"OFF-BUDGET (Δ{d_fid:+.2f} but nfe {r['nfe']:.0f}≠{a.target:.0f})"
            elif d_fid >= a.margin:
                verdict = f"PROMOTE (Δ{d_fid:+.2f} ≥ {a.margin})"
            else:
                verdict = f"no (Δ{d_fid:+.2f} < {a.margin})"
        L.append(f"{arm:22s} {r['engine']:10s} {r['fid']:8.3f} {r['nfe']:9.1f} {r['n']:7d}  {verdict}")

    if a.pareto_energy:
        pe, pn = arm_fid(a.pareto_energy, mu_r, c_r)
        se = rows.get("energy_path", {}).get("fid")
        if pe is not None and se is not None:
            L += ["", f"REPRO check: pareto energy_path FID {pe:.3f} (n={pn}) vs sweep energy_path "
                  f"{se:.3f} -> Δ {abs(pe-se):.3f} ({'OK' if abs(pe-se) < 0.5 else 'MISMATCH'})"]

    out = root / "POLICY_SWEEP_RESULTS.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref-stats", default="projects/diff-EqM/results/in1k_reference_stats.npz")
    ap.add_argument("--baseline", default="probe_k50")
    ap.add_argument("--arms", default="")
    ap.add_argument("--target", type=float, default=750.0)
    ap.add_argument("--nfe-tol", type=float, default=0.02)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--pareto-energy", default="")
    main(ap.parse_args())
