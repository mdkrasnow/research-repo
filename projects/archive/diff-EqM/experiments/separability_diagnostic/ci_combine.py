"""5-seed CI on the Pareto headline (PI item 1). Reads incremental-FID stat shards from
ci/<arm>_s<seed>/ (+ the original pareto_<arm>/ run as seed0) -> per-arm mean±std FID +
the probe-vs-random paired delta. Disk-safe stats (s1/s2/n), no per-sample feats.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from scipy import linalg


def fid(mu1, c1, mu2, c2):
    diff = mu1 - mu2
    cov, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(diff @ diff + np.trace(c1 + c2 - 2 * cov))


def seed_fid(d, mu_r, c_r):
    shards = sorted(glob.glob(str(Path(d) / "stats_rank*.npz")))
    if not shards:
        return None
    s1 = np.zeros(2048); s2 = np.zeros((2048, 2048)); n = 0
    for s in shards:
        z = np.load(s); s1 += z["s1"]; s2 += z["s2"]; n += int(z["n"])
    if n < 100:
        return None
    mu = s1 / n; cov = (s2 - n * np.outer(mu, mu)) / (n - 1)
    return fid(mu, cov, mu_r, c_r)


def main(args):
    ref = np.load(args.ref_stats); mu_r, c_r = ref["mu"], ref["sigma"]
    root = Path(args.root)
    ci = root / "ci"
    # arm -> {seed0 dir, ci dir prefix}
    arms = {"long250": "pareto_long250", "r3rand": "pareto_r3rand",
            "r3energy": "pareto_r3energy", "r3probe": "pareto_r3probe"}
    per = {}
    for arm, s0dir in arms.items():
        vals = {}
        f0 = seed_fid(root / s0dir, mu_r, c_r)
        if f0 is not None:
            vals[0] = f0
        for s in range(1, 9):
            f = seed_fid(ci / f"{arm}_s{s}", mu_r, c_r)
            if f is not None:
                vals[s] = f
        per[arm] = vals

    L = ["5-SEED CI — Pareto headline (FID at equal NFE; seed0 from original pareto run)",
         "=" * 66, "  arm        n   FID mean ± std    seeds"]
    stats = {}
    for arm in ["long250", "r3rand", "r3energy", "r3probe"]:
        v = list(per[arm].values())
        if v:
            m, sd = float(np.mean(v)), float(np.std(v))
            stats[arm] = (m, sd, v, per[arm])
            L.append(f"  {arm:9s} {len(v):2d}   {m:6.3f} ± {sd:.3f}   "
                     + " ".join(f"s{k}:{val:.2f}" for k, val in sorted(per[arm].items())))
    # paired probe vs random delta (same seeds)
    if "r3probe" in stats and "r3rand" in stats:
        common = sorted(set(per["r3probe"]) & set(per["r3rand"]))
        deltas = [per["r3rand"][s] - per["r3probe"][s] for s in common]  # +ve = probe better
        if deltas:
            dm, dsd = float(np.mean(deltas)), float(np.std(deltas))
            n = len(deltas)
            se = dsd / max(1e-9, np.sqrt(n)) if n > 1 else float("nan")
            t = dm / se if se and not np.isnan(se) else float("nan")
            L += ["", f"PAIRED probe vs random (n={n} seeds): mean Δ = {dm:+.3f} ± {dsd:.3f} FID "
                  f"(probe better when +); SE {se:.3f}, t≈{t:.2f}",
                  f"  per-seed Δ: " + " ".join(f"s{s}:{per['r3rand'][s]-per['r3probe'][s]:+.2f}" for s in common)]
    # probe vs depth + vs energy
    for ref_arm in ["long250", "r3energy"]:
        if "r3probe" in stats and ref_arm in stats:
            L.append(f"  probe {stats['r3probe'][0]:.2f} vs {ref_arm} {stats[ref_arm][0]:.2f}  "
                     f"-> Δ {stats[ref_arm][0]-stats['r3probe'][0]:+.2f} FID (probe better when +)")
    out = root / "CI_5SEED.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="projects/diff-EqM/experiments/separability_diagnostic/runs/b2_vanilla")
    ap.add_argument("--ref-stats", default="projects/diff-EqM/results/in1k_reference_stats.npz")
    main(ap.parse_args())
