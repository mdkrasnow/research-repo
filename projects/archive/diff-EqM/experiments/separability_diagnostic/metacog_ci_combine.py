"""5-seed CI + paired deltas for the metacog PROMOTION run (metacog_promote/).
Reads <root>/<arm>_s<seed>/stats_rank*.npz -> per-arm mean±std FID + paired
delta vs --baseline (default probe_k50) on common seeds (t-stat, per-seed signs).
Also reports measured NFE/img per arm (from meta) so matched-compute is auditable.
Disk-safe stats only. Mirror of ci_combine.py for the promotion layout.
"""
import argparse
import glob
import json
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


def nfe_of(d):
    m = glob.glob(str(Path(d) / "meta_rank*.json"))
    if not m:
        return float("nan")
    v = [json.load(open(x)) for x in m]
    return sum(x["nfe_per_img"] * x["n"] for x in v) / max(1, sum(x["n"] for x in v))


def main(a):
    ref = np.load(a.ref_stats); mu_r, c_r = ref["mu"], ref["sigma"]
    root = Path(a.root)
    arms = sorted({p.name.rsplit("_s", 1)[0] for p in root.iterdir()
                   if p.is_dir() and "_s" in p.name})
    per = {}; nfe = {}
    for arm in arms:
        vals = {}
        for s in range(0, 9):
            f = seed_fid(root / f"{arm}_s{s}", mu_r, c_r)
            if f is not None:
                vals[s] = f
        if vals:
            per[arm] = vals
            nfe[arm] = nfe_of(root / f"{arm}_s{min(vals)}")

    L = ["METACOG PROMOTION — 5-seed CI + paired deltas", "=" * 60,
         f"  {'arm':20s} {'n':>2s}  {'FID mean±std':>16s}  {'nfe/img':>8s}  seeds"]
    stats = {}
    for arm in sorted(per, key=lambda k: np.mean(list(per[k].values()))):
        v = list(per[arm].values()); m, sd = float(np.mean(v)), float(np.std(v))
        stats[arm] = per[arm]
        L.append(f"  {arm:20s} {len(v):2d}  {m:8.3f} ± {sd:.3f}  {nfe[arm]:8.1f}  "
                 + " ".join(f"s{k}:{val:.2f}" for k, val in sorted(per[arm].items())))

    base = a.baseline
    if base in stats:
        L += ["", f"PAIRED deltas vs {base} (same seeds; +ve = arm better):"]
        for arm in stats:
            if arm == base:
                continue
            common = sorted(set(stats[arm]) & set(stats[base]))
            if len(common) < 2:
                continue
            dl = [stats[base][s] - stats[arm][s] for s in common]
            dm, dsd = float(np.mean(dl)), float(np.std(dl))
            se = dsd / np.sqrt(len(dl)); t = dm / se if se else float("nan")
            signs = " ".join(f"s{s}:{stats[base][s]-stats[arm][s]:+.2f}" for s in common)
            L.append(f"  {arm:20s} Δ {dm:+.3f} ± {dsd:.3f}  SE {se:.3f}  t≈{t:.2f}  [{signs}]")
    out = root / "METACOG_PROMOTE_CI.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref-stats", default="projects/diff-EqM/results/in1k_reference_stats.npz")
    ap.add_argument("--baseline", default="probe_k50")
    main(ap.parse_args())
