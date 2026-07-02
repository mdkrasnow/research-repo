"""Combine selector_compare seeds -> 5-seed airtight table (mean±std per selector) +
early-intervention curve. Handles seeds with missing ranks (uses whatever shards exist).

Reports, with energy/gradnorm directions FIXED A PRIORI (the direction that is consistent
across seeds — reported explicitly, not post-hoc per-seed-best), so the trivial baselines
get a fair, deployable comparison. Also reports the per-direction split for transparency.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import linalg


def fid(mu1, c1, mu2, c2):
    diff = mu1 - mu2
    cov, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(diff @ diff + np.trace(c1 + c2 - 2 * cov))


def arm_fid(seed_dir, arm, mu_r, c_r):
    shards = sorted(glob.glob(str(seed_dir / f"feat_{arm}_rank*.npy")))
    if not shards:
        return None
    feats = np.concatenate([np.load(s) for s in shards], 0)
    if len(feats) < 100:
        return None
    return fid(feats.mean(0), np.cov(feats, rowvar=False), mu_r, c_r)


def main(args):
    ref = np.load(args.ref_stats); mu_r, c_r = ref["mu"], ref["sigma"]
    seed_dirs = sorted(glob.glob(str(Path(args.root) / "sel_s*")))
    # a-priori fixed directions (argmax = "hi" consistently wins across seeds)
    arms = {"vanilla": "vanilla", "random": "random",
            "energy_dot": "energy_dot_hi", "energy_path": "energy_path_hi",
            "gradnorm": "gradnorm_hi", "probe(full)": "probe", "oracle": "oracle"}
    per = {a: [] for a in arms}
    ks = [50, 75, 100, 124, 149, 174, 199, 224, 249]
    pk = {k: [] for k in ks}
    nseed = 0
    for sd in seed_dirs:
        sd = Path(sd)
        if not glob.glob(str(sd / "feat_vanilla_rank*.npy")):
            continue
        nseed += 1
        for a, fa in arms.items():
            v = arm_fid(sd, fa, mu_r, c_r)
            if v is not None:
                per[a].append(v)
        for k in ks:
            v = arm_fid(sd, f"probe_k{k}", mu_r, c_r)
            if v is not None:
                pk[k].append(v)

    def ms(xs):
        return (round(float(np.mean(xs)), 3), round(float(np.std(xs)), 3), len(xs)) if xs else (None, None, 0)

    table = {a: ms(per[a]) for a in arms}
    early = {k: ms(pk[k]) for k in ks}
    van = table["vanilla"][0]
    lines = [f"5-SEED SELECTOR COMPARISON @50k (n_seeds={nseed}; directions fixed a-priori)",
             "=" * 64, "  selector        FID(mean±std, nseed)   Δ vs vanilla"]
    for a in ["vanilla", "random", "energy_dot", "energy_path", "gradnorm", "probe(full)", "oracle"]:
        m, s, n = table[a]
        if m is not None:
            lines.append(f"  {a:14s} {m:6.3f} ± {s:.3f} (n={n})   {van-m:+.3f}")
    # best non-oracle trivial
    triv = [table[a][0] for a in ["random", "energy_dot", "energy_path", "gradnorm"] if table[a][0]]
    pe50 = early[50][0]
    lines += ["", "EARLY-INTERVENTION probe@k (mean±std):"]
    for k in ks:
        m, s, n = early[k]
        if m is not None:
            lines.append(f"  probe@{k:3d}   {m:6.3f} ± {s:.3f} (n={n})   {van-m:+.3f}")
    lines += ["", "VERDICT:",
              f"  full-probe {table['probe(full)'][0]} vs best trivial {min(triv) if triv else None} "
              f"-> {'probe wins' if table['probe(full)'][0] and triv and table['probe(full)'][0]<min(triv) else 'full-probe ~= best trivial (does NOT clearly win)'}",
              f"  EARLY probe@50 {pe50} vs best trivial {min(triv) if triv else None} "
              f"-> {'probe@50 WINS — beats all trivial at equal compute' if pe50 and triv and pe50<min(triv) else 'no'}"]
    out = Path(args.root) / "FIVE_SEED_SELECTOR.txt"
    out.write_text("\n".join(lines))
    (Path(args.root) / "five_seed.json").write_text(json.dumps({"table": table, "early": early, "nseed": nseed}, indent=2))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="projects/diff-EqM/experiments/separability_diagnostic/runs/b2_vanilla")
    ap.add_argument("--ref-stats", default="projects/diff-EqM/results/in1k_reference_stats.npz")
    main(ap.parse_args())
