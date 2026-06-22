"""Aggregate selector_compare feature shards -> per-selector FID + the money table.

The headline figure: FID at EQUAL COMPUTE across restart selectors. energy_* / gradnorm_*
are collapsed to their best (lower-FID) direction (steel-man the trivial baselines). Also
emits the early-intervention curve (probe@k FID vs step k) from the probe_k* arms.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


def fid(mu1, c1, mu2, c2):
    from scipy import linalg
    diff = mu1 - mu2
    cov, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(diff @ diff + np.trace(c1 + c2 - 2 * cov))


def arm_fid(out, arm, mu_r, c_r):
    shards = sorted(glob.glob(str(out / f"feat_{arm}_rank*.npy")))
    if not shards:
        return None, 0
    feats = np.concatenate([np.load(s) for s in shards], 0)
    if len(feats) < 2:
        return None, len(feats)
    return fid(feats.mean(0), np.cov(feats, rowvar=False), mu_r, c_r), len(feats)


def main(args):
    out = Path(args.out)
    ref = np.load(args.ref_stats); mu_r, c_r = ref["mu"], ref["sigma"]
    all_arms = [re.search(r"feat_(.+)_rank", Path(s).name).group(1)
                for s in glob.glob(str(out / "feat_*_rank0.npy"))]
    fids = {}
    for a in sorted(set(all_arms)):
        f, n = arm_fid(out, a, mu_r, c_r)
        if f is not None:
            fids[a] = (f, n)

    # collapse steel-manned trivial selectors to best direction
    def best(*names):
        cand = [(fids[n][0], n) for n in names if n in fids]
        return min(cand) if cand else (None, None)
    energy_dot = best("energy_dot_lo", "energy_dot_hi")
    energy_path = best("energy_path_lo", "energy_path_hi")
    gradnorm = best("gradnorm_lo", "gradnorm_hi")
    energy = min([x for x in [energy_dot, energy_path] if x[0] is not None], default=(None, None))

    van = fids.get("vanilla", (None,))[0]
    table = {"vanilla": van, "random": fids.get("random", (None,))[0],
             "energy_dot": energy_dot[0], "energy_path": energy_path[0], "energy_best": energy[0],
             "gradnorm": gradnorm[0], "probe": fids.get("probe", (None,))[0],
             "oracle": fids.get("oracle", (None,))[0]}

    # early-intervention curve
    early = {}
    for a, (f, n) in fids.items():
        m = re.match(r"probe_k(\d+)", a)
        if m:
            early[int(m.group(1))] = round(f, 3)
    early = dict(sorted(early.items()))

    nfe = (out / "nfe.txt").read_text() if (out / "nfe.txt").exists() else "n/a"
    lines = ["EqM RESTART-SELECTOR COMPARISON — FID at EQUAL COMPUTE", "=" * 56, nfe.strip(), "",
             "  selector            FID      Δ vs vanilla"]
    order = ["vanilla", "random", "energy_dot", "energy_path", "gradnorm", "probe", "oracle"]
    for k in order:
        v = table.get(k)
        if v is not None:
            dv = f"{van - v:+.3f}" if van is not None else "n/a"
            lines.append(f"  {k:18s} {v:7.3f}   {dv}")
    # verdict
    trivial = [table[k] for k in ["random", "energy_best", "gradnorm"] if table.get(k) is not None]
    if table.get("probe") is not None and trivial:
        beats = table["probe"] < min(trivial) and (van is None or table["probe"] < van)
        lines += ["", f"VERDICT: trajectory-probe FID {table['probe']:.3f} vs best trivial "
                  f"{min(trivial):.3f} (random/energy/norm) vs vanilla {van:.3f} -> "
                  + ("PROBE WINS — clean at equal compute." if beats else "probe does NOT beat all trivial selectors.")]
    if early:
        lines += ["", "EARLY INTERVENTION (probe@step_k restart FID):"] + \
                 [f"  k={k:4d}  FID={v}" for k, v in early.items()]
    summary = {"nfe": nfe.strip(), "fids": {k: (round(v, 3) if v is not None else None) for k, v in table.items()},
               "early_intervention_fid_by_k": early, "all_arm_fids": {k: round(v[0], 3) for k, v in fids.items()}}
    (out / "SELECTOR_FID.txt").write_text("\n".join(lines))
    (out / "selector_fid.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-stats", default="projects/diff-EqM/results/in1k_reference_stats.npz")
    main(ap.parse_args())
