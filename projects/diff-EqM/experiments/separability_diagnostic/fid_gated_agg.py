"""Stage 7b: aggregate per-rank Inception-feature shards from the gated sampler
and compute 50k-scale FID per arm vs the trusted reference stats.

Arms: vanilla (neg control / baseline — must reproduce ~31.41), probe (treatment),
oracle (pos control / ceiling). Reports FID + the probe's recovered fraction of the
oracle gain, with the vanilla->baseline sanity check.
"""
import argparse
import glob
from pathlib import Path

import numpy as np


def fid(mu1, c1, mu2, c2):
    from scipy import linalg
    d = mu1 - mu2
    cm, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(d @ d + np.trace(c1 + c2 - 2 * cm))


def main(args):
    out = Path(args.out)
    ref = np.load(args.ref_stats)
    mu_r, c_r = ref["mu"], ref["sigma"]
    lines = ["EqM PROBE-GATED SAMPLER — 50k FID (vs %s)" % Path(args.ref_stats).name,
             "=" * 56]
    fids = {}
    for arm in ["vanilla", "probe", "oracle"]:
        shards = sorted(glob.glob(str(out / f"feat_{arm}_rank*.npy")))
        feats = np.concatenate([np.load(s) for s in shards], 0) if shards else np.zeros((0, 2048))
        mu, c = feats.mean(0), np.cov(feats, rowvar=False)
        fids[arm] = fid(mu, c, mu_r, c_r)
        lines.append(f"  {arm:8s} n={len(feats):6d}  FID={fids[arm]:.3f}")
    print("\n".join(lines), flush=True)

    base = 31.41
    sane = abs(fids["vanilla"] - base) < args.baseline_tol
    band = fids["vanilla"] - fids["oracle"]
    recov = (fids["vanilla"] - fids["probe"]) / band if band > 0 else float("nan")
    lines.append("")
    lines.append(f"sanity: vanilla {fids['vanilla']:.2f} vs known baseline {base} -> "
                 f"{'OK' if sane else 'MISMATCH (pipeline/ref differ — interpret deltas only)'}")
    if band <= 0:
        v = "INCONCLUSIVE: oracle did not beat vanilla — restart/ref too weak at this scale."
    elif fids["probe"] < fids["vanilla"] - args.min_gain:
        v = (f"WORKS @50k: probe-gated FID {fids['probe']:.2f} < vanilla {fids['vanilla']:.2f} "
             f"(Δ {fids['vanilla']-fids['probe']:.2f}), recovering {100*recov:.0f}% of the oracle "
             f"gain (oracle {fids['oracle']:.2f}). In-line restart with the trajectory-shape probe "
             f"improves generation at scale.")
    else:
        v = (f"NULL @50k: probe {fids['probe']:.2f} ~= vanilla {fids['vanilla']:.2f} "
             f"(oracle {fids['oracle']:.2f}). Best-of-R probe selection did not move 50k FID.")
    lines += ["", "VERDICT: " + v]
    print("VERDICT: " + v, flush=True)
    (out / "GATED_FID.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dir with feat_<arm>_rank*.npy")
    ap.add_argument("--ref-stats", required=True, help="in1k_reference_stats.npz (mu,sigma)")
    ap.add_argument("--baseline-tol", type=float, default=3.0)
    ap.add_argument("--min-gain", type=float, default=0.3)
    args = ap.parse_args()
    main(args)
