"""Aggregate online-adaptive-sampler feature shards -> per-arm FID + equal-NFE verdict.

Arms: vanilla (un-adapted floor), random-restart (NEG, compute-matched),
probe-restart (TREATMENT), oracle-restart (POS ceiling). The load-bearing
comparison is probe-restart vs random-restart at IDENTICAL NFE.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ARMS = ["vanilla", "random_restart", "probe_restart", "oracle_restart"]


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
    fids, ns = {}, {}
    lines = [f"EqM ONLINE ADAPTIVE SAMPLER — FID (vs {Path(args.ref_stats).name})", "=" * 56]
    for arm in ARMS:
        shards = sorted(glob.glob(str(out / f"feat_{arm}_rank*.npy")))
        feats = np.concatenate([np.load(s) for s in shards], 0) if shards else np.zeros((0, 2048))
        if len(feats) < 2:
            fids[arm] = float("nan"); ns[arm] = len(feats); continue
        mu, c = feats.mean(0), np.cov(feats, rowvar=False)
        fids[arm] = fid(mu, c, mu_r, c_r); ns[arm] = len(feats)
        lines.append(f"  {arm:16s} n={len(feats):6d}  FID={fids[arm]:.3f}")

    base = 31.41
    sane = np.isfinite(fids["vanilla"]) and abs(fids["vanilla"] - base) < args.baseline_tol
    pr = fids["probe_restart"]; rr = fids["random_restart"]; orc = fids["oracle_restart"]
    band = rr - orc
    recov = (rr - pr) / band if (np.isfinite(band) and band > 0) else float("nan")
    lines += ["",
              f"sanity: vanilla {fids['vanilla']:.2f} vs baseline {base} -> "
              f"{'OK' if sane else 'MISMATCH (interpret deltas only)'}",
              f"equal-NFE comparison: probe-restart {pr:.2f} vs random-restart {rr:.2f} "
              f"(Δ {rr - pr:+.2f}); oracle-restart {orc:.2f}"]
    if not np.isfinite(pr) or not np.isfinite(rr):
        v = "INCOMPLETE: missing arm shards."
    elif band <= 0:
        v = "INCONCLUSIVE: oracle did not beat random — restart lever too weak at this scale."
    elif pr < rr - args.min_gain:
        v = (f"WORKS: probe-restart {pr:.2f} < random-restart {rr:.2f} at EQUAL NFE "
             f"(Δ {rr - pr:.2f}), recovering {100 * recov:.0f}% of the oracle gain. The early "
             f"trajectory-risk score actionably reallocates compute — true online metacognition.")
    elif abs(pr - rr) <= args.min_gain:
        v = (f"NULL: probe-restart {pr:.2f} ≈ random-restart {rr:.2f} at equal NFE. The gain is "
             f"compute, not the probe; revert to post-hoc best-of-R.")
    else:
        v = f"NEGATIVE: probe-restart {pr:.2f} WORSE than random {rr:.2f}; risk score anti-correlated at scale."
    lines += ["", "VERDICT: " + v]
    print("\n".join(lines), flush=True)
    (out / "ONLINE_FID.txt").write_text("\n".join(lines) + "\n")
    (out / "online_fid.json").write_text(json.dumps(
        {"fids": {k: (None if not np.isfinite(fids[k]) else round(fids[k], 3)) for k in ARMS},
         "n": ns, "probe_minus_random": (None if not np.isfinite(rr - pr) else round(rr - pr, 3)),
         "recovered_fraction": (None if not np.isfinite(recov) else round(recov, 3)),
         "verdict": v}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-stats", required=True)
    ap.add_argument("--baseline-tol", type=float, default=3.0)
    ap.add_argument("--min-gain", type=float, default=0.3)
    args = ap.parse_args()
    main(args)
