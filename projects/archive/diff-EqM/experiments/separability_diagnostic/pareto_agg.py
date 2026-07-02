"""Sum pareto_sample s1/s2/n rank shards -> mu/cov -> FID. Tiny (no per-sample feats)."""
import argparse, glob, json
from pathlib import Path
import numpy as np
from scipy import linalg

def fid(mu1, c1, mu2, c2):
    cov, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(cov): cov = cov.real
    d = mu1 - mu2
    return float(d @ d + np.trace(c1 + c2 - 2 * cov))

def main(a):
    out = Path(a.out)
    s1 = np.zeros(2048); s2 = np.zeros((2048, 2048)); n = 0
    for s in glob.glob(str(out / "stats_rank*.npz")):
        d = np.load(s); s1 += d["s1"]; s2 += d["s2"]; n += int(d["n"])
    mu = s1 / n; cov = (s2 - n * np.outer(mu, mu)) / (n - 1)
    ref = np.load(a.ref_stats)
    f = fid(mu, cov, ref["mu"], ref["sigma"])
    (out / "FID.txt").write_text(f"FID={f:.4f}  n={n}\n")
    print(json.dumps({"out": str(out), "fid": round(f, 4), "n": n}), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-stats", default="projects/diff-EqM/results/in1k_reference_stats.npz")
    main(ap.parse_args())
