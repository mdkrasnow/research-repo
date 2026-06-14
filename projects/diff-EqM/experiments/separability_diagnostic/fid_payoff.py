"""Stage 6: payoff test — does the probe's garbage-detection actually IMPROVE
generation, with positive + negative controls?

Reuses the existing 3000-sample run (trajectories + PNGs); NO new generation.
Rank all samples by probe P(garbage), then compute FID of the kept subset under
three arms (all size K = keep_frac * N, same ref, same K -> fair):

  - probe-keep   : drop the highest-P(garbage)        (the treatment)
  - random-keep  : drop a random subset               (NEGATIVE control / floor)
  - oracle-keep  : drop the highest inception-NN dist  (POSITIVE control / ceiling)
  - full         : all N (reference point)

Mechanism works iff  FID_probe < FID_random  and approaches FID_oracle. If
probe ~= random, the probe adds nothing actionable to generation.

GPU for inception features; FID via Frechet distance (numpy/scipy).
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np


def load_traj(folder):
    shards = sorted(glob.glob(str(Path(folder) / "logs" / "traj_rank*.npz")))
    P = {k: [] for k in ["sample_id", "norm", "dot"]}
    for s in shards:
        d = np.load(s)
        if d["sample_id"].shape[0] == 0:
            continue
        for k in P:
            P[k].append(d[k])
    return {k: np.concatenate(v, 0) for k, v in P.items()}


def probe_scores(folder, norm, dot):
    """P(garbage) per sample from the saved probe artifact + trajectory shape."""
    from probe_validate import feature_groups
    art = np.load(Path(folder) / "probe_artifact.npz", allow_pickle=True)
    w, b, mu, sd = art["w"], float(art["b"]), art["mu"], art["sd"]
    X = feature_groups(norm, dot)["ALL-shape"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    z = ((X - mu) / sd) @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fid(mu1, c1, mu2, c2):
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(c1 + c2 - 2 * covmean))


def stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def main(args):
    folder = Path(args.folder)
    for p in (str(Path(__file__).resolve().parent),
              str(Path(__file__).resolve().parent.parent / "exp3_fidelity_diversity")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from features import inception_features

    # --- trajectories -> probe scores ---
    tj = load_traj(folder)
    ids = tj["sample_id"]
    pg = probe_scores(folder, tj["norm"], tj["dot"])
    order_by_id = {int(i): k for k, i in enumerate(ids)}

    # --- gen inception features (aligned to ids by filename stem) ---
    gen_files = sorted([f for f in folder.iterdir() if f.suffix == ".png"])
    gfeat, gstems = inception_features(str(folder), device="cuda", batch_size=args.batch_size, files=gen_files)
    gstem_ids = np.array([int(s) for s in gstems])
    # align probe score + label nn-dist to gen feature rows
    pg_aligned = np.array([pg[order_by_id[i]] for i in gstem_ids])

    # --- oracle ranking: inception-NN dist to real (same metric as labels) ---
    from compute_quality_labels import list_real_images, knn_dist
    real_files = list_real_images(args.real_dir, args.num_real, seed=0)
    rfeat, _ = inception_features([], device="cuda", batch_size=args.batch_size, files=real_files)
    nn = knn_dist(gfeat, rfeat, k=3)               # higher = more garbage (oracle)
    mu_r, c_r = stats(rfeat)

    N = len(gfeat)
    probe_rank = np.argsort(pg_aligned)      # ascending P(garbage): keep prefix
    oracle_rank = np.argsort(nn)             # ascending inception dist: keep prefix
    fracs = [float(x) for x in args.keep_fracs]
    lines = ["EqM PROBE — FID payoff sweep (N=%d, real_ref=%d)" % (N, len(real_files)),
             "=" * 64,
             "keep   probe   random(mean±sd, 5 seeds)   oracle   recovered%"]
    print("\n".join(lines), flush=True)
    recov_all = []
    for kf in fracs:
        K = int(N * kf)
        f_probe = fid(*stats(gfeat[probe_rank[:K]]), mu_r, c_r)
        f_oracle = fid(*stats(gfeat[oracle_rank[:K]]), mu_r, c_r)
        rnds = []
        for s in range(5):
            sel = np.random.default_rng(s).permutation(N)[:K]
            rnds.append(fid(*stats(gfeat[sel]), mu_r, c_r))
        rnds = np.array(rnds); rmean, rsd = rnds.mean(), rnds.std()
        band = rmean - f_oracle
        recov = (rmean - f_probe) / band if band > 0 else float("nan")
        recov_all.append(recov)
        line = (f"{kf:.2f}  {f_probe:7.2f}  {rmean:7.2f}±{rsd:.2f}            "
                f"{f_oracle:7.2f}   {100*recov:5.0f}%")
        lines.append(line); print(line, flush=True)

    f_full = fid(*stats(gfeat), mu_r, c_r)
    lines.append(f"\nfull (N={N}) FID={f_full:.2f}")
    mean_recov = float(np.nanmean(recov_all))
    beats = all((np.array(recov_all) > 0))
    if beats and mean_recov > 0.15:
        v = (f"WORKS: across keep_fracs {fracs}, probe-keep beats the random floor at every "
             f"point, recovering {100*mean_recov:.0f}% of the oracle gain on average. The "
             f"trajectory-shape probe (no image access) actionably improves generation.")
    elif beats:
        v = (f"WEAK-WORKS: probe beats random at every frac but recovers only "
             f"{100*mean_recov:.0f}% of oracle on average.")
    else:
        v = "NULL: probe does not consistently beat the random floor across keep_fracs."
    lines += ["", "VERDICT: " + v]
    print("VERDICT: " + v, flush=True)
    (folder / "results").mkdir(exist_ok=True, parents=True)
    (folder / "results" / "FID_PAYOFF.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--real-dir", default="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train")
    ap.add_argument("--num-real", type=int, default=10000)
    ap.add_argument("--keep-fracs", type=float, nargs="+", default=[0.5,0.6,0.7,0.8,0.9])
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    main(args)
