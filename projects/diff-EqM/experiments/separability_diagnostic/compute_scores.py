"""Stage 3 of the EqM Separability Diagnostic.

Turn each sample's logged per-step trajectory (Stage 1 shards) into the 5
candidate scalars, evaluated at the 'stopping point' under two regimes.

Stopping point k* (T = num_sampling_steps - 1 logged steps):
  (A) threshold : first k with norm[k] < TAU_NORM, else last step (T-1).
  (B) fixed     : k* = T-1 for every sample  -> the DE-CONFOUNDING regime
                  (all samples at equal step count; separation can't be a
                  by-product of the stopping rule). Sweep TAU in {5,10,20} for A.

Candidate scores at k*:
  s1 = -dot[k*]                          # 'dot' energy proxy  (raw field <f,x>)
  s2 =  l2[k*]                           # 'l2' energy proxy  = 0.5||f||^2 (norm-coupled)
  s3 =  sum_{j<=k*} step_dot[j]          # path integral of f along the path taken
  s4 =  latent-space NN dist(x_final, real_latents)   # dumb baseline, NO f -> label sanity
  s5 =  norm[min(k*+1, T-1)]             # field magnitude after one more GD step (norm-coupled)

Note s5 needs NO extra model eval: x_{k*+1} = x_{k*} + eta*f(x_{k*}) is exactly
the next logged point, so ||f(x_{k*+1})|| == norm[k*+1].

s2 and s5 are deterministic functions of the gradient norm -> they exist
precisely so Stage 4's matched-norm control can show whether ANY separation is
just norm-in-disguise. Only s1 and s3 can carry norm-independent information.
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def load_shards(log_dir):
    shards = sorted(glob.glob(str(Path(log_dir) / "traj_rank*.npz")))
    assert shards, f"no traj_rank*.npz in {log_dir}"
    parts = {k: [] for k in ["sample_id", "label", "x_final", "norm", "dot", "l2", "step_dot"]}
    stepsize = None
    for s in shards:
        d = np.load(s)
        if d["sample_id"].shape[0] == 0:
            continue
        for k in parts:
            parts[k].append(d[k])
        stepsize = float(d["stepsize"])
    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    # sort by sample_id for determinism
    order = np.argsort(out["sample_id"])
    for k in out:
        out[k] = out[k][order]
    out["stepsize"] = stepsize
    return out


def latent_nn(x_final, real_latents):
    """Min L2 distance from each gen latent to the real latent bank (flattened)."""
    import torch
    if real_latents is None or len(real_latents) == 0:
        return np.full(len(x_final), np.nan)
    g = torch.tensor(x_final.reshape(len(x_final), -1), dtype=torch.float32)
    r = torch.tensor(real_latents.reshape(len(real_latents), -1), dtype=torch.float32)
    out = np.zeros(len(g), dtype=np.float64)
    chunk = 256
    for i in range(0, len(g), chunk):
        d = torch.cdist(g[i:i + chunk], r)
        out[i:i + chunk] = d.min(dim=1).values.numpy()
    return out


def kstar_threshold(norm_row, tau):
    """First index where norm < tau, else last index."""
    below = np.nonzero(norm_row < tau)[0]
    return int(below[0]) if below.size else int(len(norm_row) - 1)


def dynamics_feats(norm_row, k):
    """Trajectory-SHAPE features from the descent path actually taken (norm over
    steps 0..k). Distinct from endpoint scalars: tests whether the good/garbage
    signal lives in the DYNAMICS of convergence rather than the value at the stop.
      s6 = mean norm over the path        (AUC / total residual; magnitude -> norm-coupled)
      s7 = log-norm decay slope           (how FAST it converged; rate, not magnitude)
      s8 = norm oscillation               (fraction of sign-flips in step-to-step diff)
      s9 = late-stage slope               (slope over the final quarter; flat-at-high = spurious)
    """
    seg = norm_row[:k + 1]
    n = len(seg)
    if n < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    steps = np.arange(n, dtype=np.float64)
    s6 = float(seg.mean())
    s7 = float(np.polyfit(steps, np.log(seg + 1e-8), 1)[0])
    d = np.diff(seg)
    sign_flips = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1]))) if len(d) > 1 else 0
    s8 = float(sign_flips / max(1, len(d) - 1))
    q = max(2, n // 4)
    s9 = float(np.polyfit(steps[-q:], seg[-q:], 1)[0])
    return s6, s7, s8, s9


def compute_scores_for_regime(data, s4, regime, tau):
    norm, dot, l2, step_dot = data["norm"], data["dot"], data["l2"], data["step_dot"]
    N, T = norm.shape
    cum_step = np.cumsum(step_dot, axis=1)        # (N, T)
    rows = []
    for n in range(N):
        if regime == "fixed":
            k = T - 1
        else:
            k = kstar_threshold(norm[n], tau)
        k5 = min(k + 1, T - 1)
        s6, s7, s8, s9 = dynamics_feats(norm[n], k)
        rows.append({
            "sample_id": int(data["sample_id"][n]),
            "regime": regime,
            "tau_norm": (tau if regime == "threshold" else -1),
            "s1": float(-dot[n, k]),
            "s2": float(l2[n, k]),
            "s3": float(cum_step[n, k]),
            "s4": float(s4[n]),
            "s5": float(norm[n, k5]),
            "s6": s6, "s7": s7, "s8": s8, "s9": s9,
            "k_star": int(k),
            "norm_at_kstar": float(norm[n, k]),
        })
    return rows


def main(args):
    folder = Path(args.folder)
    data = load_shards(folder / "logs")
    print(f"[sep-diag/scores] loaded {len(data['sample_id'])} samples, "
          f"trajectory T={data['norm'].shape[1]} stepsize={data['stepsize']}", flush=True)

    real_lat_path = folder / "real_latents.npz"
    real_latents = np.load(real_lat_path)["latents"] if real_lat_path.exists() else None
    s4 = latent_nn(data["x_final"], real_latents)
    if real_latents is None:
        print("[sep-diag/scores] WARNING: no real_latents.npz -> s4 = NaN", flush=True)

    rows = []
    for tau in args.tau_sweep:
        rows += compute_scores_for_regime(data, s4, "threshold", tau)
    rows += compute_scores_for_regime(data, s4, "fixed", -1)

    out_csv = folder / "scores.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "regime", "tau_norm",
                                           "s1", "s2", "s3", "s4", "s5",
                                           "s6", "s7", "s8", "s9",
                                           "k_star", "norm_at_kstar"])
        w.writeheader()
        w.writerows(rows)
    print(f"[sep-diag/scores] wrote {len(rows)} rows -> {out_csv}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Stage 1 output dir")
    ap.add_argument("--tau-sweep", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    args = ap.parse_args()
    main(args)
