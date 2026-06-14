"""PART 5 — synthetic tests for the post-hoc ladder.

Builds 3 fake runs (cached logs + labels + scores only) and checks that
robustness_analysis.py and dynamics_probe.py classify them correctly:
  case A: endpoint scalar works           -> robustness GREEN*, dynamics GREEN
  case B: endpoint fails, dynamics works   -> robustness KILL/WEAK, dynamics GREEN/PROMISING
  case C: norm-collapse / no signal        -> robustness KILL, dynamics not-GREEN

Run: python test_posthoc.py   (CPU, ~seconds, no cluster).
"""
import csv
import shutil
import sys
import types
from pathlib import Path

import numpy as np

import robustness_analysis as RA
import dynamics_probe as DP

T = 60
N = 600


def _decay(n):
    base = np.linspace(40, 6, T)[None, :].repeat(n, 0)
    return base


def build(mode, root):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    (root / "logs").mkdir(parents=True)
    rng = np.random.default_rng(0)
    ids = np.arange(N)
    g = rng.uniform(0, 1, N)                 # latent badness; high => garbage
    norm = _decay(N) + rng.normal(0, 0.6, (N, T))
    dot = rng.normal(0, 1, (N, T))

    # per-mode SCORES (read by robustness from scores.csv) and LOG trajectories
    # (read by dynamics_probe from logs/*.npz) are deliberately decoupled.
    if mode == "A":      # endpoint scalar separates, independent of norm
        norm = norm + rng.normal(0, 3, (N, 1))             # norm overlaps (not tied to g)
        dot = dot + np.linspace(0, 1, T)[None] * (-(2.0 * g + rng.normal(0, 0.25, N)))[:, None]
        norm_end = norm[:, -1]
        s1 = -dot[:, -1]                                   # endpoint scalar tracks g
        s3 = 2.0 * g + rng.normal(0, 0.3, N)               # 2nd independent scalar tracks g
    elif mode == "B":    # endpoint SCALARS neutral; only the log curve SHAPE carries g
        # garbage curve oscillates: per-step flip rate ~ g, zero-mean, endpoint pinned
        s = np.ones((N, T))
        for i in ids:
            flips = rng.uniform(0, 1, T) < (0.05 + 0.9 * g[i])
            sgn = 1.0
            for t in range(T):
                if flips[t]:
                    sgn = -sgn
                s[i, t] = sgn
        s -= s.mean(1, keepdims=True)                      # zero-mean -> energy ~ matched
        s[:, -1] = 0.0                                     # pin endpoint
        norm = _decay(N) + 1.0 * s + rng.normal(0, 0.15, (N, T))
        norm_end = norm[:, -1]
        s1 = rng.normal(0, 1, N)                           # scalars carry NO g signal
        s3 = rng.normal(0, 1, N)
    else:                 # C: norm-collapse — score AND label both determined by norm
        norm = _decay(N) + (g[:, None] * 18.0) + rng.normal(0, 0.3, (N, T))
        norm_end = norm[:, -1]
        s1 = norm_end                                      # scores ARE the norm
        s3 = norm.mean(1)

    # labels: A/B from the true latent g (independent of norm); C from final norm
    # itself, so every norm-bin is single-class (genuine norm-collapse).
    if mode == "C":
        nn_dist = norm_end + rng.normal(0, 0.05, N)
    else:
        nn_dist = 10 + 8 * g + rng.normal(0, 0.4, N)

    l2 = 0.5 * norm ** 2
    step_dot = 0.05 * norm ** 2
    np.savez_compressed(root / "logs" / "traj_rank0.npz",
                        sample_id=ids.astype(np.int64), label=np.zeros(N, np.int64),
                        x0=np.zeros((N, 4, 8, 8), np.float32),
                        x_final=rng.normal(0, 1, (N, 4, 8, 8)).astype(np.float32),
                        norm=norm.astype(np.float32), dot=dot.astype(np.float32),
                        l2=l2.astype(np.float32), step_dot=step_dot.astype(np.float32),
                        stepsize=np.float32(0.05))

    with open(root / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["sample_id", "nn_dist", "label", "max_softmax"])
        lo, hi = np.quantile(nn_dist, 0.3), np.quantile(nn_dist, 0.7)
        for i in ids:
            lab = "good" if nn_dist[i] < lo else ("garbage" if nn_dist[i] > hi else "ambiguous")
            w.writerow([i, f"{nn_dist[i]:.4f}", lab, f"{1 - g[i]:.4f}"])

    s4 = g + rng.normal(0, 0.15, N)                        # latent-NN sanity (tracks label)
    with open(root / "scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "regime", "tau_norm", "s1", "s2", "s3", "s4", "s5",
                    "k_star", "norm_at_kstar"])
        for i in ids:
            s2 = 0.5 * norm_end[i] ** 2
            s5 = norm_end[i]
            w.writerow([i, "fixed", -1, f"{s1[i]:.5f}", f"{s2:.5f}", f"{s3[i]:.5f}",
                        f"{s4[i]:.5f}", f"{s5:.5f}", T - 1, f"{norm_end[i]:.5f}"])
    return root


def run_case(mode, tmp):
    root = build(mode, Path(tmp) / f"case_{mode}")
    ra_args = types.SimpleNamespace(folder=str(root), out="",
                                    label_quantiles="0.2,0.25,0.3,0.35", n_bins=5)
    dp_args = types.SimpleNamespace(folder=str(root), out="", n_folds=5, seed=0, n_bins=5, l2=1.0)
    rv = RA.main(ra_args)
    dv = DP.main(dp_args)
    return rv, dv


EXPECT = {
    "A": (lambda rv: rv.startswith("GREEN"), lambda dv: dv in ("GREEN", "PROMISING")),
    "B": (lambda rv: rv in ("KILL", "WEAK"), lambda dv: dv in ("GREEN", "PROMISING")),
    "C": (lambda rv: rv in ("KILL",), lambda dv: dv not in ("GREEN",)),
}


def main():
    tmp = "/tmp/sep_posthoc_test"
    ok = True
    for mode in ["A", "B", "C"]:
        rv, dv = run_case(mode, tmp)
        rce, dce = EXPECT[mode]
        rp, dp = rce(rv), dce(dv)
        ok = ok and rp and dp
        print(f"\n### case {mode}: robustness={rv} [{'PASS' if rp else 'FAIL'}]  "
              f"dynamics={dv} [{'PASS' if dp else 'FAIL'}]\n")
    print("=" * 50)
    print("ALL TESTS PASS" if ok else "SOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
