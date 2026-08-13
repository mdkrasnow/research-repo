"""Parallel process-pool runner for the five-atom campaign.

Same experiment as `toy_campaign.py`, but the embarrassingly-parallel stages
(tc sweep, main comparison, FD grid) are distributed over worker processes.
Each worker is pinned to a small thread budget so N workers actually use N
cores rather than fighting over them.

Results are appended to a single JSONL under a lock; every record carries its
full config, so partial output from a killed run is still analyzable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from .toy5 import ToyConfig, run

ALL_ARMS = (
    "btm_vector",
    "btm_scalar_exact",
    "btm_scalar_action_exact",
    "btm_scalar_fd_directional",
    "btm_scalar_fd_action",
    "eqm_legacy_vector",
    "eqm_legacy_scalar",
)

_OUT = None


def _work(item):
    stage, cfg_kwargs = item
    cfg = ToyConfig(**cfg_kwargs)
    t0 = time.time()
    try:
        rec = run(cfg, verbose=False)
        rec["stage"] = stage
        rec["error"] = None
    except Exception as exc:                       # keep the campaign alive
        rec = {"stage": stage, "config": asdict(cfg), "error":
               f"{type(exc).__name__}: {exc}", "mass_mae": float("nan"),
               "stable": False}
    rec["wall_seconds"] = time.time() - t0
    rec.pop("history", None)
    with open(_OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    tag = (f"{cfg.arm} tc{cfg.tc} K{cfg.K} eps{cfg.eps_fd:g} s{cfg.seed}")
    print(f"[{time.strftime('%H:%M:%S')}] {stage:9s} {tag:58s} "
          f"MAE {rec.get('mass_mae', float('nan')):.4f} "
          f"({rec['wall_seconds']:.0f}s)"
          + (f"  ERROR {rec['error']}" if rec.get("error") else ""),
          flush=True)
    return rec


def enumerate_jobs(args, base: ToyConfig):
    jobs = []
    stages = {s.strip() for s in args.stages.split(",")}
    if "1" in stages:
        for tc in (0.5, 0.7, 0.8, 0.9):
            for arm in ("btm_vector", "btm_scalar_fd_directional"):
                for s in range(args.tc_seeds):
                    jobs.append(("tc_sweep",
                                 asdict(replace(base, arm=arm, tc=tc, seed=s))))
    if "2" in stages:
        for arm in ALL_ARMS:
            for s in range(args.seeds):
                jobs.append(("main", asdict(replace(base, arm=arm, tc=args.tc,
                                                    seed=s))))
    if "3" in stages:
        eps_ladder = sorted({args.eps_fd / 3, args.eps_fd, args.eps_fd * 3})
        for arm in ("btm_scalar_fd_directional", "btm_scalar_fd_action"):
            for K in (1, 4, 8):
                for eps in eps_ladder:
                    for s in range(args.grid_seeds):
                        jobs.append(("fd_grid",
                                     asdict(replace(base, arm=arm, tc=args.tc,
                                                    seed=s, K=K, eps_fd=eps))))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stages", default="1,2,3")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--eval-n", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--tc-seeds", type=int, default=3)
    ap.add_argument("--grid-seeds", type=int, default=5)
    ap.add_argument("--tc", type=float, default=0.8)
    ap.add_argument("--eps-fd", type=float, default=3e-3)
    args = ap.parse_args()

    global _OUT
    os.makedirs(args.out_dir, exist_ok=True)
    # One file per shard: independent OS processes, no shared lock, no
    # interleaved-write corruption.  `load_campaign` globs them back together.
    out = os.path.join(args.out_dir, f"toy_campaign_shard{args.shard}.jsonl")
    _OUT = out
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    base = ToyConfig(steps=args.steps, batch=args.batch, eval_n=args.eval_n,
                     tc=args.tc, eps_fd=args.eps_fd, device="cpu")
    jobs = enumerate_jobs(args, base)
    jobs = [j for i, j in enumerate(jobs) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(jobs)} runs, "
          f"{args.threads} threads -> {out}", flush=True)
    for j in jobs:
        _work(j)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
