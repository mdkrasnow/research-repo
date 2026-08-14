"""Launcher for Experiment II (ImageNet B/2), staged per section 14.

Phase II-A  short stability runs, >=3 seeds per arm, enough steps to expose the
            early instability signature, evaluated frequently
Phase II-B  viable arms advanced to a substantial fraction of training, >=3
            seeds retained for the key G-vs-D comparison
Phase II-C  full B/2 horizon for the survivors

Emits the sbatch submission lines (and, with --submit, runs them through
scripts/cluster/ssh.sh), and appends one manifest record per job to
results/btm/manifest.jsonl with git commit, exact command, config, seed, GPU
count, SLURM id, timestamps and status -- the machine-readable experiment
manifest the campaign is required to keep.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(PROJECT))
MANIFEST = os.path.join(PROJECT, "results", "btm", "manifest.jsonl")

PHASES = {
    "IIA": dict(epochs=80, max_steps=20000, save_epochs="1",
                seeds=3, nproc=4, partition="seas_gpu",
                arms=("btm_vector", "btm_scalar_exact",
                      "btm_scalar_fd_directional",
                      "btm_scalar_fd_directional4")),
    "IIB": dict(epochs=15, max_steps=None, save_epochs="1,2,5,10,15",
                seeds=3, nproc=4, partition="seas_gpu",
                arms=("btm_vector", "btm_scalar_exact",
                      "btm_scalar_fd_directional")),
    "IIC": dict(epochs=80, max_steps=None, save_epochs="1,2,5,10,20,40,80",
                seeds=3, nproc=4, partition="seas_gpu",
                arms=("btm_vector", "btm_scalar_exact",
                      "btm_scalar_fd_directional")),
}

# arm -> (btm_mode, ebm, fd_k)
ARM_SPEC = {
    "btm_vector": ("btm_vector", "none", 1),
    "btm_scalar_exact": ("btm_scalar_exact", "direct", 1),
    "btm_scalar_action_exact": ("btm_scalar_action_exact", "direct", 1),
    "btm_scalar_fd_directional": ("btm_scalar_fd_directional", "direct", 1),
    "btm_scalar_fd_directional4": ("btm_scalar_fd_directional", "direct", 4),
    "btm_scalar_fd_action": ("btm_scalar_fd_action", "direct", 1),
}


def build_cmds(args):
    spec = PHASES[args.phase]
    arms = args.arms.split(",") if args.arms else spec["arms"]
    n_seeds = args.seeds if args.seeds else spec["seeds"]
    cmds = []
    for arm in arms:
        mode, ebm, fd_k = ARM_SPEC[arm]
        for s in range(n_seeds):
            tag = f"btm_{args.phase}_{arm}_s{s}"
            env = [
                f"GIT_SHA={args.git_sha}",
                f"BTM_MODE={mode}", f"EBM={ebm}", f"GLOBAL_SEED={s}",
                f"RUN_TAG={tag}", f"EPOCHS={spec['epochs']}",
                f"SAVE_EPOCHS={spec['save_epochs']}",
                f"BTM_TC={args.tc}", f"FD_EPS={args.fd_eps}", f"FD_K={fd_k}",
                f"NPROC={spec['nproc']}", f"GLOBAL_BATCH={args.global_batch}",
                f"GRAD_LOG_EVERY={args.grad_log_every}",
                f"BTM_EVAL_EVERY={args.btm_eval_every}",
            ]
            if spec["max_steps"]:
                env.append(f"MAX_STEPS={spec['max_steps']}")
            cmd = (f"cd /n/home03/mkrasnow/research-repo && {' '.join(env)} "
                   f"sbatch -p {spec['partition']} "
                   f"projects/masked-EqM/slurm/jobs/btm_image_arm.sbatch")
            cmds.append((tag, arm, mode, ebm, fd_k, s, spec, cmd))
    return cmds


def record(rec):
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), required=True)
    ap.add_argument("--git-sha", required=True)
    ap.add_argument("--tc", type=float, required=True)
    ap.add_argument("--fd-eps", type=float, required=True)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--global-batch", type=int, default=256)
    ap.add_argument("--grad-log-every", type=int, default=50)
    ap.add_argument("--btm-eval-every", type=int, default=500)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    for tag, arm, mode, ebm, fd_k, s, spec, cmd in build_cmds(args):
        job_id = None
        if args.submit:
            out = subprocess.run(
                [os.path.join(REPO, "scripts/cluster/ssh.sh"), cmd],
                capture_output=True, text=True, timeout=180).stdout
            for line in out.splitlines():
                if "Submitted batch job" in line:
                    job_id = line.strip().split()[-1]
            print(f"{tag}: {job_id or out.strip()[:120]}", flush=True)
        else:
            print(cmd)
        record({
            "run_tag": tag, "phase": args.phase, "arm": arm, "btm_mode": mode,
            "ebm": ebm, "fd_k": fd_k, "fd_eps": args.fd_eps, "tc": args.tc,
            "seed": s, "epochs": spec["epochs"], "max_steps": spec["max_steps"],
            "n_gpus": spec["nproc"], "partition": spec["partition"],
            "global_batch": args.global_batch,
            "git_sha": args.git_sha, "command": cmd, "job_id": job_id,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "submitted" if job_id else "planned",
        })


if __name__ == "__main__":
    main()
