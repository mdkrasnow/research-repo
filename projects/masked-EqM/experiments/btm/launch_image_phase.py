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

# Must match `eqm_prelude` in slurm/jobs/btm_image_arm.sbatch. Kept as one
# constant so a resume path cannot silently disagree with where the producing
# run actually wrote.
RESULTS_ROOT = "/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/btm"

PHASES = {
    "IIA": dict(epochs=80, max_steps=20000, save_epochs="1",
                seeds=1, nproc=4, partition="seas_gpu",
                arms=("btm_vector", "btm_scalar_exact",
                      "btm_scalar_fd_directional",
                      "btm_scalar_fd_directional4")),
    # seeds=1 per the 2026-08-14 user decision to skip replicates and decide
    # from seed 0; conclusions rest on dense within-run trajectories instead.
    "IIB": dict(epochs=15, max_steps=None, save_epochs="1,2,5,10,15",
                seeds=1, nproc=4, partition="seas_gpu",
                arms=("btm_vector", "btm_scalar_exact",
                      "btm_scalar_fd_directional")),
    # II-C is the DECISIVE phase and its checkpoint schedule is load-bearing.
    # The legacy direct pathology (flat median grad-norm, CV exploding
    # 0.44 -> 50) onsets at ~epoch 55; every II-B result stops at epoch 15,
    # i.e. entirely BEFORE the failure regime, which is why II-B's G~V parity
    # cannot yet be read as an improvement. The old schedule here
    # ("1,2,5,10,20,40,80") jumps 40 -> 80 and so has NO checkpoint anywhere in
    # the onset band -- a run that crossed the transition would leave no
    # artifact from which to characterise it. Sample densely across 50-60.
    "IIC": dict(epochs=60, max_steps=None,
                save_epochs="20,30,40,50,55,60",
                # The sbatch defaults to 24h; II-C needs more. ~300k steps at
                # the measured 2.46 steps/s is ~34h for a scalar arm. Hitting
                # the wall clock mid-run is recoverable (epoch ckpts + resume)
                # but only at the price of another full queue wait, which is
                # currently ~6h.
                time="48:00:00",
                # 48h is seas_gpu's MaxTime, not a choice -- it cannot be
                # raised. At the throughput actually measured on the degraded
                # filesystem (vector 2.00 steps/s, scalar 1.14) the SCALAR arms
                # do not reach epoch 60 in one job, and worse, they do not reach
                # epoch 55 either: G lands at ~54.4 and legacy_scalar at ~41.4.
                # Epoch 55 is the onset this entire phase exists to cross, so a
                # single-leg schedule would spend 48h x 8 GPUs and stop just
                # short of the only region that matters.
                #
                # The scalar arms therefore need two legs, and the thing that
                # makes a second leg cheap is checkpoint density. SAVE_EPOCHS
                # alone leaves a 5-10 epoch gap; a step-based checkpoint every
                # 25k steps (~5 epochs) bounds what a wall-clock kill can
                # destroy, independently of where the epoch boundaries fall.
                ckpt_every=10000,
                seeds=1, nproc=4, partition="seas_gpu",
                arms=("btm_vector", "btm_scalar_exact",
                      "btm_scalar_fd_directional"),
                # II-C continues each arm from its OWN II-B epoch-15 weights.
                # Restarting from scratch would spend ~15 epochs of A100 time
                # re-deriving a trajectory already measured, and -- worse --
                # would make II-C's epoch-15 state a different random draw from
                # II-B's, so the two phases could not be read as one curve.
                # The interpolant parameters of a resumed phase are NOT free.
                # II-B ran at tc=0.9; continuing from its epoch-15 weights under
                # a different tc changes the regression TARGET at the resume
                # boundary, so the two phases would no longer be one trajectory
                # -- and nothing downstream would flag it, because the loss is
                # finite and the curve is smooth on either side. Asserted in
                # build_cmds against --tc.
                resume_tc=0.9,
                resume_from={
                    "legacy_vector":
                        f"{RESULTS_ROOT}/btm_IIB_LEGACYvec_s0_job39210676/"
                        "r756900e96c4cf61f-EqM-B-2-Linear-velocity-None-ebm-none/"
                        "checkpoints/epoch10.pt",
                    "legacy_scalar":
                        f"{RESULTS_ROOT}/btm_IIB_LEGACYscalar_s0_job39210680/"
                        "re624577cb3276d07-EqM-B-2-Linear-velocity-None-ebm-direct/"
                        "checkpoints/epoch02.pt",
                    "btm_vector":
                        f"{RESULTS_ROOT}/btm_IIB_V_s0_job39134329/"
                        "000-EqM-B-2-Linear-velocity-None-ebm-none/"
                        "checkpoints/epoch15.pt",
                    "btm_scalar_exact":
                        f"{RESULTS_ROOT}/btm_IIB_G_s0_job39134331/"
                        "000-EqM-B-2-Linear-velocity-None-ebm-direct/"
                        "checkpoints/epoch15.pt",
                }),
}

# arm -> (btm_mode, ebm, fd_k)
ARM_SPEC = {
    "btm_vector": ("btm_vector", "none", 1),
    "btm_scalar_exact": ("btm_scalar_exact", "direct", 1),
    "btm_scalar_action_exact": ("btm_scalar_action_exact", "direct", 1),
    "btm_scalar_fd_directional": ("btm_scalar_fd_directional", "direct", 1),
    "btm_scalar_fd_directional4": ("btm_scalar_fd_directional", "direct", 4),
    "btm_scalar_fd_action": ("btm_scalar_fd_action", "direct", 1),
    # BTM_MODE=none runs the repository's ORIGINAL (legacy-EqM-target) training
    # path untouched. These two are the other row of the 2x2 -- same legacy
    # target, differing ONLY in parametrization -- and they are what makes the
    # mixed-derivative question answerable at all.
    "legacy_vector": ("none", "none", 1),
    "legacy_scalar": ("none", "direct", 1),
}

# The campaign's causal design, recorded so it cannot drift:
#
#                   vector (no grad_theta grad_x phi)   scalar (has it)
#   legacy target   legacy_vector                       legacy_scalar
#   corrected BTM   btm_vector                          btm_scalar_exact
#
# Reading only the corrected row cannot isolate the mixed derivative, because
# the legacy VECTOR arm has no scalar parametrization and no mixed derivative
# and is ALSO badly behaved (TABLE_C: p95 6.67 vs corrected V's 0.665). The
# legacy row must therefore be run at the SAME batch size and to the SAME epoch
# as the corrected row; the only legacy data past the ~epoch-55 onset is run
# 36847271 at batch 32, which is not comparable to the corrected row's 256.


def build_cmds(args):
    spec = PHASES[args.phase]
    # A resumed phase inherits its predecessor's interpolant, it does not get to
    # re-choose it. Enforced here rather than documented, because the failure is
    # silent: training continues, the loss stays finite, and the only symptom is
    # that the two phases are no longer one trajectory.
    if spec.get("resume_tc") is not None and args.tc != spec["resume_tc"]:
        raise SystemExit(
            f"--tc {args.tc} disagrees with the tc of the run being resumed "
            f"({spec['resume_tc']}). Phase {args.phase} continues from "
            f"checkpoints trained at tc={spec['resume_tc']}; changing tc at the "
            f"resume boundary changes the regression target mid-trajectory and "
            f"nothing downstream would detect it. Pass --tc {spec['resume_tc']}.")
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
            if spec.get("ckpt_every"):
                env.append(f"CKPT_EVERY={spec['ckpt_every']}")
            ckpt = spec.get("resume_from", {}).get(arm)
            if ckpt:
                env.append(f"CKPT={ckpt}")
            # NOTE ON QUOTING: every variable above is passed through the SHELL
            # environment (`VAR=val ... sbatch`), deliberately NOT through
            # `sbatch --export=ALL,VAR=val`. SLURM splits --export on commas, so
            # SAVE_EPOCHS="20,30,40,50,55,60" would arrive as "20" and the
            # epoch-55/60 checkpoints -- the entire reason II-C exists -- would
            # never be written, with no error anywhere. Do not "simplify" this
            # into --export.
            #
            # --job-name is what makes %x in the sbatch's log directives resolve
            # to the run tag; without it every arm's log is named `btm-image`.
            cmd = (f"cd /n/home03/mkrasnow/research-repo && {' '.join(env)} "
                   f"sbatch -p {spec['partition']} --job-name={tag} "
                   + (f"-t {spec['time']} " if spec.get("time") else "")
                   + "projects/masked-EqM/slurm/jobs/btm_image_arm.sbatch")
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
            # A dry run must not append to the experiment manifest. It used to,
            # with status="planned", so every `--phase X` inspection left a row
            # indistinguishable-by-position from a real submission and inflated
            # the manifest with runs that never existed.
            continue
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
