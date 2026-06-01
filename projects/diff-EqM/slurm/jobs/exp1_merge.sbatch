#!/bin/bash
#SBATCH --job-name=exp1-merge
#SBATCH --output=projects/diff-EqM/slurm/logs/exp1-merge_%A.out
#SBATCH --error=projects/diff-EqM/slurm/logs/exp1-merge_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Exp1 MERGE/RESUME: finish the ~23 cells lost when run 17828606 died (SIGPIPE
# at 12h56m, anm_ngd never ran). The NFS OUTDIR is pre-seeded with the 57 cells
# recovered from that run's job.log; --resume + the new "trust recorded FID"
# logic skips them and runs only the missing cells, then analyses the full 80.
#
# MUST match 17828606's params so fixed latents/labels + reference reproduce
# identically (paired comparison): NUM_SAMPLES=5000, SAMPLE_BATCH=64, SEED=123,
# single-GPU on `gpu` (A100; batch 64 + 250 steps would risk OOM on 20GB gpu_test).
set -euo pipefail
MERGE_DIR=/n/home03/mkrasnow/research-repo/projects/diff-EqM/results/exp1_run_5k_merged
export OUTDIR_OVERRIDE="$MERGE_DIR"
export SYNC_OVERRIDE="$MERGE_DIR"
export RESUME=1
export NUM_SAMPLES=5000
export NUM_GPUS=1
export SAMPLE_BATCH=64
export GLOBAL_SEED=123
export VANILLA_CKPT=projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt
export ANM_CKPT=projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt
exec bash /n/home03/mkrasnow/research-repo/projects/diff-EqM/slurm/jobs/exp1_sweep.sbatch
