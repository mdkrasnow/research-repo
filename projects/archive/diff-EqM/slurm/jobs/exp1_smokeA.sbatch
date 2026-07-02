#!/bin/bash
#SBATCH --job-name=exp1-smokeA
#SBATCH --output=projects/diff-EqM/slurm/logs/exp1-smokeA_%A.out
#SBATCH --error=projects/diff-EqM/slurm/logs/exp1-smokeA_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Smoke A for Experiment 1 (NFE/sampler robustness): 2 cells (vanilla+anm, gd,
# nfe10, step1.0), 64 samples, 1 GPU. Gates the full sweep -- verifies both
# checkpoints load, EMA selected, VAE decodes, FID/KID run, 2 CSV rows written.
# Thin env-wrapper around exp1_sweep.sbatch (repo convention). EVAL ONLY.
set -euo pipefail
export SMOKE=A
export NUM_SAMPLES=64
export NUM_GPUS=1
export SAMPLE_BATCH=32                 # 20GB gpu_test card: keep B/2 + VAE decode in memory
export VANILLA_CKPT=projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt
export ANM_CKPT=projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt
exec bash /n/home03/mkrasnow/research-repo/projects/diff-EqM/slurm/jobs/exp1_sweep.sbatch
