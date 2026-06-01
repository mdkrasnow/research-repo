#!/bin/bash
#SBATCH --job-name=exp1-smokeB
#SBATCH --output=projects/diff-EqM/slurm/logs/exp1-smokeB_%A.out
#SBATCH --error=projects/diff-EqM/slurm/logs/exp1-smokeB_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Smoke B for Experiment 1: 16 cells (vanilla+anm x gd,ngd x nfe{10,25} x
# step{1.0,2.0}), 256 samples, 1 GPU. Validates the NGD path + step2.0 stress
# (not all-NaN) + heatmap/AUC plumbing on a 2-nfe grid. EVAL ONLY.
set -euo pipefail
export SMOKE=B
export NUM_SAMPLES=256
export NUM_GPUS=1
export SAMPLE_BATCH=32
export VANILLA_CKPT=projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt
export ANM_CKPT=projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt
exec bash /n/home03/mkrasnow/research-repo/projects/diff-EqM/slurm/jobs/exp1_sweep.sbatch
