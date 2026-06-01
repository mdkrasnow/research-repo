#!/bin/bash
#SBATCH --job-name=exp1-full-gpu
#SBATCH --output=projects/diff-EqM/slurm/logs/exp1-full-gpu_%A.out
#SBATCH --error=projects/diff-EqM/slurm/logs/exp1-full-gpu_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Full Experiment 1 sweep, SINGLE-GPU on the gpu partition (faster start than the
# seas_gpu queue; single-GPU is MIG-safe -- multi-GPU DDP would risk the gpu_requeue
# MIG/NCCL roulette). 80 cells @5k samples (~11h). 50k = paper-grade RESUME=1
# follow-up. EVAL ONLY.
set -euo pipefail
export NUM_SAMPLES=5000
export NUM_GPUS=1
export SAMPLE_BATCH=64
export GLOBAL_SEED=123
export VANILLA_CKPT=projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt
export ANM_CKPT=projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt
exec bash /n/home03/mkrasnow/research-repo/projects/diff-EqM/slurm/jobs/exp1_sweep.sbatch
