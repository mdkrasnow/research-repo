#!/bin/bash
#SBATCH --job-name=exp1-full
#SBATCH --output=projects/diff-EqM/slurm/logs/exp1-full_%A.out
#SBATCH --error=projects/diff-EqM/slurm/logs/exp1-full_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Full Experiment 1 sweep: 80 cells (vanilla+anm x gd,ngd x nfe{10,25,50,100,250}
# x step_mult{0.5,1,1.5,2}) at 10k samples on 4xA100. ~6h. Interpretable
# robustness curves / AUC / nfe-to-match; the 50k paper-grade pass is a RESUME=1
# follow-up. EVAL ONLY. Resumable (RESUME=1 skips completed cells).
set -euo pipefail
export NUM_SAMPLES=10000
export NUM_GPUS=4
export SAMPLE_BATCH=128                 # global batch (32/GPU on 4xA100) -- safe for B/2 + VAE
export GLOBAL_SEED=123
export VANILLA_CKPT=projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt
export ANM_CKPT=projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt
exec bash /n/home03/mkrasnow/research-repo/projects/diff-EqM/slurm/jobs/exp1_sweep.sbatch
