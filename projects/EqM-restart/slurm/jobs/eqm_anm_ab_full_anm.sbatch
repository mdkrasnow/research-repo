#!/bin/bash
#SBATCH --job-name=eqm-full-anm
#SBATCH --output=projects/EqM-restart/slurm/logs/eqm-full-anm_%A.out
#SBATCH --error=projects/EqM-restart/slurm/logs/eqm-full-anm_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1

# FULL-level run: EqM-S/8 to a large equal-step budget on a full A100 (batch 256).
# Companion to the 3hr gpu_test probe; shows where each arm plateaus.
# Equal-step (100000) keeps it a fair comparison; MAX_MINUTES is a graceful backstop
# under the SLURM wall so sample+FID still run.
set -euo pipefail
export ARM=anm
export GLOBAL_BATCH=256
export SAMPLE_BATCH=125
export NUM_FID=10000
export MAX_STEPS=100000
export MAX_MINUTES=1350
exec bash /n/home03/mkrasnow/research-repo/projects/EqM-restart/slurm/jobs/eqm_anm_ab.sbatch
