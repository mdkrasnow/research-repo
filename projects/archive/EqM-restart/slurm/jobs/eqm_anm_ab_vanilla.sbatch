#!/bin/bash
#SBATCH --job-name=eqm-ab-vanilla
#SBATCH --output=projects/EqM-restart/slurm/logs/eqm-ab-vanilla_%A.out
#SBATCH --error=projects/EqM-restart/slurm/logs/eqm-ab-vanilla_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=05:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Vanilla arm of the ANM A/B. remote_submit only exports GIT_URL/GIT_SHA, so set
# ARM here and run the shared engine as plain bash (its #SBATCH lines are comments).
# Equal-step budget with the ANM arm (12000); vanilla finishes faster (no mining).
set -euo pipefail
export ARM=vanilla
export MAX_STEPS=12000
export MAX_MINUTES=210   # backstop only; step cap should bind first
exec bash /n/home03/mkrasnow/research-repo/projects/EqM-restart/slurm/jobs/eqm_anm_ab.sbatch
