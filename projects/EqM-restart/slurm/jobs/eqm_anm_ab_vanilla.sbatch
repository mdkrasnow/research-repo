#!/bin/bash
#SBATCH --job-name=eqm-ab-vanilla
#SBATCH --output=projects/EqM-restart/slurm/logs/eqm-ab-vanilla_%A.out
#SBATCH --error=projects/EqM-restart/slurm/logs/eqm-ab-vanilla_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Vanilla arm of the ANM A/B. remote_submit only exports GIT_URL/GIT_SHA, so set
# ARM here and run the shared engine as plain bash (its #SBATCH lines are comments).
set -euo pipefail
export ARM=vanilla
exec bash "$(dirname "$0")/eqm_anm_ab.sbatch"
