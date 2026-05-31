#!/bin/bash
#SBATCH --job-name=eqm-ab-anm
#SBATCH --output=projects/EqM-restart/slurm/logs/eqm-ab-anm_%A.out
#SBATCH --error=projects/EqM-restart/slurm/logs/eqm-ab-anm_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=05:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# ANM arm of the A/B. See eqm_anm_ab_vanilla.sbatch for why ARM is set here.
# ~3hr run: 12000 equal steps (ANM is the slow arm @~1.35 steps/s -> ~150min train).
set -euo pipefail
export ARM=anm
export MAX_STEPS=12000
export MAX_MINUTES=210   # backstop only; step cap should bind first
exec bash /n/home03/mkrasnow/research-repo/projects/EqM-restart/slurm/jobs/eqm_anm_ab.sbatch
