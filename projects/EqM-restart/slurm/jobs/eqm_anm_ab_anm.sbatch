#!/bin/bash
#SBATCH --job-name=eqm-ab-anm
#SBATCH --output=projects/EqM-restart/slurm/logs/eqm-ab-anm_%A.out
#SBATCH --error=projects/EqM-restart/slurm/logs/eqm-ab-anm_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# ANM arm of the A/B. See eqm_anm_ab_vanilla.sbatch for why ARM is set here.
set -euo pipefail
export ARM=anm
exec bash "$(dirname "$0")/eqm_anm_ab.sbatch"
