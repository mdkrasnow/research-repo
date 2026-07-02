#!/bin/bash
#SBATCH --job-name=eqm-fid-anm
#SBATCH --output=projects/EqM-restart/slurm/logs/eqm-fid-anm_%A.out
#SBATCH --error=projects/EqM-restart/slurm/logs/eqm-fid-anm_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=00:40:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# FID-only rerun: reuse synced final.pt, re-sample + fixed ref-flatten + FID.
set -euo pipefail
export ARM=anm
export RERUN_FID=1
exec bash /n/home03/mkrasnow/research-repo/projects/EqM-restart/slurm/jobs/eqm_anm_ab.sbatch
