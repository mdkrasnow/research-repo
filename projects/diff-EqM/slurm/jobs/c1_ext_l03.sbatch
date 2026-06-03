#!/bin/bash
#SBATCH --job-name=eqm-c1-ext-l03
#SBATCH --output=projects/diff-EqM/slurm/logs/eqm-c1-ext-l03_%A.out
#SBATCH --error=projects/diff-EqM/slurm/logs/eqm-c1-ext-l03_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
# C1-ext dose arm l03: v10 lambda=0.3 only (skip-vanilla; vanilla comes from l01 job). Sets ARM_MODE then execs the
# worker (env-var pattern from exp1_full_gpu.sbatch; remote_submit only forwards
# GIT_SHA/GIT_URL, so dose selection must be baked into a wrapper).
set -euo pipefail
export ARM_MODE=l03
export NUM_SAMPLES="${NUM_SAMPLES:-5000}"
export NFE_LIST="${NFE_LIST:-250,500,1000}"
export STEP_MULTS="${STEP_MULTS:-1.0}"
exec bash /n/home03/mkrasnow/research-repo/projects/diff-EqM/slurm/jobs/c1_ext_scaling.sbatch
