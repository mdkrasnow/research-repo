#!/bin/bash
#SBATCH --job-name=cons-agg
#SBATCH --output=slurm/logs/cons-agg_%A.out
#SBATCH --error=slurm/logs/cons-agg_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --partition=seas_compute

# Phase 1 — aggregate the parallel per-seed consistency results into the table.
# Submit with --dependency=afterok:<seed0>:<seed1>:<seed2>.
#   N_SEEDS=3 NUM_SLOTS=50000 sbatch --dependency=afterok:J0:J1:J2 projects/diff-EqM/slurm/jobs/consistency_aggonly.sbatch

set -euo pipefail
module load python/3.10.13-fasrc01
REPO_ROOT="${REPO_ROOT:-/n/home03/mkrasnow/research-repo}"
cd "$REPO_ROOT"
N_SEEDS="${N_SEEDS:-3}"
NUM_SLOTS="${NUM_SLOTS:-50000}"
R="${R:-3}"
RUN_TAG="${RUN_TAG:-b2_vanilla}"
SD="$REPO_ROOT/projects/diff-EqM/experiments/separability_diagnostic"
CONS="$SD/runs/$RUN_TAG/results/consistency"
export PYTHONUNBUFFERED=1
echo "=== CONSISTENCY AGG n_seeds=$N_SEEDS $(date) ==="
python "$SD/consistency_agg.py" --cons-dir "$CONS" --n-seeds "$N_SEEDS" \
    --num-slots "$NUM_SLOTS" --r "$R" 2>&1
cat "$CONS/CONSISTENCY_SUMMARY.md" || true
echo "=== AGG DONE $(date) ==="
