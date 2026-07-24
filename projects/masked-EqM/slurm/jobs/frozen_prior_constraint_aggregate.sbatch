#!/bin/bash
#SBATCH --job-name=eqm-frozen-prior-aggregate
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=projects/masked-EqM/slurm/logs/frozen-prior-aggregate-%j.out
set -euo pipefail
WORK=/tmp/frozen-prior-aggregate-${SLURM_JOB_ID}; mkdir -p "$WORK"; cd "$WORK"
git clone --depth=1 "${GIT_URL:?}" repo; cd repo; git fetch --depth=1 origin "${GIT_SHA:?}"; git checkout "$GIT_SHA"
ROOT="${OUT_ROOT:-/n/home03/mkrasnow/frozen-prior-constraint}"
python projects/masked-EqM/analyze_frozen_prior_constraint.py --input-dir "$ROOT/final" --manifest projects/masked-EqM/configs/frozen_prior_constraint_final_locked_v4.json --output "$ROOT/final_summary.json"
