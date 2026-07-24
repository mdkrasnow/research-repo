#!/bin/bash
#SBATCH --job-name=eqm-direct-aggregate
#SBATCH --output=projects/masked-EqM/slurm/logs/direct-full-aggregate-%j.out
#SBATCH --error=projects/masked-EqM/slurm/logs/direct-full-aggregate-%j.err
#SBATCH --partition=shared
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:15:00
set -euo pipefail

# Runs only after every FID-array task succeeds.  The Python aggregator itself
# fails closed if any expected arm/seed JSON file is missing.
WORK_DIR="/tmp/masked-eqm-direct-aggregate-${SLURM_JOB_ID}"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?set GIT_SHA}"
INPUT_DIR="${INPUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/results/direct_energy_campaign/full_eval}"
OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/results/direct_energy_campaign}"

mkdir -p "$WORK_DIR" "$OUT_DIR"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }

python projects/masked-EqM/experiments/direct_energy/aggregate_full_screen.py \
    --input-dir "$INPUT_DIR" \
    --out-json "$OUT_DIR/full_screen_fid.json" \
    --out-md "$OUT_DIR/full_screen_fid.md"

rm -rf "$WORK_DIR"
