#!/bin/bash
#SBATCH --job-name=eqm-energy-calibration
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/energy-calibration-%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/energy-calibration-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --time=04:00:00
#SBATCH --partition=shared
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
GIT_SHA="${GIT_SHA:?set GIT_SHA}"
SOURCE_DIR="${SOURCE_DIR:-/n/home03/mkrasnow/energy_monotonicity_full}"
OUTPUT_DIR="${OUTPUT_DIR:-/n/home03/mkrasnow/energy_calibration_eval}"
WORK_DIR="$(mktemp -d /tmp/masked-eqm-calibration-${SLURM_JOB_ID}-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT
git -C "$WORK_DIR" init
git -C "$WORK_DIR" remote add origin "${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
git -C "$WORK_DIR" fetch --depth=50 origin main
git -C "$WORK_DIR" checkout "$GIT_SHA"
cd "$WORK_DIR/projects/masked-EqM"
pip install --quiet torch pandas pyarrow matplotlib
python -m energy_monotonicity.evaluate_energy_calibration \
  --monotonicity-output-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR" \
  --epochs 1 8 --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-10000}" \
  --bootstrap-seed "${BOOTSTRAP_SEED:-23456}"
