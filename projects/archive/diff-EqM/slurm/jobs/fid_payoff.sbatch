#!/bin/bash
#SBATCH --job-name=fid-payoff
#SBATCH --output=slurm/logs/fid-payoff_%A.out
#SBATCH --error=slurm/logs/fid-payoff_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60GB
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Probe payoff test (no-clone, in-place). Regenerates the probe artifact from the
# cached shard+labels (CPU), then runs the FID payoff with positive/negative
# controls (GPU inception). Reuses the existing 3000-sample run — NO generation.
#   cd <REPO_ROOT> && RUN_TAG=b2_vanilla sbatch projects/diff-EqM/slurm/jobs/fid_payoff.sbatch

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

REPO_ROOT="${REPO_ROOT:-/n/home03/mkrasnow/research-repo}"
cd "$REPO_ROOT"
pip install timm diffusers accelerate scipy torchvision pytorch-fid matplotlib 2>&1 | tail -3

REAL_DIR="${REAL_DIR:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
NUM_REAL="${NUM_REAL:-10000}"
KEEP_FRAC="${KEEP_FRAC:-0.70}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-b2_vanilla}"
SD="$REPO_ROOT/projects/diff-EqM/experiments/separability_diagnostic"
OUT_DIR="$SD/runs/$RUN_TAG"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1

echo "=== regen probe artifact (held-out validate + save) $(date) ==="
python "$SD/probe_validate.py" --folder "$OUT_DIR" 2>&1 | tee /tmp/pv_${SLURM_JOB_ID}.log

echo "=== FID payoff (probe vs random vs oracle, keep_frac=$KEEP_FRAC) $(date) ==="
python "$SD/fid_payoff.py" --folder "$OUT_DIR" --real-dir "$REAL_DIR" \
    --num-real "$NUM_REAL" --keep-fracs 0.5 0.6 0.7 0.8 0.9 --batch-size "$BATCH_SIZE" \
    2>&1 | tee /tmp/fp_${SLURM_JOB_ID}.log

echo "=== DONE $(date) ==="
cat "$OUT_DIR/results/FID_PAYOFF.txt" || true
cp /tmp/pv_${SLURM_JOB_ID}.log /tmp/fp_${SLURM_JOB_ID}.log \
   "$REPO_ROOT/projects/diff-EqM/slurm/logs/" 2>/dev/null || true
