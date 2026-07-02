#!/bin/bash
#SBATCH --job-name=consistency
#SBATCH --output=slurm/logs/consistency_%A.out
#SBATCH --error=slurm/logs/consistency_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Phase 1 — 50k consistency: probe-gated best-of-R across N_SEEDS independent
# sampler seeds (SEED_OFFSET). Each seed -> vanilla/probe/oracle 50k FID. Builds
# results/consistency/fid_table.csv + CONSISTENCY_SUMMARY.md (mean/std across seeds).
#
#   cd <REPO_ROOT> && CKPT_PATH=<...> NUM_SLOTS=50000 N_SEEDS=3 sbatch projects/diff-EqM/slurm/jobs/consistency.sbatch
# Smoke first: NUM_SLOTS=2000 N_SEEDS=2.
# Required: CKPT_PATH. Optional: NUM_SLOTS, N_SEEDS, R, BATCH_SIZE, RUN_TAG.

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

REPO_ROOT="${REPO_ROOT:-/n/home03/mkrasnow/research-repo}"
cd "$REPO_ROOT"
WORK="/tmp/consistency-${SLURM_JOB_ID}"; mkdir -p "$WORK"
pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -3

CKPT_PATH="${CKPT_PATH:?CKPT_PATH required}"
REAL_DIR="${REAL_DIR:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
REF_STATS="${REF_STATS:-$REPO_ROOT/projects/diff-EqM/results/in1k_reference_stats.npz}"
NUM_SLOTS="${NUM_SLOTS:-50000}"
N_SEEDS="${N_SEEDS:-3}"
R="${R:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-b2_vanilla}"
SD="$REPO_ROOT/projects/diff-EqM/experiments/separability_diagnostic"
OUT_DIR="$SD/runs/$RUN_TAG"
ART="$OUT_DIR/probe_artifact.npz"
CONS="$OUT_DIR/results/consistency"; mkdir -p "$CONS"

if [ -f "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH="$REPO_ROOT/$CKPT_PATH";
elif [ -f "$CKPT_PATH" ]; then :;
elif [ -d "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH=$(find "$REPO_ROOT/$CKPT_PATH" -name '*.pt' | sort -V | tail -1);
elif [ -d "$CKPT_PATH" ]; then CKPT_PATH=$(find "$CKPT_PATH" -name '*.pt' | sort -V | tail -1); fi
[ -f "$CKPT_PATH" ] || { echo "ERROR: ckpt not found"; exit 1; }
LOCAL_CKPT="$WORK/ckpt.pt"; cp "$CKPT_PATH" "$LOCAL_CKPT"
[ -f "$ART" ] || { echo "ERROR: probe artifact missing ($ART)"; exit 1; }

export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')

for SEED in $(seq 0 $((N_SEEDS - 1))); do
  SOUT="$CONS/seed${SEED}"; mkdir -p "$SOUT"
  # reuse the seed0 real bank across seeds
  [ -f "$CONS/real_bank.npy" ] && cp "$CONS/real_bank.npy" "$SOUT/real_bank.npy" 2>/dev/null || true
  PORT=$((29500 + RANDOM % 1000))
  echo "=== CONSISTENCY seed=$SEED slots=$NUM_SLOTS R=$R $(date) ==="
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$PORT \
      "$SD/probe_gated_sample.py" \
      --model EqM-B/2 --ckpt "$LOCAL_CKPT" --probe-artifact "$ART" --out "$SOUT" \
      --real-dir "$REAL_DIR" --num-real-bank 10000 \
      --num-slots "$NUM_SLOTS" --R "$R" --seed-offset "$SEED" \
      --stepsize 0.003 --num-sampling-steps 250 --batch-size "$BATCH_SIZE" 2>&1 | tee "$WORK/seed${SEED}.log"
  [ -f "$SOUT/real_bank.npy" ] && cp "$SOUT/real_bank.npy" "$CONS/real_bank.npy" 2>/dev/null || true
  python "$SD/fid_gated_agg.py" --out "$SOUT" --ref-stats "$REF_STATS" 2>&1 | tee -a "$WORK/seed${SEED}.log"
done

echo "=== BUILD CONSISTENCY TABLE $(date) ==="
python "$SD/consistency_agg.py" --cons-dir "$CONS" --n-seeds "$N_SEEDS" \
    --num-slots "$NUM_SLOTS" --r "$R" 2>&1 | tee "$WORK/agg.log"
cat "$CONS/CONSISTENCY_SUMMARY.md" || true
cp "$WORK"/*.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/" 2>/dev/null || true
rm -rf "$WORK"
