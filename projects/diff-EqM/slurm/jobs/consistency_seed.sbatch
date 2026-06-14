#!/bin/bash
#SBATCH --job-name=cons-seed
#SBATCH --output=slurm/logs/cons-seed_%A.out
#SBATCH --error=slurm/logs/cons-seed_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Phase 1 — ONE seed of the 50k probe-gated consistency run. Submit 3 of these in
# PARALLEL (START_SEED=0/1/2), each its own GPU set, then a dependent agg job.
# Each writes results/consistency/seed${START_SEED}/{feat_*,gated_fid.json}.
#
#   START_SEED=0 CKPT_PATH=<...> NUM_SLOTS=50000 sbatch projects/diff-EqM/slurm/jobs/consistency_seed.sbatch
# Required: CKPT_PATH, START_SEED. Optional: NUM_SLOTS, R, BATCH_SIZE, RUN_TAG.

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

REPO_ROOT="${REPO_ROOT:-/n/home03/mkrasnow/research-repo}"
cd "$REPO_ROOT"
WORK="/tmp/cons-seed-${SLURM_JOB_ID}"; mkdir -p "$WORK"
pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -3

CKPT_PATH="${CKPT_PATH:?CKPT_PATH required}"
START_SEED="${START_SEED:?START_SEED required}"
REAL_DIR="${REAL_DIR:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
REF_STATS="${REF_STATS:-$REPO_ROOT/projects/diff-EqM/results/in1k_reference_stats.npz}"
NUM_SLOTS="${NUM_SLOTS:-50000}"
R="${R:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-b2_vanilla}"
SD="$REPO_ROOT/projects/diff-EqM/experiments/separability_diagnostic"
OUT_DIR="$SD/runs/$RUN_TAG"
ART="$OUT_DIR/probe_artifact.npz"
SOUT="$OUT_DIR/results/consistency/seed${START_SEED}"; mkdir -p "$SOUT"

if [ -f "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH="$REPO_ROOT/$CKPT_PATH";
elif [ -f "$CKPT_PATH" ]; then :;
elif [ -d "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH=$(find "$REPO_ROOT/$CKPT_PATH" -name '*.pt' | sort -V | tail -1);
elif [ -d "$CKPT_PATH" ]; then CKPT_PATH=$(find "$CKPT_PATH" -name '*.pt' | sort -V | tail -1); fi
[ -f "$CKPT_PATH" ] || { echo "ERROR: ckpt not found"; exit 1; }
LOCAL_CKPT="$WORK/ckpt.pt"; cp "$CKPT_PATH" "$LOCAL_CKPT"
[ -f "$ART" ] || { echo "ERROR: probe artifact missing ($ART)"; exit 1; }
# reuse a shared real bank if a sibling seed already built one
SHARED_BANK="$OUT_DIR/results/consistency/real_bank.npy"
[ -f "$SHARED_BANK" ] && cp "$SHARED_BANK" "$SOUT/real_bank.npy" 2>/dev/null || true

export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
PORT=$((29500 + RANDOM % 1000))
echo "=== CONSISTENCY-SEED seed=$START_SEED slots=$NUM_SLOTS R=$R gpus=$NUM_GPUS $(date) ==="
python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$PORT \
    "$SD/probe_gated_sample.py" \
    --model EqM-B/2 --ckpt "$LOCAL_CKPT" --probe-artifact "$ART" --out "$SOUT" \
    --real-dir "$REAL_DIR" --num-real-bank 10000 \
    --num-slots "$NUM_SLOTS" --R "$R" --seed-offset "$START_SEED" \
    --stepsize 0.003 --num-sampling-steps 250 --batch-size "$BATCH_SIZE" 2>&1 | tee "$WORK/seed.log"

# publish the bank for siblings (first to finish wins; harmless if racing)
[ -f "$SOUT/real_bank.npy" ] && cp -n "$SOUT/real_bank.npy" "$SHARED_BANK" 2>/dev/null || true
echo "=== per-seed FID $(date) ==="
python "$SD/fid_gated_agg.py" --out "$SOUT" --ref-stats "$REF_STATS" 2>&1 | tee "$WORK/agg.log"
cat "$SOUT/GATED_FID.txt" || true
cp "$WORK"/*.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/" 2>/dev/null || true
rm -rf "$WORK"
echo "=== SEED $START_SEED DONE $(date) ==="
