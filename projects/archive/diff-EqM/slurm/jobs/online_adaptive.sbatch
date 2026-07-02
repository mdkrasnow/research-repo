#!/bin/bash
#SBATCH --job-name=online-adaptive
#SBATCH --output=slurm/logs/online-adaptive_%A.out
#SBATCH --error=slurm/logs/online-adaptive_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=12:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Phase 2 — online equal-NFE adaptive sampler at scale, with controls.
# Decides at k_dec from the PARTIAL trajectory; restarts flagged slots; random
# control restarts the same fraction blindly (matched NFE). SMOKE first:
#   cd <REPO_ROOT> && NUM_SLOTS=512 sbatch ... (1 GPU gpu_test) then 5k/15k.
#
# Required: CKPT_PATH. Optional: NUM_SLOTS, FLAG_FRAC, K_DEC, RUN_TAG, BATCH_SIZE,
#   PARTIAL_PROBE (defaults to runs/<tag>/results/partial_probe/partial_probe_k100.npz).

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

REPO_ROOT="${REPO_ROOT:-/n/home03/mkrasnow/research-repo}"
cd "$REPO_ROOT"
WORK="/tmp/online-adaptive-${SLURM_JOB_ID}"; mkdir -p "$WORK"
pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -3

CKPT_PATH="${CKPT_PATH:?CKPT_PATH required}"
REAL_DIR="${REAL_DIR:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
REF_STATS="${REF_STATS:-$REPO_ROOT/projects/diff-EqM/results/in1k_reference_stats.npz}"
NUM_SLOTS="${NUM_SLOTS:-15000}"
FLAG_FRAC="${FLAG_FRAC:-0.3}"
K_DEC="${K_DEC:--1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-b2_vanilla}"
SD="$REPO_ROOT/projects/diff-EqM/experiments/separability_diagnostic"
OUT_DIR="$SD/runs/$RUN_TAG"
PARTIAL_PROBE="${PARTIAL_PROBE:-$OUT_DIR/results/partial_probe/partial_probe_k100.npz}"
ONLINE_OUT="$OUT_DIR/online_f${FLAG_FRAC}_N${NUM_SLOTS}"
mkdir -p "$ONLINE_OUT"

if [ -f "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH="$REPO_ROOT/$CKPT_PATH";
elif [ -f "$CKPT_PATH" ]; then :;
elif [ -d "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH=$(find "$REPO_ROOT/$CKPT_PATH" -name '*.pt' | sort -V | tail -1);
elif [ -d "$CKPT_PATH" ]; then CKPT_PATH=$(find "$CKPT_PATH" -name '*.pt' | sort -V | tail -1); fi
[ -f "$CKPT_PATH" ] || { echo "ERROR: ckpt not found"; exit 1; }
LOCAL_CKPT="$WORK/ckpt.pt"; cp "$CKPT_PATH" "$LOCAL_CKPT"
[ -f "$PARTIAL_PROBE" ] || { echo "ERROR: partial probe missing ($PARTIAL_PROBE); run partial_probe.py first"; exit 1; }

export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
PORT=$((29500 + RANDOM % 1000))
echo "=== ONLINE ADAPTIVE slots=$NUM_SLOTS flag=$FLAG_FRAC k_dec=$K_DEC gpus=$NUM_GPUS $(date) ==="
python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$PORT \
    "$SD/online_adaptive_sampler.py" \
    --model EqM-B/2 --ckpt "$LOCAL_CKPT" --partial-probe "$PARTIAL_PROBE" --out "$ONLINE_OUT" \
    --real-dir "$REAL_DIR" --num-real-bank 10000 \
    --num-slots "$NUM_SLOTS" --flag-frac "$FLAG_FRAC" --k-dec "$K_DEC" \
    --stepsize 0.003 --num-sampling-steps 250 --batch-size "$BATCH_SIZE" 2>&1 | tee "$WORK/online.log"

echo "=== AGG FID $(date) ==="
python "$SD/fid_online_agg.py" --out "$ONLINE_OUT" --ref-stats "$REF_STATS" 2>&1 | tee "$WORK/agg.log"
echo "=== DONE $(date) ==="
cat "$ONLINE_OUT/ONLINE_FID.txt" || true
cp "$WORK/online.log" "$WORK/agg.log" "$REPO_ROOT/projects/diff-EqM/slurm/logs/" 2>/dev/null || true
rm -rf "$WORK"
