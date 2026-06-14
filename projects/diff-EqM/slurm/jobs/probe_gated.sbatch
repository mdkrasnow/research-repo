#!/bin/bash
#SBATCH --job-name=probe-gated
#SBATCH --output=slurm/logs/probe-gated_%A.out
#SBATCH --error=slurm/logs/probe-gated_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=12:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# In-line restart sampler (probe-guided best-of-R) at scale, with controls.
# No-clone, in-place. SMOKE: NUM_SLOTS=512 R=3 (1 GPU, gpu_test) first.
#   cd <REPO_ROOT> && NUM_SLOTS=50000 R=3 sbatch projects/diff-EqM/slurm/jobs/probe_gated.sbatch
#
# Required env: CKPT_PATH. Optional: NUM_SLOTS, R, RUN_TAG, NUM_REAL_BANK, BATCH_SIZE,
#   REF_STATS, REAL_DIR.

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

REPO_ROOT="${REPO_ROOT:-/n/home03/mkrasnow/research-repo}"
cd "$REPO_ROOT"
WORK="/tmp/probe-gated-${SLURM_JOB_ID}"; mkdir -p "$WORK"
pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -3

CKPT_PATH="${CKPT_PATH:?CKPT_PATH required}"
REAL_DIR="${REAL_DIR:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
REF_STATS="${REF_STATS:-$REPO_ROOT/projects/diff-EqM/results/in1k_reference_stats.npz}"
NUM_SLOTS="${NUM_SLOTS:-50000}"
R="${R:-3}"
NUM_REAL_BANK="${NUM_REAL_BANK:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-b2_vanilla}"
SD="$REPO_ROOT/projects/diff-EqM/experiments/separability_diagnostic"
OUT_DIR="$SD/runs/$RUN_TAG"
ART="$OUT_DIR/probe_artifact.npz"
GATED_OUT="$OUT_DIR/gated_R${R}_N${NUM_SLOTS}"
mkdir -p "$GATED_OUT"

# resolve ckpt (file / repo-relative / dir of nested .pt)
if [ -f "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH="$REPO_ROOT/$CKPT_PATH";
elif [ -f "$CKPT_PATH" ]; then :;
elif [ -d "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH=$(find "$REPO_ROOT/$CKPT_PATH" -name '*.pt' | sort -V | tail -1);
elif [ -d "$CKPT_PATH" ]; then CKPT_PATH=$(find "$CKPT_PATH" -name '*.pt' | sort -V | tail -1); fi
[ -f "$CKPT_PATH" ] || { echo "ERROR: ckpt not found"; exit 1; }
LOCAL_CKPT="$WORK/ckpt.pt"; cp "$CKPT_PATH" "$LOCAL_CKPT"
[ -f "$ART" ] || { echo "ERROR: probe artifact missing ($ART); run probe_validate first"; exit 1; }

export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
PORT=$((29500 + RANDOM % 1000))
echo "=== GATED SAMPLE slots=$NUM_SLOTS R=$R gpus=$NUM_GPUS $(date) ==="
python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$PORT \
    "$SD/probe_gated_sample.py" \
    --model EqM-B/2 --ckpt "$LOCAL_CKPT" --probe-artifact "$ART" --out "$GATED_OUT" \
    --real-dir "$REAL_DIR" --num-real-bank "$NUM_REAL_BANK" \
    --num-slots "$NUM_SLOTS" --R "$R" --stepsize 0.003 --num-sampling-steps 250 \
    --batch-size "$BATCH_SIZE" 2>&1 | tee "$WORK/gated.log"

echo "=== AGG FID $(date) ==="
python "$SD/fid_gated_agg.py" --out "$GATED_OUT" --ref-stats "$REF_STATS" 2>&1 | tee "$WORK/agg.log"
echo "=== DONE $(date) ==="
cat "$GATED_OUT/GATED_FID.txt" || true
cp "$WORK/gated.log" "$WORK/agg.log" "$REPO_ROOT/projects/diff-EqM/slurm/logs/" 2>/dev/null || true
rm -rf "$WORK"
