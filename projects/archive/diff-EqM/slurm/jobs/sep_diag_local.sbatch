#!/bin/bash
#SBATCH --job-name=sep-diag
#SBATCH --output=slurm/logs/sep-diag_%A.out
#SBATCH --error=slurm/logs/sep-diag_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60GB
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# NO-CLONE variant of sep_diag.sbatch: runs the diagnostic IN PLACE from the
# rsync'd working tree at REPO_ROOT, so it works WITHOUT pushing to git. Used
# while the upstream is being fixed (can't push a clean SHA yet). Submit with:
#   cd <REPO_ROOT> && NUM_SAMPLES=64 NUM_REAL=2000 BATCH_SIZE=32 \
#       sbatch projects/diff-EqM/slurm/jobs/sep_diag_local.sbatch
#
# Required env: CKPT_PATH (repo-relative or absolute). Optional: NUM_SAMPLES
# (default 3000; 64 for smoke), REAL_DIR, MODEL, STEPSIZE, NUM_SAMPLING_STEPS,
# BATCH_SIZE, NUM_REAL, RUN_TAG.

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

REPO_ROOT="${REPO_ROOT:-/n/home03/mkrasnow/research-repo}"
cd "$REPO_ROOT"
WORK_DIR="/tmp/sep-diag-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"

pip install timm diffusers accelerate scipy torchvision pytorch-fid matplotlib 2>&1 | tail -5

CKPT_PATH="${CKPT_PATH:?CKPT_PATH required}"
REAL_DIR="${REAL_DIR:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
MODEL="${MODEL:-EqM-B/2}"
STEPSIZE="${STEPSIZE:-0.003}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
NUM_SAMPLES="${NUM_SAMPLES:-3000}"
NUM_REAL="${NUM_REAL:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-b2_vanilla}"

# resolve ckpt: file, repo-relative file, or dir holding nested .pt (pick highest)
if [ -f "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH="$REPO_ROOT/$CKPT_PATH";
elif [ -f "$CKPT_PATH" ]; then :;
elif [ -d "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH=$(find "$REPO_ROOT/$CKPT_PATH" -name '*.pt' | sort -V | tail -1);
elif [ -d "$CKPT_PATH" ]; then CKPT_PATH=$(find "$CKPT_PATH" -name '*.pt' | sort -V | tail -1); fi
[ -n "$CKPT_PATH" ] && [ -f "$CKPT_PATH" ] || { echo "ERROR: ckpt not found: '$CKPT_PATH'"; exit 1; }
LOCAL_CKPT="$WORK_DIR/ckpt.pt"; cp "$CKPT_PATH" "$LOCAL_CKPT"
echo "=== ckpt=$CKPT_PATH ($(du -h "$LOCAL_CKPT"|cut -f1)) real_dir=$REAL_DIR repo=$REPO_ROOT ==="

OUT_DIR="$REPO_ROOT/projects/diff-EqM/experiments/separability_diagnostic/runs/$RUN_TAG"
mkdir -p "$OUT_DIR"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 20GB gpu_test cards fragment on VAE decode
SD="$REPO_ROOT/projects/diff-EqM/experiments/separability_diagnostic"
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
MASTER_PORT=$((29500 + RANDOM % 1000))

Q_GOOD="${Q_GOOD:-0.40}"
Q_GARB="${Q_GARB:-0.30}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"

if [ "$SKIP_STAGE1" = "1" ] && ls "$OUT_DIR"/logs/traj_rank*.npz >/dev/null 2>&1; then
    echo "=== STAGE 1: SKIPPED (cached shards present in $OUT_DIR/logs) $(date) ==="
else
    echo "=== STAGE 1: sample + log ($NUM_SAMPLES samples, $NUM_GPUS GPU) $(date) ==="
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        "$SD/sample_with_logging.py" \
        --model "$MODEL" --ckpt "$LOCAL_CKPT" --folder "$OUT_DIR" \
        --num-samples "$NUM_SAMPLES" --stepsize "$STEPSIZE" \
        --num-sampling-steps "$NUM_SAMPLING_STEPS" --cfg-scale 1.0 \
        --batch-size "$BATCH_SIZE" 2>&1 | tee "$WORK_DIR/stage1.log"
fi

SKIP_STAGE2="${SKIP_STAGE2:-0}"
if [ "$SKIP_STAGE2" = "1" ] && [ -f "$OUT_DIR/labels.csv" ] && [ -f "$OUT_DIR/real_latents.npz" ]; then
    echo "=== STAGE 2: SKIPPED (cached labels.csv + real_latents.npz) $(date) ==="
else
    echo "=== STAGE 2: quality labels + real latent bank (q_good=$Q_GOOD q_garb=$Q_GARB num_real=$NUM_REAL) $(date) ==="
    python "$SD/compute_quality_labels.py" \
        --folder "$OUT_DIR" --real-dir "$REAL_DIR" --num-real "$NUM_REAL" \
        --q-good "$Q_GOOD" --q-garb "$Q_GARB" \
        --batch-size "$BATCH_SIZE" --build-latent-bank 2>&1 | tee "$WORK_DIR/stage2.log"
fi

echo "=== STAGE 3: candidate scores $(date) ==="
python "$SD/compute_scores.py" --folder "$OUT_DIR" 2>&1 | tee "$WORK_DIR/stage3.log"

echo "=== STAGE 4: AUROC + matched-norm control + verdict $(date) ==="
python "$SD/analyze.py" --folder "$OUT_DIR" 2>&1 | tee "$WORK_DIR/stage4.log"

echo "=== DONE $(date) ; verdict: ==="
cat "$OUT_DIR/results/VERDICT.txt" || true
for f in stage1 stage2 stage3 stage4; do
    cp "$WORK_DIR/$f.log" "$REPO_ROOT/projects/diff-EqM/slurm/logs/sep-diag-${f}_${SLURM_JOB_ID}.log" 2>/dev/null || true
done
rm -rf "$WORK_DIR"
