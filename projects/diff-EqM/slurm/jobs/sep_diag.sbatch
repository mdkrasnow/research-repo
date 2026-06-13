#!/bin/bash
#SBATCH --job-name=sep-diag
#SBATCH --output=slurm/logs/sep-diag_%A.out
#SBATCH --error=slurm/logs/sep-diag_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# EqM Separability Diagnostic -- runs all 4 stages in one GPU job.
#   Stage 1 sample_with_logging.py  : sample + log per-step signals + decode PNGs
#   Stage 2 compute_quality_labels  : independent good/garbage labels + real latent bank
#   Stage 3 compute_scores          : 5 candidate scores at the stopping point
#   Stage 4 analyze                 : AUROC + matched-norm control + VERDICT
#
# Required env: CKPT_PATH (vanilla EqM-B/2 ckpt, repo-relative or absolute)
# Optional: NUM_SAMPLES (default 3000; set 200 for a smoke), REAL_DIR, MODEL,
#           STEPSIZE, NUM_SAMPLING_STEPS, BATCH_SIZE, NUM_REAL, RUN_TAG, GIT_SHA
#
# Smoke first:  NUM_SAMPLES=200 NUM_REAL=2000 ... sbatch sep_diag.sbatch

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

WORK_DIR="/tmp/sep-diag-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?GIT_SHA required (HEAD fails on fresh clone)}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"
git init . && git remote add origin "$GIT_URL"
git fetch --depth=50 origin main 2>&1 | tail -3
git checkout "$GIT_SHA" 2>&1 | tail -3 || { git fetch --unshallow origin 2>&1 | tail -3; git checkout "$GIT_SHA"; }

pip install timm diffusers accelerate scipy torchvision pytorch-fid matplotlib 2>&1 | tail -5

REPO_ROOT="/n/home03/mkrasnow/research-repo"
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
echo "=== ckpt=$CKPT_PATH ($(du -h "$LOCAL_CKPT"|cut -f1)) real_dir=$REAL_DIR ==="

OUT_DIR="$REPO_ROOT/projects/diff-EqM/experiments/separability_diagnostic/runs/$RUN_TAG"
mkdir -p "$OUT_DIR"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
SD="projects/diff-EqM/experiments/separability_diagnostic"
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "=== STAGE 1: sample + log ($NUM_SAMPLES samples, $NUM_GPUS GPU) $(date) ==="
python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    $SD/sample_with_logging.py \
    --model "$MODEL" --ckpt "$LOCAL_CKPT" --folder "$OUT_DIR" \
    --num-samples "$NUM_SAMPLES" --stepsize "$STEPSIZE" \
    --num-sampling-steps "$NUM_SAMPLING_STEPS" --cfg-scale 1.0 \
    --batch-size "$BATCH_SIZE" 2>&1 | tee stage1.log

echo "=== STAGE 2: quality labels + real latent bank $(date) ==="
python $SD/compute_quality_labels.py \
    --folder "$OUT_DIR" --real-dir "$REAL_DIR" --num-real "$NUM_REAL" \
    --batch-size "$BATCH_SIZE" --build-latent-bank 2>&1 | tee stage2.log

echo "=== STAGE 3: candidate scores $(date) ==="
python $SD/compute_scores.py --folder "$OUT_DIR" 2>&1 | tee stage3.log

echo "=== STAGE 4: AUROC + matched-norm control + verdict $(date) ==="
python $SD/analyze.py --folder "$OUT_DIR" 2>&1 | tee stage4.log

echo "=== DONE $(date) ; verdict: ==="
cat "$OUT_DIR/results/VERDICT.txt" || true
for f in stage1.log stage2.log stage3.log stage4.log; do
    cp "$f" "$REPO_ROOT/projects/diff-EqM/slurm/logs/sep-diag-${f%.log}_${SLURM_JOB_ID}.log" 2>/dev/null || true
done
rm -rf "$WORK_DIR"
