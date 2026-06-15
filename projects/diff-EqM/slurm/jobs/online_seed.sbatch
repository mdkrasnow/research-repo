#!/bin/bash
#SBATCH --job-name=online-seed
#SBATCH --output=slurm/logs/online-seed_%A.out
#SBATCH --error=slurm/logs/online-seed_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# Online adaptive sampler — ONE seed at 50k. Submit 3 (START_SEED=0/1/2) in parallel
# to promote claim #5 to 50k/3-seed. Each writes a distinct out dir + per-seed FID.
#   START_SEED=0 CKPT_PATH=<...> sbatch projects/diff-EqM/slurm/jobs/online_seed.sbatch
# Required: CKPT_PATH, START_SEED.

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "no gpu"; exit 1; }
REPO_ROOT="${REPO_ROOT:-/n/home03/mkrasnow/research-repo}"
cd "$REPO_ROOT"
WORK="/tmp/online-seed-${SLURM_JOB_ID}"; mkdir -p "$WORK"
pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -3
CKPT_PATH="${CKPT_PATH:?CKPT_PATH required}"
START_SEED="${START_SEED:?START_SEED required}"
REAL_DIR="${REAL_DIR:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
REF_STATS="${REF_STATS:-$REPO_ROOT/projects/diff-EqM/results/in1k_reference_stats.npz}"
NUM_SLOTS="${NUM_SLOTS:-50000}"
FLAG_FRAC="${FLAG_FRAC:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-b2_vanilla}"
SD="$REPO_ROOT/projects/diff-EqM/experiments/separability_diagnostic"
OUT_DIR="$SD/runs/$RUN_TAG"
PARTIAL_PROBE="${PARTIAL_PROBE:-$OUT_DIR/results/partial_probe/partial_probe_k100.npz}"
SOUT="$OUT_DIR/online50k_seed${START_SEED}"; mkdir -p "$SOUT"

if [ -f "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH="$REPO_ROOT/$CKPT_PATH";
elif [ -f "$CKPT_PATH" ]; then :;
elif [ -d "$REPO_ROOT/$CKPT_PATH" ]; then CKPT_PATH=$(find "$REPO_ROOT/$CKPT_PATH" -name '*.pt' | sort -V | tail -1);
elif [ -d "$CKPT_PATH" ]; then CKPT_PATH=$(find "$CKPT_PATH" -name '*.pt' | sort -V | tail -1); fi
[ -f "$CKPT_PATH" ] || { echo "ckpt not found"; exit 1; }
LOCAL_CKPT="$WORK/ckpt.pt"; cp "$CKPT_PATH" "$LOCAL_CKPT"
[ -f "$PARTIAL_PROBE" ] || { echo "partial probe missing"; exit 1; }
SHARED_BANK="$OUT_DIR/online50k_real_bank.npy"
[ -f "$SHARED_BANK" ] && cp "$SHARED_BANK" "$SOUT/real_bank.npy" 2>/dev/null || true

export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
PORT=$((29500 + RANDOM % 1000))
echo "=== ONLINE 50k seed=$START_SEED slots=$NUM_SLOTS $(date) ==="
python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$PORT \
    "$SD/online_adaptive_sampler.py" \
    --model EqM-B/2 --ckpt "$LOCAL_CKPT" --partial-probe "$PARTIAL_PROBE" --out "$SOUT" \
    --real-dir "$REAL_DIR" --num-real-bank 10000 --num-slots "$NUM_SLOTS" \
    --flag-frac "$FLAG_FRAC" --k-dec -1 --stepsize 0.003 --num-sampling-steps 250 \
    --batch-size "$BATCH_SIZE" --seed "$START_SEED" 2>&1 | tee "$WORK/online.log"
[ -f "$SOUT/real_bank.npy" ] && cp -n "$SOUT/real_bank.npy" "$SHARED_BANK" 2>/dev/null || true
python "$SD/fid_online_agg.py" --out "$SOUT" --ref-stats "$REF_STATS" 2>&1 | tee "$WORK/agg.log"
cat "$SOUT/ONLINE_FID.txt" || true
cp "$WORK"/*.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/" 2>/dev/null || true
rm -rf "$WORK"
echo "=== DONE seed=$START_SEED $(date) ==="
