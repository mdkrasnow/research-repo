#!/bin/bash
#SBATCH --job-name=eqm-grad-guard
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/grad-guard-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/grad-guard-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320GB
#SBATCH --time=01:00:00
#SBATCH --partition=seas_gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:4
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT="${CKPT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer40_direct_seed0_ckpt50k_job36359213/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/1500000.pt}"
MODE="${MODE:-calibration}"
MAX_STEPS="${MAX_STEPS:-2000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_gradient_guard}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
WORK_DIR="/tmp/masked-eqm-grad-guard-${SLURM_JOB_ID}"
RESULTS_DIR="$OUTPUT_ROOT/${MODE}_job${SLURM_JOB_ID}"
MASTER_PORT=$((29500 + RANDOM % 1000))

mkdir -p "$WORK_DIR" "$RESULTS_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchdiffeq wandb torchvision

CLIP_ARGS=()
if [[ "$MODE" == "clipped" ]]; then
    CALIBRATION_JSON="${CALIBRATION_JSON:?set CALIBRATION_JSON for clipped mode}"
    MAX_GRAD_NORM=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_grad_norm"])' "$CALIBRATION_JSON")
    CLIP_ARGS=(--max-grad-norm "$MAX_GRAD_NORM")
elif [[ "$MODE" != "calibration" ]]; then
    echo "Unknown MODE=$MODE" >&2
    exit 2
fi

CMD=(python -m torch.distributed.run --nnodes=1 --nproc_per_node=4
    --master_port="$MASTER_PORT" train.py
    --model EqM-B/2 --data-path "$IMAGENET_PATH" --num-classes 1000
    --global-batch-size 32 --epochs 39 --save-epochs 39 --image-size 256
    --log-every 50 --grad-log-every 1 --ckpt-every 999999999
    --max-steps "$MAX_STEPS" --results-dir "$RESULTS_DIR"
    --ebm direct --global-seed 0 --ckpt "$CKPT" --num-workers 4
    "${CLIP_ARGS[@]}")
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee train.log

METRICS=$(find "$RESULTS_DIR" -name gradient_metrics.jsonl -type f -print -quit)
test -n "$METRICS"
cp "$METRICS" "$RESULTS_DIR/gradient_metrics.jsonl"
if [[ "$MODE" == "calibration" ]]; then
    python experiments/direct_energy/calibrate_gradient_clip.py \
        "$RESULTS_DIR/gradient_metrics.jsonl" \
        --output "$RESULTS_DIR/calibration.json"
fi
rm -rf "$WORK_DIR"
