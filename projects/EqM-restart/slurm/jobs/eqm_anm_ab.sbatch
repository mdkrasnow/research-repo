#!/bin/bash
#SBATCH --job-name=eqm-anm-ab
#SBATCH --output=projects/EqM-restart/slurm/logs/eqm-anm-ab_%A.out
#SBATCH --error=projects/EqM-restart/slurm/logs/eqm-anm-ab_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Quick A/B: vanilla EqM vs ANM EqM on the fresh EqM-restart code.
# One arm per job (submit twice: ARM=vanilla and ARM=anm). Each job:
#   1) train ~MAX_MINUTES on the chosen (smallest) model, save final.pt
#   2) sample NUM_FID images from final.pt (identical sampler settings both arms)
#   3) compute FID with pytorch-fid against a subsampled ImageNet ref
#   4) write one-line FID result, sync back to the repo
#
# CONFOUND (read the result with this in mind): ANM costs ~ (anm-steps+1)x forward
# per step, so at equal WALL-CLOCK the ANM arm takes fewer optimizer steps. We log
# the step count both arms reach (grep "step=" in train.log) so the FID delta can be
# read against steps-taken, not just wall-clock.
#
# Required env: ARM=vanilla|anm
# Optional env: MODEL, GLOBAL_BATCH, MAX_MINUTES, NUM_FID, SAMPLE_BATCH,
#               ANM_STEPS, ANM_STEP_SIZE, ANM_EPS_BALL

set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

export PYTHONUNBUFFERED=1            # live step/disp_ratio logs (fresh train.py has no -u)

ARM="${ARM:?ERROR: set ARM=vanilla or ARM=anm}"
MODEL="${MODEL:-EqM-S/8}"            # smallest-compute config: fewest tokens -> most steps/hr
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
# EQUAL-STEP budget (fair vs ANM's higher per-step cost). MAX_MINUTES is a backstop only.
MAX_STEPS="${MAX_STEPS:-3000}"
MAX_MINUTES="${MAX_MINUTES:-100}"
NUM_FID="${NUM_FID:-5000}"           # quick FID; not publishable, just signal
SAMPLE_BATCH="${SAMPLE_BATCH:-250}"
ANM_STEPS="${ANM_STEPS:-2}"          # keep overhead ~3x for the quick run
ANM_STEP_SIZE="${ANM_STEP_SIZE:-0.15}"
ANM_EPS_BALL="${ANM_EPS_BALL:-0.5}"

WORK_DIR="/tmp/eqm-anm-ab-${ARM}-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== Cloning repo at SHA ${GIT_SHA} ===" >&2
git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -5 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -3 || {
    git fetch --unshallow origin 2>&1 | tail -3
    git checkout "${GIT_SHA}" 2>&1 | tail -3 || exit 1
}

echo "=== Installing dependencies ===" >&2
pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -5

IMAGENET_PATH="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
RESULTS_DIR="projects/EqM-restart/results/eqm_anm_ab/${ARM}"
SAMPLES_DIR="$WORK_DIR/samples_${ARM}"
SYNC_DIR="$REPO_ROOT/projects/EqM-restart/results/eqm_anm_ab/${ARM}"

[ -d "$IMAGENET_PATH" ] || { echo "ERROR: ImageNet not found at $IMAGENET_PATH" >&2; exit 1; }
mkdir -p "$WORK_DIR/$RESULTS_DIR" "$SAMPLES_DIR" "$SYNC_DIR"

# ---- ANM flags only for the anm arm ----
ANM_FLAGS=""
if [ "$ARM" = "anm" ]; then
    ANM_FLAGS="--anm-steps $ANM_STEPS --anm-step-size $ANM_STEP_SIZE --anm-eps-ball $ANM_EPS_BALL"
fi
echo "=== ARM=$ARM  MODEL=$MODEL  ANM_FLAGS='${ANM_FLAGS:-none}' ===" >&2

MASTER_PORT=$((29500 + RANDOM % 1000))

# ============================ STEP 1: TRAIN ============================
echo "=== [1/3] Training $ARM for ${MAX_MINUTES} min  (start $(date)) ===" >&2
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=$MASTER_PORT \
    projects/EqM-restart/train.py \
    --model "$MODEL" \
    --data-path "$IMAGENET_PATH" \
    --num-classes 1000 \
    --global-batch-size "$GLOBAL_BATCH" \
    --image-size 256 \
    --epochs 1000 \
    --max-steps "$MAX_STEPS" \
    --max-minutes "$MAX_MINUTES" \
    --ckpt-every 1000 \
    --log-every 50 \
    --results-dir "$RESULTS_DIR" \
    $ANM_FLAGS \
    2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Train exit=$TRAIN_EXIT  ($(date)) ===" >&2
[ $TRAIN_EXIT -eq 0 ] || { cp train.log "$SYNC_DIR/train.log" 2>/dev/null || true; exit $TRAIN_EXIT; }

STEPS_REACHED=$(grep -oE 'step=[0-9]+' train.log | tail -1 | grep -oE '[0-9]+' || echo "unknown")
echo "=== $ARM reached step $STEPS_REACHED ===" >&2

# locate checkpoint: prefer final.pt, else newest *.pt
EXP_DIR=$(find "$WORK_DIR/$RESULTS_DIR" -type d -name 'checkpoints' | head -1)
CKPT="$EXP_DIR/final.pt"
if [ ! -f "$CKPT" ]; then
    CKPT=$(find "$EXP_DIR" -name '*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
fi
[ -f "$CKPT" ] || { echo "ERROR: no checkpoint produced" >&2; cp train.log "$SYNC_DIR/train.log"; exit 1; }
echo "=== Using checkpoint: $CKPT ($(du -h "$CKPT" | cut -f1)) ===" >&2

# ============================ STEP 2: SAMPLE ============================
echo "=== [2/3] Sampling $NUM_FID images ($(date)) ===" >&2
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=$((MASTER_PORT+1)) \
    projects/EqM-restart/sample_gd.py \
    --model "$MODEL" \
    --ckpt "$CKPT" \
    --num-classes 1000 \
    --sampler gd \
    --num-sampling-steps 250 \
    --cfg-scale 1.0 \
    --global-batch-size "$SAMPLE_BATCH" \
    --num-fid-samples "$NUM_FID" \
    --folder "$SAMPLES_DIR" \
    2>&1 | tee sample.log
SAMPLE_EXIT=${PIPESTATUS[0]}
echo "=== Sample exit=$SAMPLE_EXIT ($(date)) ===" >&2
[ $SAMPLE_EXIT -eq 0 ] || { cp sample.log "$SYNC_DIR/sample.log"; exit $SAMPLE_EXIT; }

# ============================ STEP 3: FID ============================
echo "=== [3/3] Building flat reference + FID ($(date)) ===" >&2
FLAT_REF="$WORK_DIR/ref_flat"
mkdir -p "$FLAT_REF"
find "$IMAGENET_PATH" -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) \
    | shuf -n "$NUM_FID" | while read f; do
        ln -s "$f" "$FLAT_REF/$(basename "$(dirname "$f")")_$(basename "$f")"
    done
echo "=== ref images: $(ls "$FLAT_REF" | wc -l) ===" >&2

FID=$(python -m pytorch_fid "$SAMPLES_DIR" "$FLAT_REF" --device cuda 2>&1 | tee fid.log | grep -oE 'FID:.*' || echo "FID: PARSE_FAIL")
echo "=== $ARM RESULT  $FID  (model=$MODEL steps=$STEPS_REACHED num_fid=$NUM_FID) ===" >&2

# ============================ SYNC ============================
{
  echo "arm=$ARM"
  echo "model=$MODEL"
  echo "steps_reached=$STEPS_REACHED"
  echo "num_fid=$NUM_FID"
  echo "anm_flags=${ANM_FLAGS:-none}"
  echo "$FID"
  echo "job_id=$SLURM_JOB_ID git_sha=$GIT_SHA date=$(date -u +%FT%TZ)"
} > "$SYNC_DIR/RESULT.txt"
cp train.log sample.log fid.log "$SYNC_DIR/" 2>/dev/null || true
cp "$CKPT" "$SYNC_DIR/final.pt" 2>/dev/null || true

echo "=== DONE arm=$ARM ($(date)) ===" >&2
rm -rf "$WORK_DIR"
