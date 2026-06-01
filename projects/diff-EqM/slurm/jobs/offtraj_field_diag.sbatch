#!/bin/bash
#SBATCH --job-name=eqm-offtraj-diag
#SBATCH --output=slurm/logs/eqm-offtraj-diag_%A.out
#SBATCH --error=slurm/logs/eqm-offtraj-diag_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Experiment 2: off-trajectory local field robustness — vanilla EqM vs ANM (v10).
# Single-GPU field-level diagnostic. Loads two matched-step checkpoints + a frozen
# probe, evaluates field MSE/cosine/norm calibration at perturbed interpolation
# points, writes per-sample + aggregate metrics + plots.
#
# Required env vars:
#   VANILLA_CKPT   repo-relative or absolute path to vanilla EqM-B/2 ckpt
#   ANM_CKPT       repo-relative or absolute path to v10 EqM-B/2 ckpt
# Optional env vars (defaults match the script):
#   PROBE_CKPT (default=VANILLA_CKPT), OUTPUT_DIR, VAL_PATH, MODEL,
#   NUM_BATCHES, BATCH_SIZE, T_VALUES, RADII, PERTURB_TYPE, PRECISION,
#   SEED, DIRECTION_SEED, EXTRA_ARGS, RESUME ("1" to add --resume)

set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/eqm-offtraj-diag-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== Cloning repo at SHA ${GIT_SHA} ===" >&2
git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -10 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || {
    echo "SHA not in last 50 commits, fetching full..." >&2
    git fetch --unshallow origin 2>&1 | tail -5
    git checkout "${GIT_SHA}" 2>&1 | tail -5 || exit 1
}

echo "=== Installing dependencies ===" >&2
pip install timm diffusers accelerate scipy torchvision torchdiffeq matplotlib 2>&1 | tail -10

# ---- Configuration ----
REPO_ROOT="/n/home03/mkrasnow/research-repo"
VANILLA_CKPT="${VANILLA_CKPT:?ERROR: VANILLA_CKPT must be set}"
ANM_CKPT="${ANM_CKPT:?ERROR: ANM_CKPT must be set}"
PROBE_CKPT="${PROBE_CKPT:-$VANILLA_CKPT}"
VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
MODEL="${MODEL:-EqM-B/2}"
OUTPUT_DIR="${OUTPUT_DIR:-projects/diff-EqM/results/diagnostics/offtraj_${SLURM_JOB_ID}}"
NUM_BATCHES="${NUM_BATCHES:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
T_VALUES="${T_VALUES:-0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95}"
RADII="${RADII:-0,0.0025,0.005,0.01,0.02,0.05,0.10}"
PERTURB_TYPE="${PERTURB_TYPE:-random_l2_orthogonal}"
PRECISION="${PRECISION:-fp32}"
SEED="${SEED:-123}"
DIRECTION_SEED="${DIRECTION_SEED:-456}"
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

# Resolve checkpoint paths (allow repo-relative)
resolve() { [ -f "$1" ] && echo "$1" || echo "$REPO_ROOT/$1"; }
VANILLA_CKPT="$(resolve "$VANILLA_CKPT")"
ANM_CKPT="$(resolve "$ANM_CKPT")"
PROBE_CKPT="$(resolve "$PROBE_CKPT")"

# Persistent output dir on home filesystem
ABS_OUT="$REPO_ROOT/$OUTPUT_DIR"
mkdir -p "$ABS_OUT"

echo "=== Running off-trajectory field diagnostic ===" >&2
echo "  vanilla: $VANILLA_CKPT" >&2
echo "  anm:     $ANM_CKPT" >&2
echo "  probe:   $PROBE_CKPT" >&2
echo "  out:     $ABS_OUT" >&2
echo "  perturb: $PERTURB_TYPE  radii=$RADII  N=$((NUM_BATCHES*BATCH_SIZE))" >&2

python projects/diff-EqM/experiments/diagnostics/offtraj_field_robustness.py \
    --vanilla-checkpoint "$VANILLA_CKPT" \
    --anm-checkpoint "$ANM_CKPT" \
    --probe-checkpoint "$PROBE_CKPT" \
    --val-path "$VAL_PATH" \
    --model "$MODEL" \
    --output-dir "$ABS_OUT" \
    --num-batches "$NUM_BATCHES" \
    --batch-size "$BATCH_SIZE" \
    --t-values "$T_VALUES" \
    --perturbation-radii "$RADII" \
    --perturbation-type "$PERTURB_TYPE" \
    --precision "$PRECISION" \
    --seed "$SEED" \
    --direction-seed "$DIRECTION_SEED" \
    $RESUME_FLAG \
    ${EXTRA_ARGS:-} \
    2>&1 | tee diag.log
DIAG_EXIT=${PIPESTATUS[0]}

cp diag.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/offtraj_diag_${SLURM_JOB_ID}.log" 2>/dev/null || true
echo "=== Diagnostic exit code: $DIAG_EXIT, time: $(date) ===" >&2
exit $DIAG_EXIT
