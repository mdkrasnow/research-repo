#!/bin/bash
#SBATCH --job-name=eqm-c3-ood
#SBATCH --output=slurm/logs/eqm-c3-ood_%A.out
#SBATCH --error=slurm/logs/eqm-c3-ood_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# C3 — OOD detection via EqM field-residual ||f(x,y)||.
# Single-GPU, forward-only (no sampler, no decode). 3 frozen arms:
# vanilla / v10 lambda=0.1 / v10 lambda=0.3. Dependency-free OOD from val data.
#
# Optional env: VANILLA_CKPT, ANM_L01_CKPT, ANM_L03_CKPT (repo-relative dirs OK;
# *.pt resolved recursively), VAL_PATH, N_SAMPLES, OOD_TYPES, OUTPUT_DIR, SEED.

set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/eqm-c3-ood-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"; cd "$WORK_DIR"

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
pip install timm diffusers accelerate scipy torchvision 2>&1 | tail -10

REPO_ROOT="/n/home03/mkrasnow/research-repo"
VANILLA_CKPT="${VANILLA_CKPT:-projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt}"
ANM_L01_CKPT="${ANM_L01_CKPT:-projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt}"
ANM_L03_CKPT="${ANM_L03_CKPT:-projects/diff-EqM/results/imagenet1k_80ep_v10_b2_lambda03_seed0}"
VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
N_SAMPLES="${N_SAMPLES:-2000}"
OOD_TYPES="${OOD_TYPES:-gaussian,uniform,wrong_label,patch_shuffle}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-projects/diff-EqM/results/c3_ood_energy/run_${SLURM_JOB_ID}}"

# Resolve checkpoints: accept absolute, repo-relative file, or a dir holding *.pt
# (prefer EMA-bearing 'final.pt', else newest checkpoints/*.pt). Mirrors exp3's
# recursive resolve for the lambda=0.3 bare-dir layout.
resolve_ckpt() {
    local p="$1"
    [ -f "$p" ] && { echo "$p"; return; }
    [ -f "$REPO_ROOT/$p" ] && { echo "$REPO_ROOT/$p"; return; }
    local d="$p"; [ -d "$d" ] || d="$REPO_ROOT/$p"
    if [ -d "$d" ]; then
        local hit
        hit="$(find "$d" -name 'final.pt' 2>/dev/null | head -1)"
        [ -z "$hit" ] && hit="$(find "$d" -path '*/checkpoints/*.pt' 2>/dev/null | sort | tail -1)"
        [ -n "$hit" ] && { echo "$hit"; return; }
    fi
    echo "RESOLVE_FAILED:$p"
}
VANILLA_CKPT="$(resolve_ckpt "$VANILLA_CKPT")"
ANM_L01_CKPT="$(resolve_ckpt "$ANM_L01_CKPT")"
ANM_L03_CKPT="$(resolve_ckpt "$ANM_L03_CKPT")"

ABS_OUT="$REPO_ROOT/$OUTPUT_DIR"; mkdir -p "$ABS_OUT"

echo "=== C3 OOD-energy detection ===" >&2
echo "  vanilla : $VANILLA_CKPT" >&2
echo "  v10_l01 : $ANM_L01_CKPT" >&2
echo "  v10_l03 : $ANM_L03_CKPT" >&2
echo "  val     : $VAL_PATH  N=$N_SAMPLES  ood=$OOD_TYPES" >&2
echo "  out     : $ABS_OUT" >&2

# Skip any arm whose ckpt failed to resolve (don't hard-fail the whole job).
CKPTS="vanilla:$VANILLA_CKPT"
case "$ANM_L01_CKPT" in RESOLVE_FAILED:*) echo "[WARN] v10_l01 unresolved, skipping" >&2;; *) CKPTS="$CKPTS,v10_l01:$ANM_L01_CKPT";; esac
case "$ANM_L03_CKPT" in RESOLVE_FAILED:*) echo "[WARN] v10_l03 unresolved, skipping" >&2;; *) CKPTS="$CKPTS,v10_l03:$ANM_L03_CKPT";; esac

python projects/diff-EqM/experiments/c3_ood_energy/ood_energy_detect.py \
    --checkpoints "$CKPTS" \
    --val-path "$VAL_PATH" \
    --n-samples "$N_SAMPLES" \
    --ood-types "$OOD_TYPES" \
    --seed "$SEED" \
    --output-dir "$ABS_OUT" \
    ${EXTRA_ARGS:-} \
    2>&1 | tee c3.log
C3_EXIT=${PIPESTATUS[0]}

cp c3.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/c3_ood_${SLURM_JOB_ID}.log" 2>/dev/null || true
cp "$ABS_OUT/auroc.csv" "$REPO_ROOT/projects/diff-EqM/documentation/c3_data_auroc_${SLURM_JOB_ID}.csv" 2>/dev/null || true
cp "$ABS_OUT/verdict.json" "$REPO_ROOT/projects/diff-EqM/documentation/c3_data_verdict_${SLURM_JOB_ID}.json" 2>/dev/null || true
echo "=== C3 exit code: $C3_EXIT, time: $(date) ===" >&2
exit $C3_EXIT
