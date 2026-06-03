#!/bin/bash
#SBATCH --job-name=eqm-c1-ext
#SBATCH --output=projects/diff-EqM/slurm/logs/eqm-c1-ext_%A.out
#SBATCH --error=projects/diff-EqM/slurm/logs/eqm-c1-ext_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# C1 extension — inference-compute scaling. Extends Exp 1 into the untested
# high-NFE regime and adds the lambda=0.3 dose arm. SINGLE-GPU (MIG-safe).
# Question: does ANM keep improving as NFE grows where vanilla plateaus/overshoots,
# and does ANM reach a lower equilibrium residual (mean_final_grad_norm)?
#
# ARM_MODE selects the dose arm and whether vanilla is (re)computed:
#   l01 -> vanilla + v10 lambda=0.1   (tag anm_l01)            [the heavier job]
#   l03 -> v10 lambda=0.3 only        (tag anm_l03, --skip-vanilla)
# Submit BOTH (parallel, separate outdirs -> no shared-CSV race); merge CSVs after.
#
# Optional env: NFE_LIST(250,500,1000), STEP_MULTS(1.0), SAMPLERS(gd,ngd),
#   NUM_SAMPLES(5000), GLOBAL_SEED(123), RESUME(1).

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORK_DIR="/tmp/eqm-c1-ext-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"; cd "$WORK_DIR"

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
pip install timm diffusers accelerate scipy torchvision pytorch-fid pandas matplotlib pyyaml 2>&1 | tail -5

REPO_ROOT="/n/home03/mkrasnow/research-repo"
EXPDIR="projects/diff-EqM/experiments/exp1_sampler_robustness"

ARM_MODE="${ARM_MODE:?set ARM_MODE=l01|l03}"
VANILLA_CKPT="${VANILLA_CKPT:-projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt}"
L01_CKPT="${L01_CKPT:-projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt}"
L03_CKPT="${L03_CKPT:-projects/diff-EqM/results/imagenet1k_80ep_v10_b2_lambda03_seed0}"

# Resolve a checkpoint: absolute file, repo-relative file, or a dir holding *.pt
# (prefer final.pt, else newest checkpoints/*.pt — handles lambda=0.3 bare dir).
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
    echo "ERROR: checkpoint not found: $p" >&2; exit 1
}
VANILLA_CKPT="$(resolve_ckpt "$VANILLA_CKPT")"

case "$ARM_MODE" in
    l01) ANM_CKPT="$(resolve_ckpt "$L01_CKPT")"; ANM_TAG="anm_l01"; SKIP_VANILLA=0 ;;
    l03) ANM_CKPT="$(resolve_ckpt "$L03_CKPT")"; ANM_TAG="anm_l03"; SKIP_VANILLA=1 ;;
    *) echo "ERROR: ARM_MODE must be l01 or l03" >&2; exit 2 ;;
esac

SYNC="$REPO_ROOT/projects/diff-EqM/results/c1_ext_scaling_${ARM_MODE}_${SLURM_JOB_ID}"
mkdir -p "$SYNC"
exec >> "$SYNC/job.log" 2>&1
trap 'echo "=== EXIT trap (rc=$?) $(date) ==="' EXIT
OUTDIR="$WORK_DIR/c1_out"

ARGS=(
    --vanilla-ckpt "$VANILLA_CKPT"
    --anm-ckpt "$ANM_CKPT"
    --anm-tag "$ANM_TAG"
    --model "${MODEL:-EqM-B/2}"
    --outdir "$OUTDIR"
    --num-samples "${NUM_SAMPLES:-5000}"
    --sample-batch-size "${SAMPLE_BATCH:-64}"
    --base-step-size "${BASE_STEP:-0.003}"
    --nag-mu "${NAG_MU:-0.3}"
    --cfg-scale "${CFG_SCALE:-1.0}"
    --imagenet-ref "${IMAGENET_REF:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
    --global-seed "${GLOBAL_SEED:-123}"
    --num-gpus 1
    --nfe-list "${NFE_LIST:-250,500,1000}"
    --step-mults "${STEP_MULTS:-1.0}"
    --samplers "${SAMPLERS:-gd,ngd}"
)
[ "$SKIP_VANILLA" = "1" ] && ARGS+=(--skip-vanilla)
[ "${RESUME:-0}" = "1" ]  && ARGS+=(--resume)

echo "=== C1 ext (ARM_MODE=$ARM_MODE tag=$ANM_TAG skip_vanilla=$SKIP_VANILLA) ===" >&2
echo "    vanilla=$VANILLA_CKPT" >&2
echo "    anm=$ANM_CKPT" >&2
echo "    nfe=${NFE_LIST:-250,500,1000} step=${STEP_MULTS:-1.0} N=${NUM_SAMPLES:-5000}" >&2

set +e
python "$EXPDIR/exp1_driver.py" "${ARGS[@]}"
RC=$?
set -e

cp -r "$OUTDIR/metrics" "$SYNC/" 2>/dev/null || true
cp "$OUTDIR/metrics/results.csv" \
   "$REPO_ROOT/projects/diff-EqM/documentation/c1_ext_${ARM_MODE}_results_${SLURM_JOB_ID}.csv" 2>/dev/null || true
echo "=== Done (rc=$RC). Synced to $SYNC ===" >&2
rm -rf "$WORK_DIR"
exit $RC
