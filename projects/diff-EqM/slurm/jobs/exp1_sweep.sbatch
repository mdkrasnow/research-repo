#!/bin/bash
#SBATCH --job-name=exp1-sampler-robust
#SBATCH --output=projects/diff-EqM/slurm/logs/exp1-sampler-robust_%A.out
#SBATCH --error=projects/diff-EqM/slurm/logs/exp1-sampler-robust_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Experiment 1: NFE / sampler robustness, vanilla EqM vs ANM EqM (v10). EVAL ONLY.
# Sweeps sampler{gd,ngd} x nfe{10,25,50,100,250} x step_mult{0.5,1,1.5,2} for both
# checkpoints (80 cells), writes FID/KID + AUC + plots. multi-GPU DDP -> seas_gpu
# (full A100/H200 to avoid MIG NCCL roulette per project SLURM rule).
#
# Required env: VANILLA_CKPT, ANM_CKPT   (repo-relative or absolute)
# Optional env: NUM_SAMPLES, SAMPLE_BATCH, NUM_GPUS, SMOKE (A|B), DRY_RUN(1),
#               RESUME(1), PLOTS_ONLY(1), GLOBAL_SEED, BASE_STEP, NAG_MU,
#               IMAGENET_REF, MODEL

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORK_DIR="/tmp/exp1-sampler-robust-${SLURM_JOB_ID}"
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
# OUTDIR override (OUTDIR_OVERRIDE) lets a resume run point at a persistent NFS
# dir pre-seeded with recovered metrics; default is node-local /tmp.
OUTDIR="${OUTDIR_OVERRIDE:-$WORK_DIR/exp1_out}"

# --- failsafe logging: results/ is NOT rsync --delete'd (unlike slurm/logs).
# Direct append to an NFS file -- NOT `exec > >(tee ...)`: that process-sub tee
# can die on an NFS hiccup and SIGPIPE-kill the whole job (lost run 17828606 at
# 12h56m, signal 13). A plain redirect has no pipe and cannot SIGPIPE. ---
SYNC="${SYNC_OVERRIDE:-$REPO_ROOT/projects/diff-EqM/results/exp1_sampler_robustness_${SLURM_JOB_ID}}"
mkdir -p "$SYNC"
exec >> "$SYNC/job.log" 2>&1
trap 'echo "=== EXIT trap (rc=$?) $(date) ==="' EXIT

# Resolve checkpoints (repo-relative -> absolute under REPO_ROOT if needed)
resolve_ckpt() {
    local c="$1"
    if [ -f "$c" ]; then echo "$c"; elif [ -f "$REPO_ROOT/$c" ]; then echo "$REPO_ROOT/$c";
    else echo "ERROR: checkpoint not found: $c (or $REPO_ROOT/$c)" >&2; exit 1; fi
}
VANILLA_CKPT="${VANILLA_CKPT:?set VANILLA_CKPT}"
ANM_CKPT="${ANM_CKPT:?set ANM_CKPT}"
VANILLA_CKPT="$(resolve_ckpt "$VANILLA_CKPT")"
ANM_CKPT="$(resolve_ckpt "$ANM_CKPT")"

NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch;print(torch.cuda.device_count())')}"
ARGS=(
    --vanilla-ckpt "$VANILLA_CKPT"
    --anm-ckpt "$ANM_CKPT"
    --model "${MODEL:-EqM-B/2}"
    --outdir "$OUTDIR"
    --num-samples "${NUM_SAMPLES:-50000}"
    --sample-batch-size "${SAMPLE_BATCH:-64}"
    --base-step-size "${BASE_STEP:-0.003}"
    --nag-mu "${NAG_MU:-0.3}"
    --cfg-scale "${CFG_SCALE:-1.0}"
    --imagenet-ref "${IMAGENET_REF:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
    --global-seed "${GLOBAL_SEED:-123}"
    --num-gpus "$NUM_GPUS"
)
[ -n "${SMOKE:-}" ]      && ARGS+=(--smoke "$SMOKE")
[ "${DRY_RUN:-0}" = "1" ]   && ARGS+=(--dry-run)
[ "${RESUME:-0}" = "1" ]    && ARGS+=(--resume)
[ "${PLOTS_ONLY:-0}" = "1" ] && ARGS+=(--plots-only)

echo "=== Running exp1_driver (gpus=$NUM_GPUS) ===" >&2
echo "    vanilla=$VANILLA_CKPT" >&2
echo "    anm=$ANM_CKPT" >&2
# No `| tee`: stdout already redirects to $SYNC/job.log; a tee pipe risks SIGPIPE.
set +e
python "$EXPDIR/exp1_driver.py" "${ARGS[@]}"
RC=$?
set -e

# === Sync results back (metrics/plots/manifest only; samples stay on scratch) ===
cp -r "$OUTDIR/metrics" "$OUTDIR/plots" "$SYNC/" 2>/dev/null || true
cp "$OUTDIR/eval_inputs/metadata.json" "$SYNC/" 2>/dev/null || true
cp exp1.log "$SYNC/" 2>/dev/null || true
cp exp1.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/exp1-sampler-robust_${SLURM_JOB_ID}.log" 2>/dev/null || true

echo "=== Done (rc=$RC). Synced to $SYNC ===" >&2
rm -rf "$WORK_DIR"
exit $RC
