#!/bin/bash
#SBATCH --job-name=eqm-energy-monotonicity
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/energy-monotonicity-%A_%a.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/energy-monotonicity-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

REPO_ROOT="/n/home03/mkrasnow/research-repo"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?set GIT_SHA}"
MODE="${MODE:?set MODE to prepare, evaluate, or aggregate}"
OUTPUT_DIR="${OUTPUT_DIR:-/n/home03/mkrasnow/energy_monotonicity_eval}"
DATA_PATH="${DATA_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/val}"
EPOCHS="${EPOCHS:-1 5 8}"
NUM_IMAGES="${NUM_IMAGES:-2048}"
NOISES_PER_IMAGE="${NOISES_PER_IMAGE:-2}"
NUM_GAMMA_POINTS="${NUM_GAMMA_POINTS:-21}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-12345}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-23456}"
FORCE="${FORCE:-0}"

WORK_DIR="$(mktemp -d "/tmp/masked-eqm-energy-${SLURM_JOB_ID}-XXXXXX")"
cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

git -C "$WORK_DIR" init
git -C "$WORK_DIR" remote add origin "$GIT_URL"
git -C "$WORK_DIR" fetch --depth=50 origin main
git -C "$WORK_DIR" checkout "$GIT_SHA" || {
  git -C "$WORK_DIR" fetch --unshallow origin
  git -C "$WORK_DIR" checkout "$GIT_SHA"
}
cd "$WORK_DIR/projects/masked-EqM"
pip install --quiet timm diffusers accelerate torchdiffeq torchvision pandas pyarrow matplotlib seaborn

COMMON=(
  python -m energy_monotonicity.evaluate_energy_monotonicity
  --none-run configs/energy_monotonicity_none.json
  --dot-run configs/energy_monotonicity_dot.json
  --direct-run configs/energy_monotonicity_direct.json
  --data-path "$DATA_PATH"
  --epochs $EPOCHS
  --num-images "$NUM_IMAGES"
  --noises-per-image "$NOISES_PER_IMAGE"
  --num-gamma-points "$NUM_GAMMA_POINTS"
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES"
  --seed "$SEED"
  --bootstrap-seed "$BOOTSTRAP_SEED"
  --batch-size "$BATCH_SIZE"
  --output-dir "$OUTPUT_DIR"
)
if [[ "$FORCE" == "1" ]]; then
  COMMON+=(--force)
fi

case "$MODE" in
  prepare)
    "${COMMON[@]}" --prepare-only
    ;;
  evaluate)
    : "${EVAL_VARIANT:?set EVAL_VARIANT}"
    : "${EVAL_EPOCH:?set EVAL_EPOCH}"
    "${COMMON[@]}" --evaluate-only "$EVAL_VARIANT" "$EVAL_EPOCH"
    ;;
  aggregate)
    "${COMMON[@]}" --aggregate-only
    ;;
  *)
    echo "unknown MODE=$MODE" >&2
    exit 2
    ;;
esac
