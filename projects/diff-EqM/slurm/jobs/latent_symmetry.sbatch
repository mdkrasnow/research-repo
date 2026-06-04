#!/bin/bash
#SBATCH --job-name=eqm-latent-sym
#SBATCH --output=slurm/logs/latent-sym_%A.out
#SBATCH --error=slurm/logs/latent-sym_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1

# Ladder rung 3: latent symmetry hidden by a nonlinear decoder.
# Tiny toy (D=16, MLPs). Compute = seconds; auto-falls back to CPU if the
# GPU/torch build mismatches. Single-GPU -> MIG-safe on gpu_requeue.
# Emits a result table; persists results.json + run.log to HOME (logs can vanish).

set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

WORK_DIR="/tmp/eqm-latent-sym-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"; cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== clone repo @ ${GIT_SHA} ===" >&2
git init . >/dev/null 2>&1
git remote add origin "$GIT_URL"
git fetch --depth=50 origin main 2>&1 | tail -3 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -3 || {
    git fetch --unshallow origin 2>&1 | tail -3
    git checkout "${GIT_SHA}" 2>&1 | tail -3 || exit 1
}

echo "=== deps ===" >&2
pip install --quiet torch numpy 2>&1 | tail -3 || pip install --quiet --user torch numpy 2>&1 | tail -3

# persistent home output (survives /tmp cleanup + vanished SLURM logs)
OUT_HOME="/n/home03/mkrasnow/research-repo/projects/diff-EqM/results/latent_symmetry"
mkdir -p "$OUT_HOME"
OUT_JSON="$OUT_HOME/results_${SLURM_JOB_ID}.json"
RUN_LOG="$OUT_HOME/run_${SLURM_JOB_ID}.log"

echo "=== GPU (informational; CPU fallback OK) ===" >&2
nvidia-smi 2>&1 | head -15 >&2 || echo "no nvidia-smi" >&2

echo "=== run ===" >&2
python projects/diff-EqM/experiments/symmetry_toys/latent_symmetry.py \
    --out-json "$OUT_JSON" 2>&1 | tee "$RUN_LOG"

echo "=== done. results: $OUT_JSON ; log: $RUN_LOG ===" >&2
