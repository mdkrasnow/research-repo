#!/bin/bash
#SBATCH --job-name=eqm-latsym-r4
#SBATCH --output=slurm/logs/latsym-r4_%A.out
#SBATCH --error=slurm/logs/latsym-r4_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1

# Rung 4: latent-coordinate symmetry discovery T=dec(M.enc(x)). Tiny (D=16), GPU
# seconds, CPU fallback. Single-GPU -> MIG-safe. Persists results+log to HOME.
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

WORK_DIR="/tmp/eqm-latsym-r4-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"; GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== clone @ ${GIT_SHA} ===" >&2
git init . >/dev/null 2>&1; git remote add origin "$GIT_URL"
git fetch --depth=50 origin main 2>&1 | tail -3 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -3 || { git fetch --unshallow origin 2>&1 | tail -3; git checkout "${GIT_SHA}" 2>&1 | tail -3 || exit 1; }

echo "=== deps ===" >&2
pip install --quiet torch numpy 2>&1 | tail -3 || pip install --quiet --user torch numpy 2>&1 | tail -3

OUT_HOME="/n/home03/mkrasnow/research-repo/projects/diff-EqM/results/latent_symmetry"; mkdir -p "$OUT_HOME"
OUT_JSON="$OUT_HOME/results_rung4_${SLURM_JOB_ID}.json"; RUN_LOG="$OUT_HOME/run_rung4_${SLURM_JOB_ID}.log"
nvidia-smi 2>&1 | head -12 >&2 || echo "no nvidia-smi" >&2

echo "=== run ===" >&2
PYTHONUNBUFFERED=1 python projects/diff-EqM/experiments/symmetry_toys/latent_symmetry_rung4.py \
    --out-json "$OUT_JSON" 2>&1 | tee "$RUN_LOG"
echo "=== done: $OUT_JSON ===" >&2
