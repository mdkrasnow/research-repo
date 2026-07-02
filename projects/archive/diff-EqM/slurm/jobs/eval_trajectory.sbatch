#!/bin/bash
#SBATCH --job-name=eqm-traj
#SBATCH --output=slurm/logs/eqm-traj_%A.out
#SBATCH --error=slurm/logs/eqm-traj_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1

# Capability ladder v2 Rungs C/E/F: trajectory-level behavioral probes.
# Frozen ckpts, identical GD sampler, matched noise+labels. Single GPU.
#   swap  (E): vanilla/v10 early<->late splice at switch t
#   edit  (F): class A->B switch partway; target success + source preservation
#   rescue(C): branch bottom-conf vanilla mid-states under vanilla vs v10
#
# Required env: VANILLA_CKPT, ANM_CKPT
# Optional: NUM_IMAGES (64), MODEL (EqM-B/2)

set -uo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

WORK_DIR="/tmp/eqm-traj-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"
git init . 2>&1; git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -3 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -3 || { git fetch --unshallow origin 2>&1|tail -3; git checkout "${GIT_SHA}" 2>&1|tail -3 || exit 1; }

if ! python -c "import timm, diffusers, scipy, torchvision, lpips" 2>/dev/null; then
    pip install --no-input timm diffusers accelerate scipy torchvision lpips pillow 2>&1 | tail -5
fi

REPO_ROOT="/n/home03/mkrasnow/research-repo"
MODEL="${MODEL:-EqM-B/2}"; NUM_IMAGES="${NUM_IMAGES:-64}"
VANILLA_CKPT="${VANILLA_CKPT:?set VANILLA_CKPT}"; ANM_CKPT="${ANM_CKPT:?set ANM_CKPT}"
OUT_DIR="$REPO_ROOT/projects/diff-EqM/results/trajectory_${SLURM_JOB_ID}"; mkdir -p "$OUT_DIR"

for mode in swap edit rescue; do
    echo "=== $mode ===" >&2
    python -u projects/diff-EqM/experiments/eval_trajectory.py \
        --mode "$mode" --vanilla-ckpt "$VANILLA_CKPT" --anm-ckpt "$ANM_CKPT" \
        --out-dir "$OUT_DIR" --model "$MODEL" --num-images "$NUM_IMAGES" 2>&1 | tail -25
done

echo "=== Done. metrics in $OUT_DIR ===" >&2
ls -la "$OUT_DIR" >&2
rm -rf "$WORK_DIR"
