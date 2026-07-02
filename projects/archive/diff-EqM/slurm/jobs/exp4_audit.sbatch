#!/bin/bash
#SBATCH --job-name=exp4-audit
#SBATCH --output=slurm/logs/exp4-audit_%A.out
#SBATCH --error=slurm/logs/exp4-audit_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1

# Experiment 4: run the stability/gap/memorization audit over a config that maps
# regime -> checkpoint_type -> seed -> samples_npz. References + NN banks must
# already exist (exp4_build_refs.sbatch) and samples generated (exp4_generate.sbatch).
#
# Required env: CONFIG (path under repo, e.g. projects/diff-EqM/experiments/experiment4/configs/full_example.json)

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

WORK_DIR="/tmp/exp4-audit-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"
git init . 2>&1; git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -5 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || { git fetch --unshallow origin 2>&1 | tail -3; git checkout "${GIT_SHA}" || exit 1; }
pip install timm scipy torchvision pillow matplotlib open_clip_torch lpips 2>&1 | tail -5

REPO_ROOT="/n/home03/mkrasnow/research-repo"
CONFIG="${CONFIG:?set CONFIG path under repo}"
# Run from REPO_ROOT so the config's relative results/ paths resolve and outputs
# land in the persistent tree (not the ephemeral /tmp clone).
cd "$REPO_ROOT"
python projects/diff-EqM/experiments/experiment4/eval_stability_memorization.py \
    --config "$CONFIG" 2>&1 | tee "$WORK_DIR/audit.log"
cp "$WORK_DIR/audit.log" "$REPO_ROOT/projects/diff-EqM/slurm/logs/exp4-audit_${SLURM_JOB_ID}.log" || true
echo "=== audit done; see projects/diff-EqM/results/experiment4/ ==="
