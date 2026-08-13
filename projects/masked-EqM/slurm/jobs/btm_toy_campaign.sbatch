#!/bin/bash
#SBATCH --job-name=btm-toy
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-toy-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-toy-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
set -euo pipefail

# Experiment I: the decisive five-atom BTM benchmark (arXiv:2608.01692v2 Sec 3).
# Pure 2D toy -- no ImageNet, no VAE, no checkpoints from prior campaigns.
# Single GPU: gpu_test MIG slices are fine (no DDP, no NCCL).

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi || true

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
WORK_DIR="/tmp/btm-toy-${SLURM_JOB_ID}"
STEPS="${STEPS:-6000}"
BATCH="${BATCH:-1024}"
EVAL_N="${EVAL_N:-100000}"
SEEDS="${SEEDS:-10}"
STAGES="${STAGES:-0,1,2,3,4}"
OUT_REL="results/btm/toy5_job${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
# Fetch ALL branches: this campaign runs on an experiment branch, not main.
git fetch --depth=50 origin '+refs/heads/*:refs/remotes/origin/*'
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM

pip install --quiet timm 2>/dev/null || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

DEV=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
echo "device=$DEV steps=$STEPS batch=$BATCH eval_n=$EVAL_N seeds=$SEEDS stages=$STAGES"

# Unit tests are a hard gate: no campaign runs on unvalidated mathematics.
python -m pytest tests/test_btm_math.py -q

python -m experiments.btm.toy_campaign \
  --out-dir "$OUT_REL" --device "$DEV" --steps "$STEPS" --batch "$BATCH" \
  --eval-n "$EVAL_N" --seeds "$SEEDS" --stages "$STAGES" 2>&1 | tee toy.log

mkdir -p "$REPO_ROOT/projects/masked-EqM/results/btm"
rsync -a "$OUT_REL/" "$REPO_ROOT/projects/masked-EqM/$OUT_REL/"
cp toy.log "$REPO_ROOT/projects/masked-EqM/slurm/logs/btm_toy_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
