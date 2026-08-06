#!/bin/bash
#SBATCH --job-name=fb-direct-fp64-bisect
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fb-direct-fp64-bisect-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fb-direct-fp64-bisect-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT="${CKPT:?set CKPT}"
WORK_DIR="/tmp/masked-eqm-fp64bisect-${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm accelerate

python experiments/direct_energy/fb_direct_fp64_bisect.py --ckpt "$CKPT" --batch-size 2 2>&1 | tee bisect.log
EXIT_CODE=${PIPESTATUS[0]}
cp bisect.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fb_direct_fp64_bisect_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
