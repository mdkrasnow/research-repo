#!/bin/bash
#SBATCH --job-name=eqm-geodesic-bank
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/geodesic-bank-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/geodesic-bank-%A.err
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
set -euo pipefail
module load python/3.10.13-fasrc01; module load cuda/11.8.0-fasrc01
: "${GIT_SHA:?}"
OUT=/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/geodesic_manifold_preliminary_epoch15/inputs/bank.pt
VAL=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/val
test ! -e "$OUT"; test -d "$VAL"
WORK=/tmp/geodesic-bank-$SLURM_JOB_ID; git clone --depth=50 https://github.com/mdkrasnow/research-repo.git "$WORK"; cd "$WORK"; git checkout "$GIT_SHA"; cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchvision
python -m geodesic_manifold.build_bank --val "$VAL" --out "$OUT"
