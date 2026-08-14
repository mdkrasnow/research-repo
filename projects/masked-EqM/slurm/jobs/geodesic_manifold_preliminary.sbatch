#!/bin/bash
#SBATCH --job-name=eqm-geodesic-preflight
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/geodesic-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/geodesic-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
: "${GIT_SHA:?}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/geodesic_manifold_preliminary_epoch15}"
BANK="${BANK:-$RESULTS_ROOT/inputs/bank.pt}"
DOT_CKPT="${DOT_CKPT:-/n/home03/mkrasnow/direct_energy_longer_retry/longer15_dot_seed0_job35524601/000-EqM-B-2-Linear-velocity-None-ebm-dot/checkpoints/epoch15.pt}"
DIRECT_CKPT="${DIRECT_CKPT:-/n/home03/mkrasnow/direct_energy_longer_retry/longer15_direct_seed0_job35524667/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/epoch15.pt}"
NONE_CKPT="${NONE_CKPT:-/n/home03/mkrasnow/direct_energy_longer_retry/longer15_none_seed0_job35524599/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch15.pt}"
VAL_PATH="${VAL_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/val}"
test -f "$BANK" && test -f "$DOT_CKPT" && test -f "$DIRECT_CKPT" && test -f "$NONE_CKPT" && test -d "$VAL_PATH"
test "$(find "$VAL_PATH" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1000
mkdir -p "$RESULTS_ROOT"
test "$(df -Pk "$RESULTS_ROOT" | awk 'NR==2 {print $4}')" -ge 20000000
WORK_DIR="/tmp/masked-eqm-geodesic-${SLURM_JOB_ID}"
git clone --depth=50 https://github.com/mdkrasnow/research-repo.git "$WORK_DIR"
cd "$WORK_DIR" && git checkout "$GIT_SHA" && cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchvision scipy pandas pyarrow
CONFIG="${CONFIG:-$WORK_DIR/projects/masked-EqM/geodesic_manifold/config_preliminary.json}"
python -m geodesic_manifold.preflight "$BANK"
python -m geodesic_manifold.benchmark --config "$CONFIG"
