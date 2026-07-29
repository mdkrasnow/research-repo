#!/bin/bash
#SBATCH --job-name=eqm-energy-outcome
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/energy-outcome-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/energy-outcome-%A.err
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --time=12:00:00
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
: "${GIT_SHA:?}"
BANK="${BANK:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/geodesic_manifold_preliminary_epoch15/inputs/bank.pt}"
OUT="${OUT:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/energy_outcome_monotonicity/epoch15_seed0/report.json}"
DOT_CHECKPOINT="${DOT_CHECKPOINT:-/n/home03/mkrasnow/direct_energy_longer_retry/longer15_dot_seed0_job35524601/000-EqM-B-2-Linear-velocity-None-ebm-dot/checkpoints/epoch15.pt}"
DIRECT_CHECKPOINT="${DIRECT_CHECKPOINT:-/n/home03/mkrasnow/direct_energy_longer_retry/longer15_direct_seed0_job35524667/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/epoch15.pt}"
WORK="$(mktemp -d /tmp/eqm-energy-outcome-${SLURM_JOB_ID}-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
git clone --depth=50 https://github.com/mdkrasnow/research-repo.git "$WORK"
git -C "$WORK" checkout "$GIT_SHA"
cd "$WORK/projects/masked-EqM"
pip install --quiet timm diffusers accelerate torchvision numpy
python -m energy_outcome_monotonicity.evaluate \
  --bank "$BANK" --dot-checkpoint "$DOT_CHECKPOINT" --direct-checkpoint "$DIRECT_CHECKPOINT" \
  --output "$OUT" --num-images "${NUM_IMAGES:-64}" --candidates "${CANDIDATES:-8}" \
  --batch-size "${BATCH_SIZE:-4}" --num-sampling-steps "${STEPS:-100}" \
  --stepsize "${STEPSIZE:-0.0017}" --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-10000}"
