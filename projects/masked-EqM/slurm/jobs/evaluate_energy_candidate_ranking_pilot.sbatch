#!/bin/bash
#SBATCH --job-name=eqm-energy-rank
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/energy-rank-%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/energy-rank-%j.err
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --time=12:00:00
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true; module load cuda/11.8.0-fasrc01 2>/dev/null || true
: "${GIT_SHA:?}"; OUT="${2:-${OUT:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/energy_candidate_ranking/pilot}}"
WORK="$(mktemp -d /tmp/eqm-energy-rank-${SLURM_JOB_ID}-XXXXXX)"; trap 'rm -rf "$WORK"' EXIT
git clone --depth=50 https://github.com/mdkrasnow/research-repo.git "$WORK"; git -C "$WORK" checkout "$GIT_SHA"
cd "$WORK/projects/masked-EqM"; pip install --quiet timm diffusers accelerate torchvision pandas matplotlib
EVAL_CONFIG="${1:-${EVAL_CONFIG:-configs/energy_candidate_ranking_pilot.json}}"
ARGS=(--config "$EVAL_CONFIG" --data-path /n/holylabs/ydu_lab/Lab/raywang4/imagenet/val --output "$OUT")
if [[ -n "${SEED:-}" ]]; then ARGS+=(--seed "$SEED"); fi
python -m energy_candidate_ranking.evaluate "${ARGS[@]}"
