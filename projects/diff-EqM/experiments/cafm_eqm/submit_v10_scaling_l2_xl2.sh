#!/usr/bin/env bash
# Launch v10 L/2 + XL/2 full 80ep IN-1K trains.
# Pre-condition: vanilla S/2 + v10 S/2 results show mining transfers at S/2 scale.
# Smokes already PASS: v10 L/2 (aux/base 1.000 — weak; flagged), v10 XL/2 GBS=32 (aux/base 1.024).
#
# Usage from repo root:
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_v10_scaling_l2_xl2.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

GIT_SHA="$(git rev-parse HEAD)"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"

ahead="$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)"
if [ "$ahead" -gt 0 ]; then
    echo "ERROR: $ahead commits ahead of origin/main. Push first." >&2
    exit 1
fi

CONTROL_PATH="$HOME/.ssh/cc-research-repo-%r@%h:%p"
rsync -az -e "ssh -o ControlPath=$CONTROL_PATH" \
    projects/diff-EqM/slurm/ \
    mkrasnow@login.rc.fas.harvard.edu:/n/home03/mkrasnow/research-repo/projects/diff-EqM/slurm/

echo "Submitting v10 L/2 + XL/2 80ep IN-1K full trains (SHA $GIT_SHA)..."

# L/2: GBS=256 (smoke ran GBS=64 to fit smoke time; full uses paper-default 256)
echo "--- v10 L/2 80ep seed 0 ---"
bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
    sbatch --partition=seas_gpu --time=48:00:00 \
    --export=GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA,MODEL=EqM-L/2,SEED=0,RESULTS_TAG=v10_l2_seed0 \
    projects/diff-EqM/slurm/jobs/imagenet1k_v10_scaling.sbatch"

# XL/2: GBS=128 (smoke validated 32 fits; full uses 128 as compromise; reduce to 64 if OOM on full run)
echo "--- v10 XL/2 80ep seed 0 (GBS=128) ---"
bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
    sbatch --partition=seas_gpu --time=48:00:00 \
    --export=GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA,MODEL=EqM-XL/2,SEED=0,GLOBAL_BATCH_SIZE=128,RESULTS_TAG=v10_xl2_seed0 \
    projects/diff-EqM/slurm/jobs/imagenet1k_v10_scaling.sbatch"

echo "Done. Add active_runs entries to pipeline.json."
echo "Note: XL/2 80ep at GBS=128 estimated 80-100h compute — will hit 48h cap; use submit_resume_train.sh v10 EqM-XL/2 0."
