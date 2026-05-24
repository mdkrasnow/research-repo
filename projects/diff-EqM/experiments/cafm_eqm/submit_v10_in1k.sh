#!/usr/bin/env bash
# Submit v10-only 80ep ImageNet-1K EqM-B/2 run.
#
# Usage from repo root:
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_v10_in1k.sh [SEED]
#
# Pre-condition: vanilla baseline FID 31.41 verified at this scale.
# Smoke-probe protocol: ckpt_every=5000; run FID eval on first 5K ckpt
# before letting full 80ep complete (catches v10 mechanism failures fast).
set -euo pipefail

SEED="${1:-${SEED:-0}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

GIT_SHA="$(git rev-parse HEAD)"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"

ahead="$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)"
if [ "$ahead" -gt 0 ]; then
    echo "ERROR: $ahead commits ahead of origin/main. Push first." >&2
    exit 1
fi

# Sync slurm/ to cluster (sbatch directives parsed from cluster-side file).
CONTROL_PATH="$HOME/.ssh/cc-research-repo-%r@%h:%p"
rsync -az -e "ssh -o ControlPath=$CONTROL_PATH" \
    projects/diff-EqM/slurm/ \
    mkrasnow@login.rc.fas.harvard.edu:/n/home03/mkrasnow/research-repo/projects/diff-EqM/slurm/

echo "Submitting v10 IN-1K 80ep run:"
echo "  SHA: $GIT_SHA"
echo "  SEED: $SEED"

bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
    sbatch --export=GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA,SEED=$SEED \
    projects/diff-EqM/slurm/jobs/imagenet1k_80ep_v10.sbatch"
