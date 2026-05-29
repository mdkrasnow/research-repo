#!/usr/bin/env bash
# Submit capability eval (denoise/inpaint/compose) on vanilla + v10 ckpts.
# Pure eval, no retrain. ~30-60min single GPU on gpu_requeue.
#
# Usage from repo root:
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_capabilities.sh
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

VANILLA_CKPT="/n/home03/mkrasnow/research-repo/projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt"
V10_CKPT="/n/home03/mkrasnow/research-repo/projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt"

echo "Submitting capability eval (SHA $GIT_SHA)"
echo "  vanilla: $VANILLA_CKPT"
echo "  v10:     $V10_CKPT"

bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
    sbatch --export=GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA,\
VANILLA_CKPT=$VANILLA_CKPT,V10_CKPT=$V10_CKPT,NUM_IMAGES=8 \
    projects/diff-EqM/slurm/jobs/eval_capabilities.sbatch"
