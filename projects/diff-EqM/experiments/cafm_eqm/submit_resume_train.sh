#!/usr/bin/env bash
# Resume a vanilla or v10 train from latest ckpt in persistent results dir.
# Auto-detects latest ckpt cluster-side. Suitable for 48h-cap recovery.
#
# Usage:
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_resume_train.sh \
#       <vanilla|v10> <MODEL> <SEED> [extra_export_kv]
# Examples:
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_resume_train.sh vanilla EqM-L/2 0
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_resume_train.sh v10 EqM-B/2 1
set -euo pipefail

KIND="${1:?usage: <vanilla|v10> <MODEL> <SEED>}"
MODEL="${2:?usage: <vanilla|v10> <MODEL> <SEED>}"
SEED="${3:?usage: <vanilla|v10> <MODEL> <SEED>}"
EXTRA="${4:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
GIT_SHA="$(git rev-parse HEAD)"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"

MODEL_TAG=$(echo "$MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')

if [ "$KIND" = "vanilla" ]; then
    RESULTS_DIR="projects/diff-EqM/results/imagenet1k_80ep_vanilla_${MODEL_TAG}_seed${SEED}"
    SBATCH="projects/diff-EqM/slurm/jobs/imagenet1k_80ep_vanilla_scaling.sbatch"
elif [ "$KIND" = "v10" ]; then
    # v10 RESULTS_TAG depends on what was used at submit; common pattern:
    RESULTS_TAG="${RESULTS_TAG:-v10_${MODEL_TAG}_seed${SEED}}"
    RESULTS_DIR="projects/diff-EqM/results/imagenet1k_80ep_${RESULTS_TAG}"
    SBATCH="projects/diff-EqM/slurm/jobs/imagenet1k_v10_scaling.sbatch"
else
    echo "ERROR: KIND must be vanilla or v10, got $KIND" >&2
    exit 1
fi

CONTROL_PATH="$HOME/.ssh/cc-research-repo-%r@%h:%p"
rsync -az -e "ssh -o ControlPath=$CONTROL_PATH" \
    projects/diff-EqM/slurm/ \
    mkrasnow@login.rc.fas.harvard.edu:/n/home03/mkrasnow/research-repo/projects/diff-EqM/slurm/

# Auto-resolve latest ckpt cluster-side. Prefers final.pt > highest step number > 0005000.pt.
LATEST_CKPT=$(bash scripts/cluster/ssh.sh "
DIR=/n/home03/mkrasnow/research-repo/$RESULTS_DIR
if [ -f \$DIR/*/checkpoints/final.pt ]; then ls \$DIR/*/checkpoints/final.pt; exit 0; fi
ls \$DIR/*/checkpoints/*.pt 2>/dev/null | sort -V | tail -1
" | tr -d '[:space:]')

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No ckpt found in $RESULTS_DIR" >&2
    exit 1
fi

echo "Resuming $KIND $MODEL seed=$SEED from: $LATEST_CKPT"

EXPORT="GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA,MODEL=$MODEL,SEED=$SEED,RESUME_CKPT=$LATEST_CKPT"
if [ -n "$EXTRA" ]; then
    EXPORT="$EXPORT,$EXTRA"
fi

bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
    sbatch --partition=seas_gpu --time=48:00:00 --export=$EXPORT $SBATCH"
