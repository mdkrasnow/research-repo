#!/usr/bin/env bash
# Phase 1 gate FID eval: v10 IN-1K seed 0 final ckpt vs vanilla baseline 31.41.
# Pre-staged; run when train job 15638767 (or successor) completes.
#
# Usage:
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_v10_phase1_fid.sh
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_v10_phase1_fid.sh <CKPT_REL_PATH>
#
# Defaults to highest-numbered ckpt in seed0 dir (resolved cluster-side).
# NUM_FID_SAMPLES=50000 per CLAUDE.md Phase 1 gate spec.
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

CKPT_REL="${1:-}"
CKPT_DIR_REL="projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints"

# Resolve ckpt path cluster-side if not given.
if [ -z "$CKPT_REL" ]; then
    CKPT_REL=$(bash scripts/cluster/ssh.sh "ls /n/home03/mkrasnow/research-repo/$CKPT_DIR_REL/*.pt | sort -V | tail -1")
    echo "Auto-resolved latest ckpt: $CKPT_REL"
fi

IN1K_REF="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train"

echo "Submitting Phase 1 gate FID:"
echo "  ckpt: $CKPT_REL"
echo "  ref:  $IN1K_REF"
echo "  N:    50000 samples"
echo "  SHA:  $GIT_SHA"

bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
    sbatch --export=GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA,\
CHECKPOINT_PATH=$CKPT_REL,\
NUM_CLASSES=1000,\
NUM_FID_SAMPLES=50000,\
SAMPLE_BATCH_SIZE=64,\
IMAGENET_REF=$IN1K_REF,\
MODEL=EqM-B/2 \
    projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch"
