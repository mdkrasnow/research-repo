#!/usr/bin/env bash
# Submit FID evaluation for the latest Phase 1b CAFM-EqM checkpoint.
#
# Usage from repo root:
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_phase_1b_fid.sh [SEED]
#
# Looks under projects/diff-EqM/results/cafm_eqm_b2_in256_seed${SEED}/ on the
# cluster, picks the highest-numbered eqm_compat_*.pt (or override CKPT env).
# eqm_compat ckpts have `model` key compatible with sample_gd.py.
set -euo pipefail

SEED="${1:-${SEED:-0}}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"

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

# Resolve latest eqm_compat ckpt on cluster, unless CKPT env overrides.
if [ -z "${CKPT:-}" ]; then
    PERSIST_DIR="projects/diff-EqM/results/cafm_eqm_b2_in256_seed${SEED}"
    CKPT="$(bash scripts/cluster/ssh.sh \
        "ls -1 /n/home03/mkrasnow/research-repo/$PERSIST_DIR/eqm_compat_*.pt 2>/dev/null | sort | tail -1")"
    if [ -z "$CKPT" ]; then
        echo "ERROR: no eqm_compat_*.pt found in cluster $PERSIST_DIR" >&2
        exit 1
    fi
fi

echo "Submitting Phase 1b FID eval:"
echo "  SHA: $GIT_SHA"
echo "  SEED: $SEED"
echo "  CKPT: $CKPT"
echo "  NUM_FID_SAMPLES: $NUM_FID_SAMPLES"

bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
    sbatch --export=GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA,CHECKPOINT_PATH=$CKPT,NUM_FID_SAMPLES=$NUM_FID_SAMPLES,SAMPLE_BATCH_SIZE=$SAMPLE_BATCH_SIZE \
    projects/diff-EqM/slurm/jobs/imagenet1k_fid_eval.sbatch"
