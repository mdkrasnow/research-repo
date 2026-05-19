#!/usr/bin/env bash
# Helper: submit Phase 1b full CAFM-EqM 10-epoch post-training run.
# Pre-condition: smoke (13848066) PASSED gate evaluation.
#
# Usage from repo root:
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_phase_1b.sh
#
# Submits to seas_gpu (mining variants OOM gpu_test 20G; need 40G+).
# Sets ENABLE_V10=0 by default; pass ENABLE_V10=1 for v10+CAFM combined (Phase 2).
set -euo pipefail

SEED="${SEED:-0}"
ENABLE_V10="${ENABLE_V10:-0}"

# Ensure latest commit pushed.
GIT_SHA="$(git rev-parse HEAD)"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"

# Verify push state.
if ! git diff-index --quiet HEAD --; then
    echo "WARNING: uncommitted local changes. Cluster will check out only pushed commits." >&2
fi
ahead="$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)"
if [ "$ahead" -gt 0 ]; then
    echo "ERROR: $ahead commits ahead of origin/main. Push first." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-projects/diff-EqM/configs/cafm/eqm_b2_in256_cafm.yaml}"

echo "Submitting Phase 1b run:"
echo "  SHA: $GIT_SHA"
echo "  CONFIG: $CONFIG"
echo "  SEED: $SEED"
echo "  ENABLE_V10: $ENABLE_V10"

bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
  sbatch --export=GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA,SEED=$SEED,ENABLE_V10=$ENABLE_V10,CONFIG=$CONFIG \
  projects/diff-EqM/slurm/jobs/cafm_eqm_b2_in256.sbatch"
