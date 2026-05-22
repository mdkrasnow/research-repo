#!/usr/bin/env bash
# Helper: submit Phase 2 v10+CAFM combined 10-epoch run.
# Pre-condition: Phase 1b CAFM-only completed AND CAFM-only FID < 25.
#
# Usage:
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_phase_2.sh
#
# Override SEED env to launch additional seeds (default 0).
set -euo pipefail

SEED="${SEED:-0}"
ENABLE_V10=1
CONFIG="${CONFIG:-projects/diff-EqM/configs/cafm/eqm_b2_in256_cafm.yaml}"

GIT_SHA="$(git rev-parse HEAD)"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"

if ! git diff-index --quiet HEAD --; then
    echo "WARNING: uncommitted local changes. Cluster checks out only pushed commits." >&2
fi
ahead="$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)"
if [ "$ahead" -gt 0 ]; then
    echo "ERROR: $ahead commits ahead of origin/main. Push first." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "Submitting Phase 2 v10+CAFM:"
echo "  SHA: $GIT_SHA"
echo "  CONFIG: $CONFIG"
echo "  SEED: $SEED"
echo "  ENABLE_V10: $ENABLE_V10"

# Sync slurm/ first (sbatch directives parsed from cluster file at submit time).
CONTROL_PATH="$HOME/.ssh/cc-research-repo-%r@%h:%p"
rsync -az -e "ssh -o ControlPath=$CONTROL_PATH" \
    projects/diff-EqM/slurm/ \
    mkrasnow@login.rc.fas.harvard.edu:/n/home03/mkrasnow/research-repo/projects/diff-EqM/slurm/

bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
  sbatch --export=GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA,SEED=$SEED,ENABLE_V10=$ENABLE_V10,CONFIG=$CONFIG \
  projects/diff-EqM/slurm/jobs/cafm_eqm_b2_in256.sbatch"
