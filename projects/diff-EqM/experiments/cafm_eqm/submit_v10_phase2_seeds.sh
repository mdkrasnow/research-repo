#!/usr/bin/env bash
# Phase 2 multi-seed launch: v10 IN-1K seeds 1 + 2 (seed 0 is Phase 1 gate run).
# Gate: 3-seed Welch t p<0.05, mean >=1 FID gain vs vanilla 31.41 (CLAUDE.md).
#
# Pre-condition: Phase 1 seed-0 PASS gate (FID <= 30.41).
# Usage from repo root:
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_v10_phase2_seeds.sh
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

# Diversify partitions per CLAUDE.md QOS rule (2 jobs OK on one partition,
# but seas_gpu typically lower-queue).
for SEED in 1 2; do
    echo "Submitting Phase 2 v10 seed=$SEED (SHA $GIT_SHA)..."
    bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
        sbatch --export=GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA,SEED=$SEED \
        projects/diff-EqM/slurm/jobs/imagenet1k_80ep_v10.sbatch"
done

echo "Done. Add active_runs entries to pipeline.json."
