#!/usr/bin/env bash
# Helper: submit Phase 3 multi-seed runs for v10+CAFM and CAFM-only.
# Launches 2 conditions × 2 additional seeds (seeds 1, 2; seed 0 already done
# in Phase 1b/2). Splits across gpu + seas_gpu per CLAUDE.md QOS rule (4+ jobs).
#
# Usage:
#   bash projects/diff-EqM/experiments/cafm_eqm/submit_phase_3_seeds.sh
set -euo pipefail

GIT_SHA="$(git rev-parse HEAD)"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
CONFIG="projects/diff-EqM/configs/cafm/eqm_b2_in256_cafm.yaml"

if ! git diff-index --quiet HEAD --; then
    echo "WARNING: uncommitted local changes." >&2
fi
ahead="$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)"
if [ "$ahead" -gt 0 ]; then
    echo "ERROR: $ahead commits ahead of origin/main. Push first." >&2; exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

# Submit pattern: split across partitions for QOS budget.
# 4 jobs total: (cafm s1, cafm s2, v10cafm s1, v10cafm s2)
declare -a JOBS=(
    "1 0 seas_gpu cafm_s1"
    "2 0 gpu     cafm_s2"
    "1 1 seas_gpu v10_s1"
    "2 1 gpu     v10_s2"
)

for spec in "${JOBS[@]}"; do
    read -r SEED ENABLE_V10 PART NAME <<<"$spec"
    echo "=== Submitting $NAME (SEED=$SEED, ENABLE_V10=$ENABLE_V10, PART=$PART) ==="
    bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
      sbatch --partition=$PART \
        --export=GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA,SEED=$SEED,ENABLE_V10=$ENABLE_V10,CONFIG=$CONFIG \
        projects/diff-EqM/slurm/jobs/cafm_eqm_b2_in256.sbatch"
done

echo "=== All 4 Phase 3 seeds submitted ==="
