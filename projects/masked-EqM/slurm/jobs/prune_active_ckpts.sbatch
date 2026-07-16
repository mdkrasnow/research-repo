#!/bin/bash
#SBATCH --job-name=eqm-prune
#SBATCH --output=projects/masked-EqM/slurm/logs/prune-%x_%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/prune-%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=06:00:00
#SBATCH --partition=shared

# Standing pruner for masked-EqM corruption_sanity results dirs on holylabs.
# Watches all corruption_sanity_* dirs (all corruption families) and keeps
# only the anchor ckpt (0005000) + latest 1 ckpt per dir; also nukes
# leftover rsync partial-transfer temp files (.NAME.pt.XXXXXX). Auto-exits
# once no corrupt-sanity jobs remain in queue. Mirrors the diff-EqM
# quota-deadlock mitigation (see AGENTS.md); scoped to masked-EqM only,
# do not read/touch projects/archive/diff-EqM.
# KEEP_LATEST reduced 2->1 and poll tightened 300s->120s (2026-07-15) after
# the group quota (already near its 4.0Ti cap externally) was pushed over
# by our own ~100G checkpoint footprint even at KEEP_LATEST=2 -- see
# pi-updates.md 2026-07-15 draft. Combined with reduced job concurrency
# (batches of ~6 instead of all 18 parallel) as the primary mitigation.
set -uo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/results}"
KEEP_LATEST=1
ANCHOR_STEP="0005000"

while squeue -u "$USER" -o '%j' 2>/dev/null | grep -qE 'eqm-corr'; do
    for dir in "$RESULTS_ROOT"/corruption_sanity_*/*/checkpoints; do
        [ -d "$dir" ] || continue
        ls -t "$dir"/*.pt 2>/dev/null | tail -n +$((KEEP_LATEST + 1)) | grep -v "$ANCHOR_STEP" | xargs -r rm -f
    done
    find "$RESULTS_ROOT" -name '.*.pt.*' -delete 2>/dev/null
    sleep 120
done

echo "=== No eqm-corr jobs remain in queue, pruner exiting ==="
