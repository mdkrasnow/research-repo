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
# only the anchor ckpt (0005000) + latest 2 ckpts per dir; also nukes
# leftover rsync partial-transfer temp files (.NAME.pt.XXXXXX). Auto-exits
# once no corrupt-sanity jobs remain in queue. Mirrors the diff-EqM
# quota-deadlock mitigation (see AGENTS.md); scoped to masked-EqM only,
# do not read/touch projects/archive/diff-EqM.
set -uo pipefail

RESULTS_ROOT="/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/results"
KEEP_LATEST=2
ANCHOR_STEP="0005000"

while squeue -u "$USER" -o '%j' 2>/dev/null | grep -qE 'eqm-corr'; do
    for dir in "$RESULTS_ROOT"/corruption_sanity_*/*/checkpoints; do
        [ -d "$dir" ] || continue
        ls -t "$dir"/*.pt 2>/dev/null | tail -n +$((KEEP_LATEST + 1)) | grep -v "$ANCHOR_STEP" | xargs -r rm -f
    done
    find "$RESULTS_ROOT" -name '.*.pt.*' -delete 2>/dev/null
    sleep 300
done

echo "=== No eqm-corr jobs remain in queue, pruner exiting ==="
