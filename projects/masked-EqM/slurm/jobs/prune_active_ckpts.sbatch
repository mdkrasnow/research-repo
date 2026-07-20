#!/bin/bash
#SBATCH --job-name=eqm-prune
#SBATCH --output=projects/masked-EqM/slurm/logs/prune-%x_%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/prune-%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=48:00:00
#SBATCH --partition=shared

# Standing pruner for masked-EqM results dirs on holylabs.
# Watches corruption_sanity_* dirs (all corruption families) and stage1_*
# dirs (longer-training curve, 2026-07-20) and keeps only the anchor ckpt
# (0005000) + latest 1 step-ckpt per dir; also nukes leftover rsync
# partial-transfer temp files (.NAME.pt.XXXXXX). epoch*.pt checkpoints
# (train.py epoch-boundary saves, needed for the Stage 1 frozen eval at
# epochs 1/2/3/5) are NEVER deleted regardless of KEEP_LATEST -- these are
# the "final and evaluation checkpoints" the pruner must keep safe.
# Auto-exits once no corrupt-sanity/stage1 jobs remain in queue. Mirrors
# the diff-EqM quota-deadlock mitigation (see AGENTS.md); scoped to
# masked-EqM only, do not read/touch projects/archive/diff-EqM.
# --time bumped 6h->48h (2026-07-20) after the pruner TIMEOUT'd mid-run
# during a 6+ hour Stage1 5-epoch training batch, leaving quota unprotected
# for the back half and re-blowing it -- all 9 Stage1 jobs died a second
# time. Root cause of the *quota pressure itself* fixed separately in
# train_stage1_longer.sbatch: --ckpt-every raised from 5000 to effectively
# disabled (999999999) for multi-epoch runs, since Stage1 only needs the
# epoch-boundary checkpoints (epoch01..05.pt) -- the periodic 5000-step
# checkpointing was inherited from the 1-epoch corruption_sanity template
# and was writing ~8 x 2GB ckpts/epoch/arm for no reason at 5-epoch scale
# (~720GB across 9 arms vs ~90GB needed). This pruner remains a backstop.
# KEEP_LATEST reduced 2->1 and poll tightened 300s->120s (2026-07-15) after
# the group quota (already near its 4.0Ti cap externally) was pushed over
# by our own ~100G checkpoint footprint even at KEEP_LATEST=2 -- see
# pi-updates.md 2026-07-15 draft. Combined with reduced job concurrency
# (batches of ~6 instead of all 18 parallel) as the primary mitigation.
set -uo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/results}"
KEEP_LATEST=1
ANCHOR_STEP="0005000"

while squeue -u "$USER" -o '%j' 2>/dev/null | grep -qE 'eqm-corr|eqm-stage1'; do
    for dir in "$RESULTS_ROOT"/corruption_sanity_*/*/checkpoints "$RESULTS_ROOT"/stage1_*/*/checkpoints; do
        [ -d "$dir" ] || continue
        ls -t "$dir"/*.pt 2>/dev/null | grep -v '/epoch[0-9]*\.pt$' | tail -n +$((KEEP_LATEST + 1)) | grep -v "$ANCHOR_STEP" | xargs -r rm -f
    done
    find "$RESULTS_ROOT" -name '.*.pt.*' -delete 2>/dev/null
    sleep 120
done

echo "=== No eqm-corr/eqm-stage1 jobs remain in queue, pruner exiting ==="
