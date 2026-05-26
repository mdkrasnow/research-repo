#!/bin/bash
#SBATCH --job-name=prune-v10-ckpts
#SBATCH --output=slurm/logs/prune-v10-ckpts_%A.out
#SBATCH --error=slurm/logs/prune-v10-ckpts_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=48:00:00
#SBATCH --partition=shared

# Periodic ckpt pruner. Keeps anchor ckpts (5000, 65000) + latest 2.
# Exits when MAIN_JOB_ID no longer in squeue.
# Usage: sbatch --export=MAIN_JOB_ID=15638767 prune_v10_ckpts.sbatch

set -uo pipefail

MAIN_JOB_ID="${MAIN_JOB_ID:-}"
if [ -z "$MAIN_JOB_ID" ]; then
    echo "ERROR: MAIN_JOB_ID env not set" >&2
    exit 1
fi

CKPT_DIR="/n/home03/mkrasnow/research-repo/projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints"
ANCHORS="0005000.pt 0065000.pt"

echo "=== Pruner started, watching main job $MAIN_JOB_ID ===" >&2
while squeue -j "$MAIN_JOB_ID" -h 2>/dev/null | grep -q "$MAIN_JOB_ID"; do
    if [ -d "$CKPT_DIR" ]; then
        cd "$CKPT_DIR"
        ls *.pt 2>/dev/null | sort > /tmp/prune_all.txt
        KEEP_LAST=$(tail -2 /tmp/prune_all.txt)
        for f in $ANCHORS $KEEP_LAST; do echo "$f"; done | sort -u > /tmp/prune_keep.txt
        TO_DEL=$(comm -23 /tmp/prune_all.txt /tmp/prune_keep.txt)
        if [ -n "$TO_DEL" ]; then
            N=$(echo "$TO_DEL" | wc -l)
            echo "$(date) pruning $N ckpts" >&2
            echo "$TO_DEL" | xargs -r rm -f
        fi
        cd /
    fi
    sleep 600
done
echo "=== Main job done, pruner exiting at $(date) ===" >&2
