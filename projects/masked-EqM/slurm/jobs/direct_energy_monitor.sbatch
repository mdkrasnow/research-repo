#!/bin/bash
#SBATCH --job-name=eqm-direct-monitor
#SBATCH --output=projects/masked-EqM/slurm/logs/direct-monitor-%j.out
#SBATCH --error=projects/masked-EqM/slurm/logs/direct-monitor-%j.err
#SBATCH --partition=shared
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=12:00:00
set -euo pipefail

REPO_ROOT="/n/home03/mkrasnow/research-repo"
STATUS_DIR="$REPO_ROOT/projects/masked-EqM/results/direct_energy_campaign/monitor"
JOB_IDS="${JOB_IDS:?set comma-separated JOB_IDS}"
mkdir -p "$STATUS_DIR"

while true; do
    timestamp="$(date -Is)"
    sacct -n -P -j "$JOB_IDS" --format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS \
        | awk -F'|' -v at="$timestamp" 'NF >= 4 && $1 != "" && $1 !~ /\./ {printf "{\"at\":\"%s\",\"job_id\":\"%s\",\"state\":\"%s\",\"exit_code\":\"%s\",\"elapsed\":\"%s\",\"max_rss\":\"%s\"}\n", at,$1,$2,$3,$4,$5}' \
        >> "$STATUS_DIR/scheduler_events.jsonl"
    sleep 300
done
