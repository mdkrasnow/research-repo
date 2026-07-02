#!/usr/bin/env bash
# One-shot submitter for the C1-extension + C3 capability probes.
# Prereq: SSH ControlMaster up (scripts/cluster/ssh_bootstrap.sh).
# Submits 3 jobs to gpu_test, prints job IDs. Idempotent-ish: re-running submits
# fresh jobs (use only once). Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
RS="scripts/cluster/remote_submit.sh"
SLUG="diff-EqM"

echo "=== C1+C3 submission (gpu_test) — SHA $(git rev-parse --short HEAD) ===" >&2
echo "[submit] C3 OOD-energy ..." >&2
C3_ID="$(bash "$RS" projects/diff-EqM/slurm/jobs/c3_ood_energy.sbatch "$SLUG")"
echo "  c3_ood_energy        -> $C3_ID" >&2
echo "[submit] C1-ext l01 (vanilla + v10 l=0.1) ..." >&2
C1L01_ID="$(bash "$RS" projects/diff-EqM/slurm/jobs/c1_ext_l01.sbatch "$SLUG")"
echo "  c1_ext_l01           -> $C1L01_ID" >&2
echo "[submit] C1-ext l03 (v10 l=0.3 only) ..." >&2
C1L03_ID="$(bash "$RS" projects/diff-EqM/slurm/jobs/c1_ext_l03.sbatch "$SLUG")"
echo "  c1_ext_l03           -> $C1L03_ID" >&2

echo "" >&2
echo "C3_JOB=$C3_ID"
echo "C1_L01_JOB=$C1L01_ID"
echo "C1_L03_JOB=$C1L03_ID"
echo "" >&2
echo "Monitor: scripts/cluster/status.sh <job_id>  OR  scripts/cluster/ssh.sh \"squeue -u \\\$USER\"" >&2
