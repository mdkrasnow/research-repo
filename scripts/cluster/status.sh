#!/usr/bin/env bash
set -euo pipefail
jid="${1:?job id required}"

if command -v squeue >/dev/null 2>&1 || command -v sacct >/dev/null 2>&1; then
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$jid" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS -n | head -n 1
    exit 0
  fi
  squeue -j "$jid" -o "%i %T %M %R" -h
  exit 0
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$root/scripts/cluster/remote_status.sh" "$jid"
