#!/usr/bin/env bash
set -euo pipefail
jid="${1:?job id required}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! "$root/scripts/cluster/ensure_session.sh" >/dev/null 2>&1; then
  echo "No SSH session. Run: scripts/cluster/ssh_bootstrap.sh" >&2
  exit 21
fi

# Prefer sacct if available, else squeue.
"$root/scripts/cluster/ssh.sh" "command -v sacct >/dev/null 2>&1 && sacct -j '$jid' --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS -n | head -n 1 || squeue -j '$jid' -o '%i %T %M %R' -h"
