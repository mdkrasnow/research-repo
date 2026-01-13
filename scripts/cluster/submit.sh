#!/usr/bin/env bash
set -euo pipefail
script="${1:?path to sbatch script required}"
project_slug="${2:-}"

if command -v sbatch >/dev/null 2>&1; then
  out="$(sbatch "$script")"
  echo "$out" | awk '{print $4}'
  exit 0
fi

if [[ -z "$project_slug" ]]; then
  echo "sbatch not found locally. Provide project_slug to submit remotely." >&2
  exit 22
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$root/scripts/cluster/remote_submit.sh" "$script" "$project_slug"
