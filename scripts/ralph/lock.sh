#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"
proj_dir="${2:-}"
lock_dir="$proj_dir/.state/lock"

is_pid_running() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 1
  ps -p "$pid" >/dev/null 2>&1
}

case "$cmd" in
  acquire)
    mkdir -p "$proj_dir/.state"
    for i in {1..50}; do
      # If lock exists, check for staleness and clean it up.
      if [[ -d "$lock_dir" ]]; then
        pid="$(cat "$lock_dir/pid" 2>/dev/null || true)"
        if ! is_pid_running "$pid"; then
          rm -rf "$lock_dir" || true
        fi
      fi

      if mkdir "$lock_dir" 2>/dev/null; then
        echo "$$" > "$lock_dir/pid"
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$lock_dir/acquired_at"
        exit 0
      fi
      sleep 0.1
    done
    echo "Could not acquire lock for $proj_dir" >&2
    exit 1
    ;;
  release)
    [ -d "$lock_dir" ] && rm -rf "$lock_dir" || true
    ;;
  *)
    echo "Usage: lock.sh acquire|release <project_dir>" >&2
    exit 1
    ;;
esac
