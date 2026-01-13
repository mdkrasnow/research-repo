#!/usr/bin/env bash
set -euo pipefail
exec python3 "$CLAUDE_PROJECT_DIR/scripts/ralph/stop_hook.py"
