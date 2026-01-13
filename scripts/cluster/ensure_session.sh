#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cfg_local="$root/.claude/cluster.local.json"
cfg_example="$root/.claude/cluster.example.json"

cfg="${cfg_local}"
if [[ ! -f "$cfg" ]]; then
  echo "Missing $cfg_local"
  echo "Copy $cfg_example -> $cfg_local and fill in user + remote.repo_root."
  exit 20
fi

user="$(python3 - "$cfg" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg["ssh"]["user"])
PY
)"

host="$(python3 - "$cfg" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg["ssh"]["host"])
PY
)"

port="$(python3 - "$cfg" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg["ssh"].get("port",22))
PY
)"

control_path="$(python3 - "$cfg" <<'PY'
import json,sys,os
cfg=json.load(open(sys.argv[1]))
cp=cfg["ssh"]["control_path"]
print(os.path.expanduser(cp))
PY
)"

# Check existing control socket
if ssh -p "$port" -o ControlPath="$control_path" -O check "$user@$host" >/dev/null 2>&1; then
  exit 0
fi

echo "No active SSH control session for $user@$host."
echo "Run: scripts/cluster/ssh_bootstrap.sh"
exit 21
