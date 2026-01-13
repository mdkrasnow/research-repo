#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cfg="$root/.claude/cluster.local.json"
if [[ ! -f "$cfg" ]]; then
  echo "Missing $cfg. Copy .claude/cluster.example.json -> .claude/cluster.local.json"
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

control_persist="$(python3 - "$cfg" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg["ssh"].get("control_persist","30m"))
PY
)"

mkdir -p "$(dirname "$control_path")"

echo "Bootstrapping SSH ControlMaster session for $user@$host ..."
echo "Complete login + 2FA interactively when prompted."

ssh -p "$port" \
  -o ControlMaster=auto \
  -o ControlPersist="$control_persist" \
  -o ControlPath="$control_path" \
  "$user@$host" "echo 'SSH session ready.'"
