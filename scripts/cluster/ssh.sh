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
print(os.path.expanduser(cfg["ssh"]["control_path"]))
PY
)"

cmd="${*:?remote command required}"

ssh -p "$port" \
  -o BatchMode=yes \
  -o ControlPath="$control_path" \
  "$user@$host" "$cmd"
