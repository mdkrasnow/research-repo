#!/usr/bin/env bash
set -euo pipefail

project_slug="${1:?project slug required}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cfg="$root/.claude/cluster.local.json"

if ! "$root/scripts/cluster/ensure_session.sh" >/dev/null 2>&1; then
  echo "No SSH session. Run: scripts/cluster/ssh_bootstrap.sh" >&2
  exit 21
fi

remote_root="$(python3 - "$cfg" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg["remote"]["repo_root"])
PY
)"

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

mkdir -p "$root/projects/$project_slug/slurm/logs"

rsync -az \
  -e "ssh -p $port -o BatchMode=yes -o ControlPath=$control_path" \
  "$user@$host:$remote_root/projects/$project_slug/slurm/logs/" \
  "$root/projects/$project_slug/slurm/logs/"
