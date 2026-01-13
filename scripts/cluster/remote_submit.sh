#!/usr/bin/env bash
set -euo pipefail

sbatch_path="${1:?path to sbatch script required (local)}"
project_slug="${2:?project slug required}"

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

# Sync the project folder to remote (minimal, excludes runs/results).
rsync -az --delete \
  -e "ssh -p $port -o BatchMode=yes -o ControlPath=$control_path" \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "runs/" \
  --exclude "results/tmp/" \
  "$root/projects/$project_slug/" \
  "$user@$host:$remote_root/projects/$project_slug/"

# Compute relative path for sbatch script.
remote_sbatch="$remote_root/projects/$project_slug/$(python3 - "$sbatch_path" "$root/projects/$project_slug" <<'PY'
import os,sys
print(os.path.relpath(sys.argv[1], sys.argv[2]))
PY
)"

# Submit on the cluster.
jobid="$("$root/scripts/cluster/ssh.sh" "cd $remote_root && sbatch $remote_sbatch" | awk '{print $4}')"
echo "$jobid"
