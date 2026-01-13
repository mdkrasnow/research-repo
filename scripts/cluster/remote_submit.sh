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

git_url="$(python3 - "$cfg" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg.get("git", {}).get("url", "https://github.com/mdkrasnow/research-repo.git"))
PY
)"

# Get current git SHA
git_sha="$(cd "$root" && git rev-parse HEAD)"

# Sync only the slurm directory to remote (for sbatch scripts and log directory structure).
rsync -az --delete \
  -e "ssh -p $port -o BatchMode=yes -o ControlPath=$control_path" \
  "$root/projects/$project_slug/slurm/" \
  "$user@$host:$remote_root/projects/$project_slug/slurm/"

# Compute relative path for sbatch script.
remote_sbatch="$remote_root/projects/$project_slug/$(python3 - "$sbatch_path" "$root/projects/$project_slug" <<'PY'
import os,sys
print(os.path.relpath(sys.argv[1], sys.argv[2]))
PY
)"

# Submit on the cluster with environment variables for git clone.
jobid="$("$root/scripts/cluster/ssh.sh" "cd $remote_root && sbatch --export=GIT_URL=$git_url,GIT_SHA=$git_sha $remote_sbatch" | awk '{print $4}')"
echo "$jobid"
