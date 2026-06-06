#!/usr/bin/env bash
# Poll a Token Factory fine-tune job to completion, then download its LoRA adapter result_files.
# (Serving is beta-gated, so we preserve artifacts for later GPU-VM / beta serving.)
# Usage: poll_and_download.sh <job_json> <tag>
set -uo pipefail
cd "$(dirname "$0")/.."
JOB_FILE="${1:?job_json}"; TAG="${2:?tag}"
export GPT_OSS_BASE_URL="${GPT_OSS_BASE_URL:-https://api.tokenfactory.nebius.com/v1/}"
export GPT_OSS_API_KEY="${GPT_OSS_API_KEY:-$NEBIUS_API_KEY}"
MARKER="gpt_oss/data/out/ft_download_done_${TAG}.txt"
JOB_ID="$(python3 -c "import json;print(json.load(open('$JOB_FILE'))['job_id'])")"

for i in $(seq 1 120); do
  ST="$(python3 - "$JOB_ID" <<'PY'
import json,os,sys,requests
jid=sys.argv[1]; key=os.getenv("GPT_OSS_API_KEY"); base=os.getenv("GPT_OSS_BASE_URL").rstrip("/")
print(requests.get(f"{base}/fine_tuning/jobs/{jid}",headers={"Authorization":f"Bearer {key}"},timeout=60).json().get("status",""))
PY
)"
  echo "[$TAG] poll $i: $ST"
  case "$ST" in
    succeeded) python3 gpt_oss/download_ft_result.py --job-file "$JOB_FILE" --tag "$TAG" && echo "DONE $TAG adapters downloaded" > "$MARKER"; exit 0 ;;
    failed|cancelled) echo "$TAG job $ST" > "$MARKER"; exit 1 ;;
  esac
  sleep 30
done
echo "$TAG poll timeout" > "$MARKER"; exit 1
