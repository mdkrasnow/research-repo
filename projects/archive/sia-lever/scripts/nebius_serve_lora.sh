#!/usr/bin/env bash
# Path-2 orchestrator: provision a Nebius H200 VM, vLLM-serve base gpt-oss-120b + our LoRA adapter,
# tunnel it locally, then run the base-vs-LoRA selector eval. Requires an AUTHENTICATED nebius CLI
# profile (run `nebius profile create` first — interactive/2FA) and an SSH keypair.
#
# Usage: scripts/nebius_serve_lora.sh <adapter_dir> <cache_jsonl> <tag>
#   e.g. scripts/nebius_serve_lora.sh adapters/gpt_oss_120b/hard_<jobid> gpt_oss/data/out/hard_cache.jsonl hard
#
# Env knobs: PLATFORM (default gpu-h200-sxm), PRESET (default 8gpu-...; auto-discovered if unset),
#   SSH_KEY (default ~/.ssh/id_ed25519), HF_TOKEN (for base weights), REGION (default eu-north1).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.nebius/bin:$PATH"

ADAPTER_DIR="${1:?adapter_dir}"; CACHE="${2:?cache_jsonl}"; TAG="${3:?tag}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
PLATFORM="${PLATFORM:-gpu-h200-sxm}"
VM_NAME="sia-lever-serve-${TAG}"

command -v nebius >/dev/null || { echo "nebius CLI missing"; exit 1; }
nebius iam whoami >/dev/null 2>&1 || { echo "CLI not authenticated — run: nebius profile create"; exit 1; }
[ -f "$SSH_KEY.pub" ] || { echo "no ssh pubkey at $SSH_KEY.pub — ssh-keygen -t ed25519"; exit 1; }

PROJECT_ID="$(nebius iam project list --format json 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['items'][0]['metadata']['id'])")"
echo "[orch] project=$PROJECT_ID platform=$PLATFORM"

if [ -z "${PRESET:-}" ]; then
  PRESET="$(nebius compute preset list --parent-id "$PROJECT_ID" --format json 2>/dev/null \
    | python3 -c "import json,sys;d=json.load(sys.stdin);ps=[p['metadata']['name'] for p in d.get('items',[]) if '$PLATFORM' in str(p)];print(ps[-1] if ps else '8gpu-128vcpu-1600gb')" 2>/dev/null || echo '8gpu-128vcpu-1600gb')"
fi
echo "[orch] preset=$PRESET"

echo "[orch] creating VM $VM_NAME ..."
nebius compute instance create \
  --parent-id "$PROJECT_ID" --name "$VM_NAME" \
  --resources-platform "$PLATFORM" --resources-preset "$PRESET" \
  --boot-disk-managed-disk-size-gibibytes 1024 \
  --boot-disk-managed-disk-source-image-family-image-family ubuntu22.04-cuda12 \
  --cloud-init-user-data "$(printf 'users:\n  - name: ubuntu\n    ssh_authorized_keys:\n      - %s\n' "$(cat "$SSH_KEY.pub")")" \
  --format json > /tmp/sia_vm_${TAG}.json
VM_ID="$(python3 -c "import json;print(json.load(open('/tmp/sia_vm_${TAG}.json'))['metadata']['id'])")"
echo "[orch] vm id=$VM_ID — waiting for RUNNING + public IP ..."

IP=""
for i in $(seq 1 40); do
  IP="$(nebius compute instance get --id "$VM_ID" --format json 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('status',{}).get('network_interfaces',[{}])[0].get('public_ip_address',{}).get('address','') )" 2>/dev/null || true)"
  [ -n "$IP" ] && break
  sleep 15
done
[ -n "$IP" ] || { echo "[orch] no public IP after wait — check console"; exit 1; }
echo "[orch] VM ip=$IP"

SSH="ssh -o StrictHostKeyChecking=no -i $SSH_KEY ubuntu@$IP"
echo "[orch] waiting for sshd ..."
for i in $(seq 1 40); do $SSH true 2>/dev/null && break; sleep 10; done

echo "[orch] rsync adapter + bootstrap ..."
rsync -az -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY" "$ADAPTER_DIR/" "ubuntu@$IP:~/adapter/"
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" scripts/remote_bootstrap_vllm.sh "ubuntu@$IP:~/"

echo "[orch] launching vLLM on VM (background) ..."
$SSH "HF_TOKEN='${HF_TOKEN:-}' nohup bash ~/remote_bootstrap_vllm.sh > ~/vllm.log 2>&1 &"
echo "[orch] waiting for vLLM ready (model load is slow for 120B) ..."
for i in $(seq 1 80); do
  $SSH "curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/v1/models" 2>/dev/null | grep -q 200 && { echo "[orch] vLLM up"; break; }
  sleep 30
done

echo "[orch] opening tunnel localhost:8001 -> VM ..."
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -N -L 8001:localhost:8001 "ubuntu@$IP" &
TUN=$!; sleep 3

echo "[orch] running base-vs-LoRA eval via local tunnel ..."
export GPT_OSS_BASE_URL="http://localhost:8001/v1/" GPT_OSS_API_KEY=dummy
python3 gpt_oss/rollout/rollout_base.py --model lever_lora --cache "$CACHE" --tag "lora_${TAG}" --eval-seeds 1
python3 gpt_oss/eval/eval_selector.py --rollouts "results/gpt_oss/lora_${TAG}_rollouts_*.jsonl" --cache "$CACHE" --tag "lora_${TAG}"
python3 gpt_oss/eval/eval_adapter.py \
  --base-rollouts "results/gpt_oss/base_${TAG}_rollouts_*.jsonl" \
  --adapter-rollouts "results/gpt_oss/lora_${TAG}_rollouts_*.jsonl" --tag "$TAG" --cache "$CACHE"

kill $TUN 2>/dev/null || true
echo "[orch] DONE. VM $VM_ID still running — DELETE to stop billing:"
echo "       nebius compute instance delete --id $VM_ID"
