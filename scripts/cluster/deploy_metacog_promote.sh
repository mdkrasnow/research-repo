#!/usr/bin/env bash
# Promotion run for metacog policies that beat probe_k50 at matched NFE in the
# n=10k seed0 screen. Fires the promoted SELECTION arms (+ baselines for paired
# deltas) at n=50000 across SEEDS, paired by construction (seed-offset shifts the
# slot->seed map identically for every arm). Reuses run_metacog_policy_sweep.py.
#
# Usage (from repo root via ssh.sh):
#   ARMS="stacked_selector|stacked_selector|{\"k\":50} probe_k50|probe_k|{\"k\":50}" \
#   SEEDS="0 1 2 3 4" NSLOTS=50000 bash scripts/cluster/deploy_metacog_promote.sh
# ARMS = space-separated  name|policy|kw_json  triples. Default = full headline set.
set -uo pipefail
RR=/n/home03/mkrasnow/research-repo
D=$RR/projects/diff-EqM/experiments/separability_diagnostic
CKPT=$RR/projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt
PROBE=$D/runs/b2_vanilla/probe_artifact.npz
ROOT=$D/runs/b2_vanilla/metacog_promote
NSLOTS=${NSLOTS:-50000}
SEEDS=${SEEDS:-"0 1 2 3 4"}
ARMS=${ARMS:-"probe_k50|probe_k|{\"k\":50} energy_path|energy_path|{} random|random|{}"}
mkdir -p "$ROOT" "$RR/slurm/logs"
ML="module load python/3.10.13-fasrc01 cuda/11.8.0-fasrc01"

i=0
for triple in $ARMS; do
  name="${triple%%|*}"; rest="${triple#*|}"; policy="${rest%%|*}"; kw="${rest#*|}"
  for s in $SEEDS; do
    out=$ROOT/${name}_s${s}
    if (( i % 2 == 0 )); then PART=gpu; else PART=seas_gpu; fi
    i=$((i+1))
    wrap="$ML; cd $RR; nvidia-smi -L || exit 99; \
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
P=\$((29500+RANDOM%1000)); \
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=\$P \
$D/run_metacog_policy_sweep.py --ckpt $CKPT --probe-artifact $PROBE --out $out \
--engine selection --policy $policy --policy-kw '$kw' --num-slots $NSLOTS --steps 250 --R 3 \
--seed-offset $s"
    jid=$(sbatch -p $PART --gres=gpu:4 -c 32 --mem=256G -t 10:00:00 \
      -J mcp_${name}_s${s} -o $RR/slurm/logs/mcp_${name}_s${s}_%A.out --wrap="$wrap" 2>&1 | awk '{print $NF}')
    echo "submitted ${name}_s${s} -> job $jid ($PART)"
  done
done
echo "ALL_SUBMITTED promotion: ARMS=[$ARMS] SEEDS=[$SEEDS] N=$NSLOTS"
