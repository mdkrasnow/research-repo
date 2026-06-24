#!/usr/bin/env bash
# 5-seed CI on the Pareto headline (PI item 1). Reuses pareto_sample.py (incremental-FID,
# disk-safe — avoids the per-feature dump that OOM'd 3/5 selector seeds). Seed0 already exists
# from the original pareto run (seed_offset=0); this fires seeds 1-4 for the 4 headline arms.
# Partition-split to dodge QOSMaxSubmitJobPerUserLimit. Run from repo root via ssh.sh.
set -uo pipefail
RR=/n/home03/mkrasnow/research-repo
D=$RR/projects/diff-EqM/experiments/separability_diagnostic
CKPT=$RR/projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt
PROBE=$D/runs/b2_vanilla/probe_artifact.npz
REF=$RR/projects/diff-EqM/results/in1k_reference_stats.npz
CI=$D/runs/b2_vanilla/ci
mkdir -p "$CI"

# arm: name|mode-args
ARMS=(
  "long250|--mode long --steps 250"
  "r3rand|--mode restart --select random --R 3 --steps 250"
  "r3energy|--mode restart --select energy_path --R 3 --steps 250"
  "r3probe|--mode restart --select probe_k --k-dec 50 --R 3 --steps 250"
)

i=0
for arm in "${ARMS[@]}"; do
  name="${arm%%|*}"; margs="${arm#*|}"
  for s in 1 2 3 4; do
    out=$CI/${name}_s${s}
    # alternate partitions to spread QOS load
    if (( i % 2 == 0 )); then PART=gpu; else PART=gpu_test; fi
    i=$((i+1))
    wrap="module load python/3.10.13-fasrc01 cuda/11.8.0-fasrc01; cd $RR; \
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
P=\$((29500+RANDOM%1000)); \
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=\$P \
$D/pareto_sample.py --ckpt $CKPT --probe-artifact $PROBE --out $out \
$margs --num-slots 50000 --seed-offset $s; \
python $D/pareto_agg.py --out $out --ref-stats $REF"
    jid=$(sbatch -p $PART --gres=gpu:4 -c 32 --mem=256G -t 06:00:00 \
      -J ci_${name}_s${s} -o $RR/slurm/logs/ci_${name}_s${s}_%A.out \
      --wrap="$wrap" 2>&1 | awk '{print $NF}')
    echo "submitted ci_${name}_s${s} -> job $jid ($PART)"
  done
done
echo "ALL_SUBMITTED 16 jobs (seeds 1-4 x 4 arms); seed0 = original pareto run"
