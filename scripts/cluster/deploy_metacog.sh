#!/usr/bin/env bash
# Metacog policy method-improvement sweep (EqM B/2, inference-time).
#  * SELECTION screen (matched NFE=750, seed0, n=$NSCREEN): random, energy_path,
#    probe_k50 (promotion baseline), stacked_selector, smc_metacog, multiread_triage.
#  * SEGMENTED smokes (n=256): churn_rescue, heun_corrector, optimizer_switch,
#    risk_compute_allocator — validate real-model NFE/no-crash before any full screen.
# Reuses pareto incremental-FID (disk-safe). Run from repo root via ssh.sh.
# Every job: nvidia-smi guard (fail-fast on bad GPU nodes) + logs its command.
set -uo pipefail
RR=/n/home03/mkrasnow/research-repo
D=$RR/projects/diff-EqM/experiments/separability_diagnostic
CKPT=$RR/projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt
PROBE=$D/runs/b2_vanilla/probe_artifact.npz
REF=$RR/projects/diff-EqM/results/in1k_reference_stats.npz
ROOT=$D/runs/b2_vanilla/metacog
NSCREEN=${NSCREEN:-10000}
mkdir -p "$ROOT" "$RR/slurm/logs"

ML="module load python/3.10.13-fasrc01 cuda/11.8.0-fasrc01"

# ---- build frozen stacked-ranker artifacts (DEV labels only), once ----------
$ML 2>/dev/null
python -c "import sys; sys.path.insert(0,'$D'); import metacog_policies as M; \
[print('built',M.build_stacked_artifact('$D/runs/b2_vanilla',k=k).name) for k in (50,75,100)]" \
  || { echo 'STACKED BUILD FAILED'; exit 2; }

submit() {  # $1=name $2=engine $3=policy $4=policy_kw_json $5=nslots $6=part $7=tlimit
  local name="$1" engine="$2" policy="$3" kw="$4" ns="$5" part="$6" tl="$7"
  local out=$ROOT/$name
  local wrap="$ML; cd $RR; nvidia-smi -L || exit 99; \
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
P=\$((29500+RANDOM%1000)); \
echo CMD: $policy $engine $kw n=$ns; \
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=\$P \
$D/run_metacog_policy_sweep.py --ckpt $CKPT --probe-artifact $PROBE --out $out \
--engine $engine --policy $policy --policy-kw '$kw' --num-slots $ns --steps 250 --R 3"
  local jid=$(sbatch -p $part --gres=gpu:4 -c 32 --mem=256G -t $tl \
    -J mc_$name -o $RR/slurm/logs/mc_${name}_%A.out --wrap="$wrap" 2>&1 | awk '{print $NF}')
  echo "submitted $name -> job $jid ($part, $engine, n=$ns)"
}

# ---- SELECTION screen (matched NFE=750), seed0 -----------------------------
submit random           selection random            '{}'                 $NSCREEN gpu      10:00:00
submit energy_path      selection energy_path        '{}'                 $NSCREEN seas_gpu 10:00:00
submit probe_k50        selection probe_k            '{"k":50}'           $NSCREEN gpu      10:00:00
submit stacked_selector selection stacked_selector   '{"k":50}'           $NSCREEN seas_gpu 10:00:00
submit smc_metacog      selection smc_metacog        '{"k":50,"beta":8}'  $NSCREEN gpu      10:00:00
submit multiread_triage selection multiread_triage   '{"hi":0.85}'        $NSCREEN seas_gpu 10:00:00

# ---- SEGMENTED smokes (n=256), validate real-model NFE + no-crash ----------
submit smoke_churn   segmented churn_rescue           '{"hi":0.7,"sigma":0.3}' 256 gpu      01:30:00
submit smoke_heun    segmented heun_corrector         '{"hi":0.7}'             256 seas_gpu 01:30:00
submit smoke_optsw   segmented optimizer_switch       '{"hi":0.7}'             256 gpu      01:30:00
submit smoke_alloc   segmented risk_compute_allocator '{"frac":0.33}'          256 seas_gpu 01:30:00

echo "ALL_SUBMITTED: 6 selection screen arms (n=$NSCREEN) + 4 segmented smokes (n=256)"
echo "AGGREGATE when done: python $D/aggregate_policy_sweep.py --root $ROOT --baseline probe_k50 --ref-stats $REF"
