#!/usr/bin/env bash
set -euo pipefail

mode="${1:?direct or none}"
if [[ "$mode" == direct ]]; then
  job_name=direct-fid2k
  wrapper=projects/masked-EqM/slurm/jobs/direct_energy_eval_fid2k.sbatch
  ckpt=/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/direct_clipped_epoch40_to_epoch80_job36847271/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/epoch80.pt
  ebm=direct
  out_name=direct_clipped_epoch80_fid2k_smoke
elif [[ "$mode" == none ]]; then
  job_name=none-fid2k
  wrapper=projects/masked-EqM/slurm/jobs/none_energy_eval_fid2k.sbatch
  ckpt=/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer80_none_seed0_ckpt50k_job36632776/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch80.pt
  ebm=none
  out_name=none_epoch80_fid2k_eta003
else
  echo "unknown mode: $mode" >&2
  exit 2
fi

while true; do
  if scripts/cluster/ssh.sh "squeue -u \$USER -h -n $job_name | grep -q $job_name"; then
    exit 0
  fi
  source_sha=$(git rev-parse HEAD)
  source_archive="/n/home03/mkrasnow/masked_eqm_source_archives/${source_sha}.tar.gz"
  exports="CKPT=$ckpt,OUT_NAME=$out_name,EBM=$ebm,NUM_SAMPLES=2000,NUM_SAMPLING_STEPS=250,STEPSIZE=0.003,SOURCE_ARCHIVE=$source_archive"
  if job_id=$(SBATCH_EXPORTS="$exports" scripts/cluster/remote_submit.sh "$wrapper" masked-EqM 2>&1); then
    state=$(scripts/cluster/ssh.sh "sacct -j $job_id -X -o State -n -P | head -1" || true)
    if [[ "$state" != FAILED* && "$state" != CANCELLED* && "$state" != TIMEOUT* ]]; then
      echo "$mode smoke FID submitted as $job_id"
      exit 0
    fi
    echo "$mode smoke FID $job_id failed immediately with state $state; retrying" >&2
  else
    echo "$mode smoke FID submission deferred: $job_id" >&2
  fi
  sleep 300
done
