# Bundled frozen-prior workflow

FASRC counts array indices as individual jobs.  This workflow therefore uses
ordinary batch jobs only; every command is resume-safe because per-record JSONL
records are skipped only when their exact record ID already exists.

* Smoke: one `STAGE=smoke` job evaluates representative Gaussian, Bernoulli,
  and mixed checkpoints sequentially with none/hard projection.
* Pilot: `STAGE=pilot BUNDLE_ID=0`, then confirm with `BUNDLE_ID=1`.  Each
  job sequentially evaluates all three matched arms and rho {0.25,0.5,0.75}.
* Final: one `STAGE=final BUNDLE_ID=<0..8> SOFT_RHO=<locked>` job per
  checkpoint, with `SHARD_START..SHARD_END` selecting several shards per job.
  Submit conservatively as capacity permits.  A requeue/timeout reruns only
  missing records.
* Aggregate only after all nine checkpoints and all final shards are present.

All jobs record immutable git SHA, manifest hash, checkpoint ID, stage, shard,
and output directory in the log and JSONL.
