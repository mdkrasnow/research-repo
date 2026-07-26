# Exact launch commands

Training continuations were submitted remotely with:

```text
sbatch --parsable --export=ALL,GIT_SHA=696c49c,EBM={none|dot|direct},GLOBAL_SEED={0|1|2},CKPT=/n/home03/mkrasnow/direct_energy_full_retry/{epoch1 checkpoint},RUN_TAG=longer_{arm}_seed{seed},RESULTS_ROOT=/n/home03/mkrasnow/direct_energy_longer_retry projects/masked-EqM/slurm/jobs/direct_energy_longer_train.sbatch
```

The checkpoint field is resolved to the exact seed-matched epoch-1 file at
submission time. The diagnostic retry was submitted with `GIT_SHA=5dfd9b1` and
all nine epoch-1 checkpoint specifications.
