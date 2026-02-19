# Results — q208_20260219_scalar_cl_100k

## Job Status
- Job ID: 61186972
- Array spec: 0-9%2 (throttled to 2 concurrent)
- Partition: gpu
- Git SHA: e2dfbd8
- Strategy: cd_langevin_replay (scalar contrastive + Langevin + replay buffer)
- Status: PARTIALLY COMPLETE (seed 0 done, 1-2 OOM, 3-4 failed, 5-6 running, 7-9 pending)
- Submitted: 2026-02-19T01:00:00Z
- Last checked: 2026-02-19 (~2h after submission)

## Per-Seed Results (sacct)

| Seed | State | Exit Code | Elapsed | val_mse |
|------|-------|-----------|---------|---------|
| 0 | COMPLETED | 0:0 | 2:57:30 | **1.460** |
| 1 | OUT_OF_MEMORY | 0:125 | 0:23:34 | — |
| 2 | OUT_OF_MEMORY | 0:125 | 2:19:00 | — |
| 3 | FAILED | 120:0 | 0:25:04 | — |
| 4 | FAILED | 120:0 | 0:11:00 | — |
| 5 | RUNNING | — | 1:06 | — |
| 6 | RUNNING | — | 1:06 | — |
| 7-9 | PENDING | — | — | — |

## Completed Seed 0 Results

### val MSE progression:
- val MSE stays ~1.46-1.47 across all checkpoints observed
- train MSE ~1.83 (also very bad)

### Catastrophic Performance
val MSE = **1.460** vs baseline q101 val MSE = **0.00969**
**~150x worse than NCE baseline.**

## OOM Analysis (Seeds 1, 2)
- Seeds 1,2 hit OUT_OF_MEMORY (exit 125)
- Langevin sampling + replay buffer requires more GPU memory than simple adversarial
- Seed 2 lasted 2:19:00 before OOM — suggesting memory leak or growing replay buffer
- Seed 1 failed quickly (23 min) — possibly worse memory allocation

## Exit 120 Analysis (Seeds 3, 4)
- Seeds 3,4 failed very quickly (25 min, 11 min) with exit 120
- Could be node failures or resource contention

## Conclusion
**Scalar contrastive with Langevin+replay: val MSE ~1.46 (150x worse than baseline)**
This is catastrophically bad. The scalar energy head with CD training is not functioning.
OOM issues on ~40% of seeds suggest memory problem with current implementation.
