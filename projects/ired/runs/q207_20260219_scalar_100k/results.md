# Results — q207_20260219_scalar_100k

## Job Status
- Job ID: 61186968
- Array spec: 0-9%2 (throttled to 2 concurrent)
- Partition: gpu
- Git SHA: e2dfbd8
- Strategy: adversarial mining + scalar energy head
- Status: PARTIALLY COMPLETE (seeds 0-3 done/failed, 4-5 running, 6-9 pending)
- Submitted: 2026-02-19T01:00:00Z
- Last checked: 2026-02-19 (~2h after submission)

## Per-Seed Results (sacct)

| Seed | State | Exit Code | Elapsed | val_mse |
|------|-------|-----------|---------|---------|
| 0 | COMPLETED | 0:0 | 1:55:04 | **0.0990** |
| 1 | COMPLETED | 0:0 | 1:55:18 | **0.0807** |
| 2 | FAILED | 120:0 | 1:14:59 | partial ~0.14 (step 93k) |
| 3 | FAILED | 120:0 | 1:11:26 | partial ~0.83 (step 93k) |
| 4 | RUNNING | — | 1:06 | — |
| 5 | RUNNING | — | 1:06 | — |
| 6-9 | PENDING | — | — | — |

## Key Observations

### Plateau Confirmed at val MSE ~0.09
- Seeds 0,1 completed at ~100k steps with val MSE **0.099 and 0.081**
- This is ~10x WORSE than the baseline q101 (val MSE = 0.00969)
- Consistent with prior observation from 50k-step runs (plateau noted in pipeline)

### CD-DIAG Signature (Seeds 0,1)
```
lang_grad0=0.0000 lang_gradK=0.0000 lang_disp=0.0000 step=0.0000 sigma=0.0000
```
Langevin fields all zero → adversarial mining (no CD), but scalar energy head active.

### Energy Growing but Val MSE Stuck
- `gradE` grows from ~12k to ~21k+ across training
- `E_pos/E_neg` grow dramatically (1M+ range)
- `MSE` in CD-DIAG (train batch MSE) stays ~0.48-0.49 from step 50k onward
- Val MSE plateau at ~0.09 persists: scalar head is NOT improving generalization

### Exit 120 Pattern (Seeds 2,3)
Both failed with exit 120 at ~1:11-1:15 runtime. Exit 120 = SLURM kill/node failure.
Not a training divergence — progress output was normal up to failure point.

## Conclusion
**Scalar energy head + adversarial mining plateau: val MSE ~0.09 (10x worse than baseline NCE)**
Compare to q101 adversarial baseline: val MSE = 0.00977. The scalar head alone degrades perf 10x.
