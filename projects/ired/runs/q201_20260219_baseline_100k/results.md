# Results — q201_20260219_baseline_100k

## Job Status
- Job ID: 61186975
- Array spec: 0-9%2 (throttled to 2 concurrent)
- Partition: gpu
- Git SHA: e2dfbd8
- Strategy: adversarial (CD baseline — same NCE code with adversarial mining)
- Status: PARTIALLY COMPLETE (seeds 0-1 done, 2-3 failed, 4-5 running, 6-9 pending)
- Submitted: 2026-02-19T01:00:00Z
- Last checked: 2026-02-19 (~2h after submission)

## Per-Seed Results (sacct)

| Seed | State | Exit Code | Elapsed | val_mse |
|------|-------|-----------|---------|---------|
| 0 | COMPLETED | 0:0 | 1:54:16 | **1.704** |
| 1 | COMPLETED | 0:0 | 1:56:57 | **1.105** |
| 2 | FAILED | 120:0 | 1:10:40 | partial (step ~4k) |
| 3 | FAILED | 120:0 | 1:07:23 | partial (step ~1.4k) |
| 4 | RUNNING | — | 1:06 | — |
| 5 | RUNNING | — | 1:06 | — |
| 6-9 | PENDING | — | — | — |

## Analysis

### Catastrophically Bad vs Previous Baseline
- Seeds 0,1 val MSE: **1.10–1.70**
- Previous q101 baseline (same strategy): val MSE = **0.00969**
- This is **110-175x WORSE** than the prior identical experiment

### CD-DIAG Signature (Seeds 2,3 partial logs)
At steps 1k-4.5k:
```
[CD-DIAG] E_pos growing exponentially (2.4k → 841k)
[CD-DIAG] E_wtd fluctuating wildly (0.0009 to 39.58)
[CD-DIAG] MSE(batch) = 0.58-1.00 (still very high at step 4k)
[CD-DIAG] lang_grad0=0.0000 (Langevin inactive)
```

### What Changed Between q101 and q201?
q101 baseline (val MSE 0.009): git_sha=fef8849, strategy=none (no mining), no CD-DIAG code
q201 current (val MSE 1.70): git_sha=e2dfbd8, strategy=adversarial, WITH CD-DIAG diagnostics

**Critical suspect**: The CD-DIAG diagnostic code or scalar energy head changes in e2dfbd8 may be corrupting the baseline NCE training path. The adversarial mining in q201 previously reached val MSE 0.00977 (q103 multiseed avg). Now it's at 1.70.

### Exit 120 Pattern (Seeds 2,3)
Both failed at ~1:07-1:10 runtime with exit 120. Same pattern as q207 seeds 2,3.
Consistent with node failures on specific compute nodes.

## Conclusion
**This baseline (q201) is broken vs. q101 which used identical strategy.**
The ~2-3 months of code changes between fef8849 and e2dfbd8 introduced a regression
that breaks even the basic adversarial baseline. Root cause investigation needed.
