# Results — q208_20260219_scalar_cl_100k_fixed

## Job Info
- Job ID: 61220454
- Git SHA: 87dc0d1 (batch-size-invariant grad fix)
- Array: 0-9%2, Partition: gpu
- Config: q208_scalar_contrastive.json (IRED-CL, Langevin, replay buffer)

## Seed Results

| Seed | Status | Val MSE |
|------|--------|---------|
| 0 | FAILED (exit 120, node crash) | — |
| 1 | OOM | — |
| 2 | FAILED (exit 120, node crash) | — |
| 3 | FAILED (exit 120, node crash) | — |
| 4 | FAILED (exit 120, node crash) | — |
| 5 | OOM (2h runtime) | — |
| 6 | COMPLETED (3h) | 1.8672 |
| 7 | OOM | — |
| 8 | COMPLETED (3h) | 1.80081 |
| 9 | COMPLETED (3h) | 1.796 |

## Statistics (3 seeds)
- Mean: 1.821337
- Std: 0.032490

## Key Finding
IRED-CL (q208) completely fails — val MSE ~1.82, 188x worse than baseline.
Three root causes diagnosed (see debugging.md):
1. energy_reg_weight=0 → energies reach ~440k → softplus saturates immediately (grad ≈ 0)
2. 10 Langevin steps → avg displacement 33.7 → all negatives clamped to boundary [-2,2]
3. Langevin used B_norm-normalized gradient (inconsistent with sigma scale)
Fix implemented in q210 (energy_reg_weight=0.01, opt_steps=3, energy.mean() grad).
High OOM rate suggests Langevin+replay buffer is memory-intensive; opt_steps=3 should help.
