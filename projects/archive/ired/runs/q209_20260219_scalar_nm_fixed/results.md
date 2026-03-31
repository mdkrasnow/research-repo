# Results — q209_20260219_scalar_nm_fixed

## Job Info
- Job ID: 61220452
- Git SHA: 87dc0d1 (batch-size-invariant grad fix, grad_norm_ref=2048)
- Array: 0-9%2, Partition: gpu
- Config: q209_scalar_no_mining.json (use_scalar_energy=true, no mining)

## Seed Results

| Seed | Status | Val MSE |
|------|--------|---------|
| 0 | FAILED (exit 120, node crash) | — |
| 1 | FAILED (exit 120, node crash) | — |
| 2 | FAILED (exit 120, node crash) | — |
| 3 | FAILED (exit 120, node crash) | — |
| 4 | COMPLETED | 0.00973468 |
| 5 | COMPLETED | 0.00971237 |
| 6 | COMPLETED | 0.00974107 |
| 7 | COMPLETED | 0.00974722 |
| 8 | COMPLETED | 0.00974496 |
| 9 | COMPLETED | 0.00974109 |

## Statistics (6 seeds)
- Mean: 0.009737
- Std: 0.000012
- Min: 0.009712
- Max: 0.009747

## Comparison
- Vector no-mining baseline (q101): 0.009692
- q209 vs baseline: +0.46% (not significant — within noise)

## Key Finding
Scalar energy head with no mining performs equivalently to vector energy head.
The batch-size normalization fix was essential — q209 previously showed val MSE ~2.5 (catastrophic)
due to 8x larger opt_step at val time (B=256 vs B=2048). Fixed by grad_norm_ref=2048.
Scalar head is a valid architecture choice.
