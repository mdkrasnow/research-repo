# Results — q207_20260219_scalar_100k_fixed

## Job Info
- Job ID: 61220453
- Git SHA: 87dc0d1 (batch-size-invariant grad fix)
- Array: 0-9%2, Partition: gpu
- Config: q207_scalar_baseline.json (use_scalar_energy=true, adversarial mining, opt_steps=10)

## Seed Results

| Seed | Status | Val MSE |
|------|--------|---------|
| 0 | FAILED (exit 120, node crash) | — |
| 1 | FAILED (exit 120, node crash) | — |
| 2 | FAILED (exit 120, node crash) | — |
| 3 | FAILED (exit 120, node crash) | — |
| 4 | COMPLETED | 0.331426 |
| 5 | COMPLETED | 0.357987 |
| 6 | COMPLETED | 0.380445 |
| 7 | COMPLETED | 0.246895 |
| 8 | COMPLETED | 0.296778 |
| 9 | COMPLETED | 0.246052 |

## Statistics (6 seeds)
- Mean: 0.309930
- Std: 0.051592
- Min: 0.246052
- Max: 0.380445

## Key Finding
Scalar head + adversarial mining (opt_steps=10) is catastrophically bad (~32x worse than baseline).
Compare: vector head + adversarial mining (q201) also catastrophic at 0.259.
This confirms adversarial mining with opt_steps=10 destroys both scalar AND vector heads.
The adversarial mining config (opt_steps=10) is far too aggressive — q103 used opt_steps=2 and got 0.00977.
