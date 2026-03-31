# Results — q201_20260219_baseline_100k_fixed

## Job Info
- Job ID: 61220455
- Git SHA: 87dc0d1 (batch-size-invariant grad fix)
- Array: 0-9%2, Partition: gpu
- Config: q201_baseline.json (vector energy head, adversarial mining, opt_steps=10)

## Seed Results

| Seed | Status | Val MSE |
|------|--------|---------|
| 0 | FAILED (exit 120, node crash) | — |
| 1 | FAILED (exit 120, node crash) | — |
| 2 | FAILED (exit 120, node crash) | — |
| 3 | FAILED (exit 120, node crash) | — |
| 4 | COMPLETED | 0.25872 |
| 5 | COMPLETED | 0.268939 |
| 6 | COMPLETED | 0.282036 |
| 7 | COMPLETED | 0.237189 |
| 8 | COMPLETED | 0.240538 |
| 9 | COMPLETED | 0.267932 |

## Statistics (6 seeds)
- Mean: 0.259226
- Std: 0.015946
- Min: 0.237189
- Max: 0.282036

## Key Finding
Vector head + adversarial mining with opt_steps=10 is catastrophically bad (~27x worse than baseline).
Compare: q103 (vector head, adversarial, opt_steps=2) got 0.00977 — only 0.8% worse than baseline.
Conclusion: opt_steps=10 adversarial mining destroys performance regardless of energy head type.
The IRED-CD ablation config (opt_steps=10) is too aggressive for adversarial mining.
