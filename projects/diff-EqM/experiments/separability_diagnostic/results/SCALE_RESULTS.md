# Scale results — EqM trajectory-metacognition (2026-06-14)

Decision-grade. All vs trusted 50k `in1k_reference_stats.npz`.

## Phase 1 — 50k probe-gated consistency (3 seeds, parallel)
Jobs 22931315/22931323/22931328 (gpu, 4×A100 each) + agg 22931344.

| seed | vanilla | probe | oracle | Δ(van−probe) | recovered |
|---|---|---|---|---|---|
| 0 | 28.199 | 26.205 | 16.258 | 1.995 | 16.7% |
| 1 | 27.783 | 25.952 | 16.131 | 1.831 | 15.7% |
| 2 | 27.833 | 26.037 | 16.123 | 1.796 | 15.3% |
| **mean±std** | 27.94±0.23 | 26.06±0.13 | 16.17±0.08 | **1.87±0.11** | 15.9% |

95% CI on the gain = ±0.12 (excludes 0). **VERDICT: CONSISTENT** — probe beats
vanilla on all 3 seeds; identical NFE (R×N=750/slot). Answers Yilun's "are gains
consistent?" at the real metric.

## Phase 2 — online equal-NFE adaptive sampler (15k)
Job 22975626 (gpu, 4×A100). k_dec=100, flag-frac 0.3.

| arm | role | FID |
|---|---|---|
| vanilla | un-adapted floor | 29.55 (sanity OK vs 31.41) |
| random-restart | NEG, compute-matched | 29.76 |
| **probe-restart** | TREATMENT | **28.51** |
| oracle-restart | POS ceiling | 23.32 |

probe-restart < random-restart **Δ1.24 at EQUAL NFE**, 19% of oracle. **VERDICT:
WORKS** — restart only the probe-flagged slots mid-flight, beats restarting a random
equal-size subset. True online metacognition at scale, vanilla sanity restored.

## Lineage (direction stable across scale)
best-of-R restart Δ: 15k 1.69 → smoke-2k 1.82/2.45 → **50k 3-seed 1.87±0.11**.
online equal-NFE Δ: mock(quality) → 512 1.87(relative) → **15k 1.24(sanity OK)**.
