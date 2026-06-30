# Selector replication — S/2 vanilla seed0 (scale-axis test)

Locked reference B/2 seed0: probe-restart 24.66±0.16, SHAPE-deconf AUROC 0.813±0.002.
B/2 seed1 replication: probe-restart 24.81, AUROC 0.746 (s4 label sanity 0.577 weak).

## Step 1 — Separability diagnostic (job 26334954, gpu_test, COMPLETED 21m, 3000 samples)
Scalars dead: best_independent_auroc = 0.593 (s8 norm-osc). s4 label sanity = 0.576 (WEAK — same
pattern as B/2 seed1's 0.577). Reproduces "energy scalars dead" finding at second model scale.

## Step 2 — Learned probe over descent dynamics (CPU, test partition)
| probe group | S/2 seed0 within-norm AUROC | B/2 seed1 | locked B/2 seed0 |
|---|---|---|---|
| MAG-only | 0.587 | 0.626 | ~0.61 |
| SHAPE-only (de-conf) | 0.734 (learned) / 0.736±0.009 (5-seed) | 0.746±0.007 | 0.813±0.002 |
| FULL | 0.728 | 0.745 | — |

Mechanism replicates a third time: SHAPE ≫ MAG, FULL≈SHAPE, weak label sanity depresses absolute
AUROC but pattern intact. probe_artifact.npz (dim=30) saved to runs/s2_seed0_repl/.

## Step 3 — Selector smoke (job 26392208, gpu_test, NUM_SLOTS=512 R=3, MODEL=EqM-S/2)
SUBMITTED 2026-06-30. Pending result — poll before launching full 50k.

## Step 3 (cont.) — Smoke result (job 26392208, COMPLETED 10m35s, NUM_SLOTS=512)
```
vanilla  n=512  FID=123.196
probe    n=512  FID=120.376
oracle   n=512  FID=100.569
```
sanity: vanilla 123.20 vs known baseline 31.41 -> MISMATCH (small-n smoke + ref-stats artifact; interpret
deltas only, matches known smoke-scale behavior). probe<vanilla, Δ2.82, 12% oracle recovery — direction
SANE, consistent with locked result. Launched FULL 50k: job **26395188**, seas_gpu, 4-GPU.
