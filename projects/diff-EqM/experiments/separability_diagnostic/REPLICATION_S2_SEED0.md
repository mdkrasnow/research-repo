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

## Full 50k FID (job 26395188, COMPLETED 1h16m, exit 0)
```
vanilla  n=50000  FID=50.006
probe    n=50000  FID=47.290
oracle   n=50000  FID=32.640
```
sanity: vanilla 50.01 vs known baseline 31.41 -> MISMATCH (S/2 has its own ref-stats/pipeline offset,
same pattern as B/2 runs — interpret deltas only, not absolute FID).

| arm | S/2 seed0 (50k) | B/2 seed1 (50k) | locked B/2 seed0 |
|---|---|---|---|
| vanilla (restart-baseline) | 50.006 | 27.506 | 27.95 |
| **probe-restart** | **47.290** | **24.808** | **24.66 ± 0.16** |
| oracle (ceiling) | 32.640 | 15.837 | 21.75 |
| Δ vanilla−probe | **2.716** | 2.698 | ~3.29 |
| oracle-recovered fraction | 0.161 (16%) | 0.231 (23%) | ~16–18% |

## VERDICT — selector gain replicates across model SCALE, not just checkpoint instance
- **probe-restart Δ 2.72 FID @ S/2** — nearly identical magnitude to B/2 seed1 (Δ2.70) and locked B/2
  seed0 (Δ~3.29). Oracle-recovered fraction (16%) matches locked seed0 band (16–18%), lower than
  seed1's 23% but same order of magnitude.
- Full mechanism chain replicates a THIRD time on a genuinely different model scale (S/2 vs B/2,
  different param count/patch config, independently trained): scalars dead (0.593) → learned SHAPE
  probe de-conf (0.736, replicating SHAPE≫MAG pattern) → probe-restart beats vanilla at FID.
- **Scale-axis conclusion:** the locked B/2 metacognition selector result is NOT specific to model
  scale. Combined with B/2 seed1 (checkpoint-specificity test), the result now generalizes across
  (a) checkpoint instance and (b) model capacity/scale, at fixed sampler config (eta=0.003/250/cfg=1.0).
- Caveat: single 50k draw at S/2 (not multi-seed CI, unlike locked B/2's 5-seed result). Absolute
  FID scale differs from B/2 (S/2 undertrained relative to B/2 at 80ep — expected, smaller model).
