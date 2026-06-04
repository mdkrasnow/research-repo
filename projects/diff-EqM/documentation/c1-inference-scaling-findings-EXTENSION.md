# C1 — Inference-Compute Scaling: EXTENSION result (high-NFE)

Pre-registered: `c1-inference-compute-scaling-proposal.md`. Zero-compute pre-analysis:
`c1-inference-scaling-findings.md` (leaned negative; extension was the decider for the
untested nfe>250 regime). Jobs **18949428** (l01: vanilla+λ0.1) + **18952135** (l03: λ0.3),
gpu_test/gpu_requeue. BOTH FAILED exit 1 on **home-full (OSError Errno 28)** near the end, but
captured the decisive cells. Absolute FIDs are 5K-inflated (read deltas, per Exp 1).

## Captured data
- l01: vanilla **gd** nfe{250,500,1000} (3 cells; crashed before vanilla-ngd + the entire anm_l01 arm —
  nfe1000 alone took 5.2 h/cell, so the 12-cell grid was wildly under-budgeted).
- l03: anm **λ=0.3** all 6 cells (gd+ngd × nfe{250,500,1000}) — complete.

| FID (5K) | nfe250 | nfe500 | nfe1000 | rise past min |
|---|---|---|---|---|
| vanilla gd | 45.71 | 45.73 | 47.88 | **+2.17** |
| anm λ0.3 gd | 41.25 | 41.73 | 45.27 | **+4.02** |
| anm λ0.3 ngd | 41.11 | 41.57 | 45.10 | +3.99 |

Final-step residual ‖f‖ (mean_final_grad_norm):
| | nfe250 | nfe500 | nfe1000 |
|---|---|---|---|
| vanilla gd | 5.19 | 4.18 | 4.11 |
| anm λ0.3 gd | 6.11 | 5.20 | 5.32 |

## Verdict — NEGATIVE / KILL the inference-compute-scaling capability

The pre-registered promotion rule (vanilla turns up ≥0.5 past min WHILE ANM stays monotone to ≥2×
nfe, dose-ordered, ANM lower residual) is refuted on every clause:

1. **No crossover.** BOTH arms turn up past nfe250. ANM degrades MORE (+4.02) than vanilla (+2.17),
   not less. There is no regime where more solver steps help ANM while hurting vanilla — extra
   optimization hurts both, ANM worse.
2. **Advantage shrinks with compute.** ANM−vanilla (gd) = −4.46 (nfe250) → −4.00 (500) → −2.61 (1000).
   The FID gain ERODES as inference compute grows — the opposite of a scaling capability.
3. **Higher residual, not lower.** ANM's final ‖f‖ (5.3–6.1) exceeds vanilla's (4.1–5.2). ANM does
   NOT reach a lower-energy equilibrium; its field stays larger-magnitude at convergence. (Consistent
   with C3: v10 in-dist field-norm is higher — the field doesn't vanish as hard at the data point.)

Missing arms (anm_l01, vanilla-ngd) cannot rescue the claim: the STRONGEST dose (λ0.3) already
degrades most with NFE, so λ0.1 would be intermediate — the monotone-in-dose degradation is the wrong
sign for the capability regardless.

## What survives (not a capability)
ANM λ0.3 beats vanilla at EVERY NFE by ~3–4.5 FID (gd: −4.46 to −2.61). This is the uniform quality
gain (dose-confirmed: bigger than λ0.1's ~2 FID from the zero-compute pass), matching the 50K headline
(λ0.3 FID 27.09 vs vanilla 31.41, Δ−4.3). Quality result, workshop-tier — NOT inference scaling.

## Decision (pre-registered)
- **C1 killed** per kill rule. No retune (the mechanism, not an HP, produces the anti-scaling sign).
- **C2 (restoration) does NOT launch.** Its gate required C1 high-nfe crossover OR C3 ΔAUROC>0
  dose-ordered. C1 → no crossover (anti-scaling); C3 → v10 worse, dose-anti-ordered. Neither capability
  probe cleared → no positive evidence to justify C2's GPU spend. Holding C2 is the pre-registered call.
- Both C3 and C1 negatives share ONE clean mechanism: v10's off-manifold field-robustness (Exp 2)
  raises the field magnitude at/near data and lowers it far off-manifold → worse OOD novelty signal
  (C3) and no lower-energy equilibrium / no extra-step refinement (C1). The capability story does not
  hold; the differentiation rests on the uniform FID gain + the Exp-2 robustness mechanism (workshop-
  tier), exactly as the skeptic flag warned.

## Operational note
Failure mode was home-quota (Errno 28) mid-run despite 2 pruners — the 5K-PNG-per-cell × 12-cell grid
+ live trains outran reclamation, and nfe1000 at 5.2 h/cell made the grid a multi-day job. Any C1
redo must (a) write samples to scratch/holylabs not home, (b) drop to ≤2 nfe points or ≤2.5K samples.
Not worth doing — verdict is already decisive.
