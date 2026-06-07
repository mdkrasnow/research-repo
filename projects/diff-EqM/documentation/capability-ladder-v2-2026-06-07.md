# Capability ladder v2 — what behavior changed? (2026-06-07)

Re-opens the ANM/v10 capability question after the v1 ladder NULL. v1 only
falsified **naive zero-shot clamped restoration** (denoise/inpaint/transfer).
v2 asks: does the −3.83 FID gain correspond to ANY real behavioral change?

Frozen checkpoints. Sampler / seeds / labels / VAE / EMA / eval held fixed.
Arms: vanilla seed0 vs ANM v10 λ0.3 seed0 (matches Exp 3). Pre-registered gates,
nulls reported honestly, no long random-control train unless a checkpoint signal
appears.

Gate to even proceed past A: a checkpoint-level signal beyond aggregate FID.
**PASSED at Rung A** (class-adherence + hard-class localization, below).

---

## Rung A — gain localization  [DONE, from Exp 3, single seed λ0.3]
Source: `results/exp3_metrics_out/` (49996 matched ids, parity-controlled).

| axis | Δ (v10 − vanilla) | reads as |
|---|---|---|
| FID | −4.38 (CIs disjoint) | gain real |
| KID | −0.0057 | confirms |
| precision | +0.023 | fidelity up |
| recall | +0.0008 | **diversity FLAT — no tax** |
| density | +0.044 | local density up |
| coverage | +0.072 | mode coverage up |
| conditional top-1 | +0.050 | **more on-class** |
| classifier TV→requested | −0.019 | better class balance |
| frac classes improved (feat-dist) | 0.913 | broad |

**Verdict: the FID gain is QUALITY + CLASS-ADHERENCE + COVERAGE, not mode-dropping.**
Better conditional top-1 = a behavioral change (images land in requested class more
often), independent of FID.

## Rung B — hard-class improvement  [DONE, from Exp 3, single seed λ0.3]
Classes ranked by vanilla difficulty (per-class feature distance; per-class FID is
50-sample-noisy and not used for ranking).

| quartile | Δfeat-dist | Δcond-top1 |
|---|---|---|
| EASY (lowest van feat-dist) | −0.417 | +0.033 |
| HARD (highest van feat-dist) | −0.557 | +0.062 |
| **HARD/EASY ratio** | **1.34×** | **~1.9×** |

Also: bottom-quartile weak-class FID −5.61 vs mean −4.38 (Exp 3). 91.3% classes
improve feat-dist; 72.9% improve top-1.

**Verdict: v10 preferentially fixes the model's weak regions** — the hard-example
mining mechanism prediction, confirmed at checkpoint level.

## Rung C — failed-generation rescue  [building]
## Rung D — sampler robustness (NFE × step_mult)  [Exp 1 infra, submitting]
## Rung E — trajectory swap (vanilla/v10 early↔late)  [building]
## Rung F — class-guided counterfactual edit  [building]

(Inpainting/outpainting/translation revisited ONLY after A–F.)

## Caveats
- Rung A/B = single seed λ0.3. 3-seed (27.58±0.36 exists) would tighten; per-class
  signal is broad (91%/73%) so unlikely to vanish.
- Per-class FID noisy at 50 samples/class → ranking uses feature distance.
