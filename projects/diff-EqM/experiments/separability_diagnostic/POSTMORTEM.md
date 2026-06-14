# Separability Diagnostic — Postmortem (vanilla EqM-B/2 80ep)

**Date:** 2026-06-13 · **Jobs:** 22505503 (OOM b128), 22507052 (full run b64),
22518284 (retune, labels) · **Run dir:** `runs/b2_vanilla/`

## Question
Does ANY cheap scalar, computable at the GD stopping point ("gradient goes
quiet"), reliably assign worse values to garbage outputs than to good ones —
**independent of the gradient norm**? This is the load-bearing assumption of the
proposed metacognition sampler (detect the low-norm-but-high-energy "spurious
minimum" quadrant). Cheap measurement, no training. KILL = success.

## Setup
- 3000 GD samples, vanilla EqM-B/2 (FID 31.41 ckpt), η=0.003, cfg=1.0, 250 steps.
- Independent labels = Inception (FID pool3) k-NN distance to 20k real ImageNet,
  crisp tails q_good=q_garb=0.25 → good=750/garbage=750. Label sanity s4=0.627,
  secondary resnet50 max-softmax agreement +0.176.
- 5 candidate scores at stopping point: s1=−⟨f,x⟩ (dot), s2=0.5‖f‖², s3=path
  integral Σ⟨f,dx⟩, s4=latent-NN dist (no f; baseline), s5=post-step ‖f‖.
- s1,s3 de-confoundable; s2,s5 norm-coupled by construction; s4 label control.
- Matched-norm control = AUROC within gradient-norm quantile bins (strips out
  any separation that is just norm-in-disguise).

## Result (de-confounded, within-norm-bin AUROC; robust across fixed + τ∈{5,10,20})
| score | within-norm | note |
|---|---|---|
| s1 dot | 0.609 | de-confoundable, best independent |
| s3 path-integral | 0.605 | de-confoundable |
| **s4 latent-NN (no f)** | **0.627** | dumb baseline — BEATS the energy scores |
| s2 / s5 | ~0.53 | norm-coupled, ≈chance after de-confound |

`best_independent_auroc = 0.609` → **WEAK** (0.60–0.80 band). Action bar 0.80.

## Verdict: metacognition sampler NOT justified
1. A norm-independent energy signal **exists but is faint** (0.61 vs 0.80 bar) —
   barely above chance after de-confounding.
2. **Decisive:** the no-`f` latent-NN baseline (0.627) **beats** both energy
   scores (0.609/0.605). EqM's own energy carries *less* good/garbage information
   than a trivial geometric probe. Using energy as the metacognitive signal buys
   nothing over a cheaper non-energy baseline.
3. Consistent across all 4 stopping-rule regimes — not a single-regime artifact.

This matches the prior skeptic read: EqM's energy is a **density** skeleton
(validated for OOD/density), not a usable causal/quality axis. The 2×2's
load-bearing cell (low-norm + high-energy spurious minimum) is **not reliably
detectable** from local scalars on the vanilla field.

## Retune (the one allowed per CLAUDE.md)
Initial labels (q 0.40/0.30, 10k refs, s4=0.620) → best_independent 0.582 → KILL.
Sharper labels (q 0.25/0.25, 20k refs, s4=0.627) → 0.609 → WEAK. The label-noise
caveat was partly real (KILL→WEAK on cleaner labels), but the ceiling held: even
with crisp labels the energy tops out ~0.61. No further retunes.

## Cost / payoff
~1 GPU-hour total (28 min sampling cached + ~12 min relabel). Killed a
weeks-long build (metacognition sampler / conservativity fix) before writing it.

## Caveats / scope
- Vanilla checkpoint only. The explicit-energy EqM-E variant was NOT tested — but
  EqM-E is the paper's fragile mode (FID 57→73), so it is a weaker, not stronger,
  bet for a clean energy axis.
- Single seed, B/2 scale, GD sampler. The negative is specific to "local scalar
  at the GD stopping point on the vanilla field."
- Label quality (s4=0.627) bounds sensitivity; a fundamentally better quality
  oracle could lift the ceiling, but the dumb-baseline-beats-energy result is the
  robust signal and does not depend on the absolute label quality.

## Decision
Do not build the metacognition sampler on EqM energy. Direction archived. If
revisited, the bar is explicit: a scalar must reach ≥0.80 de-confounded AND beat
the latent-NN baseline — neither holds for the vanilla field.
