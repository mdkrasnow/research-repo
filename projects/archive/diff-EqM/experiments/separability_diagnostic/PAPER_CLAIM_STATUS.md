# Paper Claim Status — EqM trajectory-metacognition

A claim ledger. Each row: the claim, its current support level, the evidence, and
exactly what would upgrade it. "Decision-grade" = can be written in a paper as
stated. Updated 2026-06-14.

| # | Claim | Status | Evidence | Upgrade condition |
|---|---|---|---|---|
| 1 | Endpoint energy predicts sample quality | **DEAD** | dot −⟨f,x⟩ 0.609, path-integral 0.605 de-conf; latent-NN baseline 0.627 beats both (3000-sample pool, de-confounded within grad-norm bins) | — (negative result; reportable as-is) |
| 2 | Descent **dynamics** predict failure | **SUPPORTED** | SHAPE probe within-norm 0.816, held-out 30% 0.818±0.012 (5 seeds), dim 30 ≪ 1200 train; early at k=100/249 = 0.814 | Replicate on a 2nd checkpoint / scale; already robust |
| 3 | Probe-guided inference improves FID | **DECISION-GRADE @50k 3-seed** | best-of-R=3 restart, **n=50000 × 3 seeds (jobs 22931315/323/328)**: probe<vanilla every seed (28.20→26.21, 27.78→25.95, 27.83→26.04), **mean Δ1.87±0.11 FID** | — (met; paper-ready) |
| 4 | Gains are **consistent** | **SUPPORTED** | 3 independent sampler seeds @50k, **95% CI ±0.12 excludes 0**, all-positive; matches 15k (Δ1.69) + smoke | replicate at another scale/model (optional) |
| 5 | Online equal-NFE adaptive sampler beats random | **SUPPORTED @15k real EqM** | mock probe 0.212 vs random 0.385; **15k real (job 22975626): probe-restart 28.51 < random-restart 29.76 @ equal NFE, Δ1.24, oracle 23.32, vanilla 29.55 sanity OK vs 31.41** | 50k online + multi-seed (optional tightening) |
| 6 | Mechanism unlocks **capabilities** (general) | **PARTIAL** | maze planning RUN: probe 0.928 vs random 0.794 valid-rate at equal compute (89% of oracle); +0.228 at hardest tier | image-domain rung (inpainting/repair) at equal NFE on EqM |
| 7 | Capability: maze planning | **SUPPORTED (toy)** | as above; controlled equal-compute, neg+pos controls | harder planner / real maze benchmark; note toy detection is trivial (AUC≈1.0) |
| 8 | Capability: inpainting / translation | **UNPROVEN** | designed only (`CAPABILITY_LADDER_RESULTS.md` B/C) | run the scoped EqM repair ladder |
| 9 | General "EBM energy = metacognition" principle | **UNPROVEN** | claim #1 (energy) is dead; the live signal is *dynamics*, not energy — so any general-EBM framing must be about descent dynamics, not energy scalars | transfer of the *dynamics* probe across ≥2 model families / tasks (EqM + maze is a start, not sufficient) |

## How to phrase it today (safe wording)

- ✅ "EqM's energy scalar does not separate good from garbage samples; the
  good/garbage signal instead lives in the **shape of the descent trajectory** and
  is recoverable by a small learned probe at ~0.82 de-confounded AUROC."
- ✅ "Acting on this probe via best-of-R restart gives a **consistent** FID gain:
  mean Δ1.87±0.11 FID across 3 seeds at 50k (95% CI excludes 0), inside oracle/random
  controls." (consistency now established — drop the single-seed caveat.)
- ✅ "An equal-NFE online adaptive sampler that restarts probe-flagged samples beats
  a random-restart control by Δ1.24 FID at 15k (vanilla sanity OK)."
- ✅ "In a controlled maze-planning analog, risk-guided branching beats random
  branching at equal compute (0.928 vs 0.794 valid-path-rate)."
- ❌ Do NOT yet write "unlocks inpainting/translation" — #8 unproven (designed only).
- ❌ Do NOT write "EqM energy enables metacognition" — energy is dead (#1). The
  correct framing is **descent-dynamics** metacognition.

## Status (2026-06-14): headline claims PASS at scale
Phase 1 (#3/#4 consistency) and Phase 2 (#5 online sampler) both **decision-grade**.
Remaining open: capability rungs B/C (inpainting/translation, #8) and the general
cross-family EBM principle (#9) — gated on Yilun's priority call (maze depth vs image
repair). The headline metacognition result is paper-ready.
