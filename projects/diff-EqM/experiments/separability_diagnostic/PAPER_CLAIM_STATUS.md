# Paper Claim Status — EqM trajectory-metacognition

A claim ledger. Each row: the claim, its current support level, the evidence, and
exactly what would upgrade it. "Decision-grade" = can be written in a paper as
stated. Updated 2026-06-14.

| # | Claim | Status | Evidence | Upgrade condition |
|---|---|---|---|---|
| 1 | Endpoint energy predicts sample quality | **DEAD** | dot −⟨f,x⟩ 0.609, path-integral 0.605 de-conf; latent-NN baseline 0.627 beats both (3000-sample pool, de-confounded within grad-norm bins) | — (negative result; reportable as-is) |
| 2 | Descent **dynamics** predict failure | **SUPPORTED** | SHAPE probe within-norm 0.816, held-out 30% 0.818±0.012 (5 seeds), dim 30 ≪ 1200 train; early at k=100/249 = 0.814 | Replicate on a 2nd checkpoint / scale; already robust |
| 3 | Probe-guided inference improves FID | **SUPPORTED @15k, single seed** | best-of-R=3 restart: vanilla 29.53→probe 27.84→oracle 17.75; beats random floor in band | **50k × ≥3 seeds, mean gain with CI excluding 0** (`consistency.sbatch`) |
| 4 | Gains are **consistent** | **NOT YET** | one sampler seed at 15k; direction only | same as #3 — the multi-seed 50k run |
| 5 | Online equal-NFE adaptive sampler beats random | **SUPPORTED @512 real EqM (relative)** | mock: probe 0.212 vs random 0.385 @ identical NFE; **real EqM 512 (job 22865794): probe-restart 97.34 < random-restart 99.21 @ equal NFE, Δ1.87, oracle 90.50** (512-FID inflated → relative only) | 15k/50k `online_adaptive.sbatch` with vanilla≈31.41 sanity restored |
| 6 | Mechanism unlocks **capabilities** (general) | **PARTIAL** | maze planning RUN: probe 0.928 vs random 0.794 valid-rate at equal compute (89% of oracle); +0.228 at hardest tier | image-domain rung (inpainting/repair) at equal NFE on EqM |
| 7 | Capability: maze planning | **SUPPORTED (toy)** | as above; controlled equal-compute, neg+pos controls | harder planner / real maze benchmark; note toy detection is trivial (AUC≈1.0) |
| 8 | Capability: inpainting / translation | **UNPROVEN** | designed only (`CAPABILITY_LADDER_RESULTS.md` B/C) | run the scoped EqM repair ladder |
| 9 | General "EBM energy = metacognition" principle | **UNPROVEN** | claim #1 (energy) is dead; the live signal is *dynamics*, not energy — so any general-EBM framing must be about descent dynamics, not energy scalars | transfer of the *dynamics* probe across ≥2 model families / tasks (EqM + maze is a start, not sufficient) |

## How to phrase it today (safe wording)

- ✅ "EqM's energy scalar does not separate good from garbage samples; the
  good/garbage signal instead lives in the **shape of the descent trajectory** and
  is recoverable by a small learned probe at ~0.82 de-confounded AUROC."
- ✅ "Acting on this probe via best-of-R restart improves FID at 15k (29.53→27.84)
  inside oracle/random controls" — **must add** "single sampler seed; multi-seed 50k
  pending" until #4 passes.
- ✅ "In a controlled maze-planning analog, risk-guided branching beats random
  branching at equal compute (0.928 vs 0.794 valid-path-rate)."
- ❌ Do NOT yet write "consistent FID improvement", "online metacognition sampler
  works", or "unlocks inpainting/translation" — #4, #5(scale), #8 unproven.
- ❌ Do NOT write "EqM energy enables metacognition" — energy is dead (#1). The
  correct framing is **descent-dynamics** metacognition.

## Single highest-value next run
The 50k × 3-seed consistency run (#3→#4). Everything downstream — the online
sampler claim, the capability story — is more citable once the headline FID gain
has a confidence interval.
