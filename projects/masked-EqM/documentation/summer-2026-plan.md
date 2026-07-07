# masked-EqM — Summer 2026 Plan

Written 2026-07-06, triggered by step2 (masking) showing signal at sanity scale, per CLAUDE.md
("write this once step 2 shows signal").

## North star

NeurIPS 2026 workshop (deadline 2026-08-29), stretch ICLR 2027 main (~2026-10-01).

## Where we are

- Step 1 (baseline reproduction): done, prior diff-EqM baseline carries architecture/code sanity.
- Step 2 (Bernoulli masking): **PASS at sanity scale** (EqM-B/2, IN-1K, 1 epoch/40k steps, seed 0).
  Masked-recovery eval: mask-trained normalized_gap=0.568 vs gaussian floor=0.0, oracle=1.0 —
  mask-trained model recovers masked regions ~2.1x better than gaussian-only baseline.
- Step 4 partial (3-arm mixture, launched early per explicit user call, arm D-equivalent):
  normalized_gap=0.449 — underperforms isolated mask.
- Step 3 (fourier, isolated) + step4 arm B (gaussian+mask 2-way): RESOLVED 2026-07-06/07.
  Fourier isolated normalized_gap = **-0.720 (negative, worse than gaussian baseline)** on the
  masked-recovery task. Arm B (gaussian+mask, no fourier) = 0.464. USER DECISION: fourier dropped
  from build path entirely; gaussian+mask kept alive as a generalization/regularizer hypothesis
  (not a masked-recovery contender — mask-only will trivially win that task).
- Generation-quality FID (2026-07-07): built `eval_fid.py` (new, sanity-scale 2000-sample
  unconditional FID), ran on gaussian/mask/arm-B checkpoints. Result: gaussian floor FID=172.567,
  gaussian+mask arm B FID=176.711 (+4.1), mask-only FID=239.512 (+66.9). CONFIRMS the
  generalization hypothesis — mask-only's larger masked-recovery gain (0.568 vs arm B's 0.464)
  comes at a much larger generation-quality cost. Genuine tradeoff, not a clean winner. See
  pi-updates.md 2026-07-07 draft; PI decision pending on which recipe becomes the paper claim
  (needs_user_input=true in pipeline.json).

## Phased plan

### Phase 1 — sanity signal (IN PROGRESS)
Exit gate: step2 mask PASS (done), step3 fourier result + arm B result (running).
Decision point once both land: is mixture underperformance dilution or fourier-specific? Either
answer feeds directly into what mixture recipe to scale.

### Phase 2 — multi-seed confirmation
Once phase 1's arm picture is settled (which corruption(s) actually help), re-run the winning
arm(s) at 2-3 seeds, same sanity scale, to rule out single-seed noise on the masked-recovery gap.
Exit gate: mean normalized_gap held across seeds (no formal p-value threshold set yet — TBD with
PI, analogous to diff-EqM's 3-seed Welch-t discipline).

### Phase 3 — outcome-metric breadth
Sanity scale but broaden the win beyond masked-recovery MSE: FID from pure-noise sampling (does
masking-trained model still generate well unconditionally?), energy ordering check (E(clean) <
E(corrupt) < E(noise)). This is the "no free lunch" check — masking gains on masked-recovery task
must not come at a generation-quality cost, or the paper story weakens substantially.
Exit gate: FID not meaningfully worse than gaussian baseline; energy ordering holds.

### Phase 4 — scale to paper-comparable size
Only after phase 2 + phase 3 both pass. Re-baseline at whatever scale is chosen (same CLAUDE.md
discipline as diff-EqM: re-baseline before comparing variants at a new scale). IN-1K-256 EqM-B/2
80-epoch full training (matches diff-EqM's own trusted baseline scale) is the natural target given
existing tooling.
Exit gate: gain holds at paper scale, ideally with the same multi-seed protocol as phase 2.

### Phase 5 — write-up
Workshop draft target: complete methods + phase 1-4 results by 2026-08-22, buffer before 08-29
deadline. ICLR fallback only if workshop timeline slips or reviewers want more scale/seeds.

## Open questions (track here, resolve via debate/PI update as they come up)

1. ~~Mixture recipe: does gaussian+mask alone (2-way) recover most of isolated mask's gain?~~
   RESOLVED 2026-07-06/07: arm B=0.464 gap, between 3-arm mixture (0.449) and isolated mask (0.568)
   — both dilution and fourier's negative pull contribute.
2. ~~Is fourier corruption a net-positive ingredient on its own?~~ RESOLVED 2026-07-06/07: NO —
   isolated fourier normalized_gap=-0.720, worse than gaussian baseline, on masked-recovery task
   specifically. See pi-updates.md for mechanism read and PI decision ask (drop / retest on a
   different task / one retune at different cutoff).
3. ~~Does the masked-recovery gain transfer to generation quality (FID) at all?~~ RESOLVED
   2026-07-07: NO for mask-only (large -66.9 FID cost), mostly YES for gaussian+mask (-4.1 FID
   cost). Real tradeoff between recipes, not a universal answer. See pi-updates.md 2026-07-07 draft
   for the 4-option PI ask on which recipe becomes the paper claim.
4. What mask_prob sweep, if any, is worth doing before locking phase 4's recipe?
5. Fourier corruption dropped from build path per user decision 2026-07-07 — not pursuing the
   deblurring-task-specific question raised earlier unless PI redirects.
6. NEW (2026-07-07): would a mask-heavy mixture (3:1 or 4:1 mask:gaussian) land on a better point of
   the recovery/generation-quality tradeoff curve than either mask-only or the current 1:1 arm B?
   Untested — cheap to run (existing mixture_sample code), pending PI's choice among the 4 options
   in pi-updates.md (this is explicitly option (d) there).

## Scope discipline

Same as CLAUDE.md: no jump to IN-1K-scale confirmation runs without passing the current stage's
gate. Proxy/sanity results are filters, not publishable claims on their own (same rule as
diff-EqM's proxy-scale discipline).
