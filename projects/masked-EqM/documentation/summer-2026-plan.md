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
  comes at a much larger generation-quality cost. Genuine tradeoff, not a clean winner.
- Gaussian:mask Pareto sweep + cross-corruption generalization (2026-07-08): 1:2/1:3/1:4 all
  Pareto-dominate arm B on trained-corruption gap+FID, but only 1:1 and 1:4 show genuine
  compositional generalization (beat both pure arms on block_mask) -- 1:2/1:3 do not, despite
  better raw numbers. 1:4 is the least-compromised single choice (best gap in sweep + clears
  generalization bar + perfect field-ordering). See pi-updates.md 2026-07-08 draft for full tables;
  PI decision pending on which recipe/framing the paper leads with (needs_user_input=true in
  pipeline.json).
- Phase 2 multi-seed confirmation (2026-07-08): PASS for both arm B and 1:4, 3 seeds each.
  Arm B: recovery gap 0.467+-0.0055, FID 175.74+-1.35. 1:4: recovery gap 0.518+-0.0015, FID
  178.03+-0.61. Gap ranges do not overlap across seeds -- tradeoff is a real, seed-stable effect,
  not single-seed noise. Same shape as seed0: 1:4 wins recovery, arm B wins FID. PI call still
  pending on flagship recipe (needs_user_input=true in pipeline.json).
- Generalization grid, 3-seed (2026-07-09): RETRACTION of 2026-07-08 block_mask claim. All 6
  recipes now have seed1/seed2 on the full 9-corruption suite. block_mask: gaussian 0.7264+-0.0399,
  mask 0.6783+-0.0222, 1:1 0.6708+-0.0389, 1:2 0.6683+-0.0300, 1:3 0.6624+-0.0095, 1:4
  0.6762+-0.0191 -- heavy overlap, no ratio distinguishable, "1:1/1:4 uniquely generalize" was
  single-seed noise. noisy_masked DOES hold: mask-only 0.1586+-0.0012 stays clearly best, no
  mixture close, tight non-overlapping spreads. See pi-updates.md 2026-07-09 for full tables and
  recommendation (drop compositional-generalization claim, lead with Pareto tradeoff instead).

## Phased plan

### Phase 1 — sanity signal (IN PROGRESS)
Exit gate: step2 mask PASS (done), step3 fourier result + arm B result (running).
Decision point once both land: is mixture underperformance dilution or fourier-specific? Either
answer feeds directly into what mixture recipe to scale.

### Phase 2 — multi-seed confirmation (PASS 2026-07-08)
Re-ran arm B and 1:4 at 3 seeds each, same sanity scale. Exit gate (mean normalized_gap held
across seeds): PASS — arm B 0.467±0.0055, 1:4 0.518±0.0015, non-overlapping ranges across all 3
seeds each. Recovery gap and FID tradeoff both replicate the seed0 shape.

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
6. ~~Would a mask-heavy mixture (1:2/1:3/1:4 gaussian:mask) land on a better point of the
   recovery/FID tradeoff curve than 1:1 arm B?~~ RESOLVED 2026-07-08: YES on raw Pareto numbers —
   1:2 (gap=0.502/FID=176.19), 1:3 (0.513/178.0), 1:4 (0.518/178.5) all Pareto-dominate arm B
   (0.464/176.71). Trend hadn't reversed by 1:4; ceiling not yet found, sweep stopped at 1:4 as
   pre-approved. See pi-updates.md 2026-07-08 draft, Table 1.
7. ~~Does gaussian+mask training learn a genuinely general repair field, or memorize two exact
   corruptions?~~ RESOLVED 2026-07-08 single-seed, RETRACTED 2026-07-09 at 3 seeds. Single-seed
   claimed only 1:1/1:4 beat both pure arms on block_mask; 3-seed replication shows heavy
   overlapping spread (sd up to 0.04) and no ratio is distinguishable — that was noise, not a real
   compositional-generalization effect. noisy_masked result DOES hold at 3 seeds: mask-only stays
   clearly, tightly best, no mixture close. Field-ordering (8/9 tie, arm B & 1:4) was only ever
   checked at seed0 — not yet replicated, flagged as an open gap. Recommendation now: drop the
   compositional-generalization claim, lead with the Pareto tradeoff (arm B vs 1:4, confirmed
   3-seed) instead. See pi-updates.md 2026-07-09 for full tables.
8. NEW (2026-07-08): would 1:5/1:6+ continue the Pareto-dominance trend, or is there a ceiling
   between 1:4 and mask-only where it collapses toward mask-only's FID disaster? Untested — user
   did not pre-approve past 1:4, pending redirect.
9. NEW (2026-07-08): does noisy_masked have a mixture ratio that beats mask-only (0.160)? None of
   1:1-1:4 do yet (best is 1:2 at 0.163). Untested whether a different ratio or a dedicated
   noisy+masked training arm (not yet built) would close this specific gap.

## Scope discipline

Same as CLAUDE.md: no jump to IN-1K-scale confirmation runs without passing the current stage's
gate. Proxy/sanity results are filters, not publishable claims on their own (same rule as
diff-EqM's proxy-scale discipline).
