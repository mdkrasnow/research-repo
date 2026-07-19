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
- Phase 2 confirmation COMPLETE, 1:4 PROMOTED FLAGSHIP (2026-07-09): filled remaining gaps
  (FID+recovery p=0.5 for gaussian/mask seed1/2; field-ordering for gaussian/mask/1:1/1:4 x 3
  seeds each -- was seed0-only before). Pre-registered gate: 1:4 FID +4.54 over gaussian (within
  +10/+15 budget) AND beats both pure arms on block_mask in 2/3 seeds -> PROMOTE 1:4. New clean
  result: field-ordering is a stable qualitative split -- gaussian/mask both FAIL (E(corrupt) >
  E(noise), backwards) at all 3 seeds; 1:1/1:4 both PASS at all 3 seeds. See pi-updates.md
  2026-07-09 (later) for full table. No 1:5/1:6 sweep or new eval types launched, per user gate.
- Blur extension COMPLETE, GATE FAILED, no promotion (2026-07-14): built second structured
  start-state family (Gaussian blur, depthwise conv on VAE latent). Severity calibrated via
  pixel-space LPIPS matching to p=0.5 mask task (raw MSE matching impossible -- see
  blur-calibration.md). Trained blur-only, gaussian:blur 1:1, gaussian:blur 1:4, 3 seeds each.
  Pre-registered gate (FID within +10 of gaussian AND blur-MSE within 15% of blur-only) FAILED
  for both mixed arms: 1:1 FID +15.2 over gaussian (fails), 1:4 FID +34.6 (fails); MSE passes for
  both (+14.75%/+10.4%) but FID failure dominates. blur-only itself is far more destructive to
  generation than mask-only ever was (FID ~388 vs gaussian ~173, vs masking's much milder cost).
  Narrow positive: mixture arms generalize better than blur-only at held-out severities (2.0,
  4.0 sigma), echoing masking's qualitative flavor but far too weak to clear the gate. Direct
  answer: masking principle does NOT replicate for blur at this scale/severity. See pi-updates.md
  2026-07-14 for full 3-seed tables, severity grid, zero-shot cross-family recovery.
- Fourier + downsample extension COMPLETE, sanity-scale gate PASS but cross-generalization
  hypothesis FAILED (2026-07-15): built two more structured start-state families (Fourier
  radial low-pass and bilinear downsample-then-upsample on VAE latent, LPIPS-calibrated severity:
  cutoff=0.4181, factor=2.4615). Trained only/1:1/1:4 arms per family, 3 seeds each (1:4
  deprioritized per user, not analyzed in depth). Baseline correction mid-session: sanity-scale
  gaussian FID floor is ~173.5 (3-seed: 172.57/172.83/175.08), NOT diff-EqM's archived 80-epoch
  31.41 -- different project/scale, do not reuse across projects.
  Sanity-scale result: both 1:1 mixtures PASS (fourier 1:1 FID ~182, +8; downsample 1:1 FID ~187,
  +13/+14, passed under user's relaxed cutoff) -- generation quality near baseline, recovery MSE
  near the single-family supervised ceiling for their own trained corruption.
  Cross-generalization matrix (the stronger claim -- does mixed training expand the basin of
  attraction to *unseen* corruptions, not just retain the two trained ones): ran full off-diagonal
  matrix, every 1:1 mixture zero-shot evaluated on the other unseen corruptions, 3 seeds. Result:
  HYPOTHESIS NOT SUPPORTED. Mask<->Fourier are not a shared "partial-observation" family --
  gaussian+fourier 1:1 transfers *worse* to mask (0.315) than plain gaussian-only (0.242). No
  mixture beats both single-objective parents on any off-diagonal cell. One narrow positive:
  gaussian+fourier 1:1 -> downsample is the best zero-shot downsample result (0.086), most likely
  because fourier low-pass and downsample are both literally low-pass operations in frequency
  space (signal-processing overlap), not evidence of general energy-landscape robustness.
  Recommendation: restrict paper claim to "diverse mixture preserves multitask capability without
  degrading generation" (a real, modest result across masking/fourier/downsample), drop the
  zero-shot-generalization claim. See pi-updates.md 2026-07-15 for full matrix + tables.
  needs_user_input=true in pipeline.json -- PI read requested on whether multitask-only framing
  is still publication-worthy before further compute on this direction.
- LPIPS correction to cross-generalization matrix (2026-07-15, same day): MSE-only matrix above
  UNDERSTATED the result -- added LPIPS to eval_masked_recovery.py (previously MSE-only) and
  re-ran full matrix on perceptual metric. Found 2 genuine "beats both single-objective parents"
  cells: G+M 1:1 -> fourier LPIPS 0.579 (vs gaussian-only 0.611, mask-only 0.647) -- strongest,
  confirmed real, not an MSE artifact; and G+D 1:1 -> mask LPIPS 0.584 (vs gaussian-only 0.594,
  downsample-only 0.774) -- new, invisible under MSE. G+M 1:4 (checked, no retrain needed) also
  passes on fourier but weaker than 1:1 (0.598) -- more mask weight dilutes rather than
  strengthens the effect. 4 of 6 off-diagonal cells still fail (G+F->mask, G+D->fourier,
  G+F/G+M->downsample no signal). Revised claim: "mixed training produces emergent perceptual
  generalization to certain (not all) unseen corruptions" -- narrower than universal robustness,
  stronger than pure negative. See pi-updates.md 2026-07-15 (later) for full table.
  needs_user_input=true still set -- PI call pending on whether 2-cell result is pub-worthy as-is
  vs pursuing cross-corruption-consistency loss / 3-source leave-one-out next.
- Preregistered 3-seed replication of G+M->fourier (2026-07-15, final): built a fixed 1024-image
  manifest evaluator with per-image deterministic inputs + hierarchical bootstrap analysis, retrained
  the missing G+M seed0 (2 infra incidents en route: stale cached sbatch script from skipping
  remote_submit.sh's rsync; holylabs quota hard cap at first checkpoint write, same recurring
  failure -- both resolved, see pi-updates.md). Preregistered gate: 4/5 criteria pass. G+M beats
  mask-only decisively (delta_M 95% CI [0.009,0.020], Holm p<0.0001, 3/3 seeds, 4/5 cutoffs).
  G+M beats gaussian-only directionally (2/3 seeds, 4/5 cutoffs) but the hierarchical CI for
  delta_G includes zero at n=3 seeds -- NOT yet statistically decisive. FID 176.03 (+2.5, well
  within gate). Stage 2 (seeds 3-5) not auto-triggered since strict gate is 4/5, not 5/5.
  needs_user_input=true -- PI call: extend to n=5 or land claim on mask-only comparison alone.
  See pi-updates.md 2026-07-15 (final) for full tables and qualitative grids.
- n=5 checkpoint (2026-07-16): delta_G drifted toward null (mean 0.0029->0.00047, Holm p
  0.176->0.817) as seeds 3,4 completed. Per predeclared stopping rule, neither stop condition
  triggered (CI still includes zero, mean hasn't reversed sign) -- continue to n=8, no inference
  yet. delta_M stayed robust (Holm p<0.0001 throughout). needs_user_input=true for PI call on n=8
  vs stopping here.
- Harder-cutoff experiment (2026-07-16, same day): re-analyzed the SAME n=5 checkpoints (no
  retrain) at the already-collected cutoffs 0.20/0.30 (vs primary 0.4181). **Cutoff 0.20 PASSES
  the full 3-part gate**: G+M mean LPIPS lower than both parents, both hierarchical CIs exclude
  zero after Holm (delta_G p=0.0008, delta_M p<0.0001), 4/5 seeds beat both parents. Cutoff 0.30
  fails on CI significance for delta_G (p=0.318) despite 4/5 seeds and correct direction.
  Severity-response is monotone and clean: delta_G shrinks from +0.0043 (0.20) -> +0.0019 (0.30)
  -> +0.0005 (0.4181, null). CONCLUSION: mixed Gaussian+mask training DOES broaden recovery to
  unseen Fourier corruption, but only when the corruption is severe (cutoff <=0.20); at milder
  unseen corruption the effect is not reliable. Real, narrower, defensible result -- not the
  universal claim, not a pure null either. MSE flagged again as misleading (mask-only wins MSE at
  every cutoff despite worst LPIPS -- blurry-reconstruction artifact). Qualitative grids built at
  both cutoffs (4 selection types x 2 cutoffs, no cherry-picking). See pi-updates.md 2026-07-16
  harder-cutoff section for full tables.

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

## Fourier zero-shot severity sweep (2026-07-16)

Reused existing matched seed0-4 checkpoints (gaussian/mask/gm) + fourier specialist (seed0), no
retrain, evaluated unseen-Fourier-recovery LPIPS at 6 cutoffs (0.05,0.10,0.15,0.20,0.30,0.4181) on
the same frozen 1024-image manifest. **Result is a narrow severity sweet spot, not a monotone
trend**: delta_G (gaussian-vs-GM) is statistically reliable (Holm-corrected CI excludes zero,
>=4/5 seeds beat both parents) only in the **0.10-0.20 cutoff band**, peaking at 0.10-0.15
(5/5 seeds). It is non-significant at both the milder end (0.30, 0.4181 — matches prior finding)
and, newly, at the most severe cutoff tested (0.05). delta_M (mask-vs-GM) stays significant at
every cutoff 0.05-0.4181. Caveat: the fourier-specialist reference model itself degrades badly at
severe cutoffs (underperforms gaussian-only at 0.05-0.30), so the normalized specialist-gap-closure
metric is not trustworthy outside the specialist's own ~0.25-0.4181 training regime — flagged, not
used for headline claims. Full curve, plots, 12 qualitative grids + 3 blinded A/B sheets, and honest
writeup in pi-updates.md 2026-07-16 (later) draft. No new training/ratio/corruption-family work
launched (explicit scope limit this round). n=8 stopping-rule decision for cutoff 0.4181 remains
open and separate.

## Recovery convergence-curve diagnostic (2026-07-19)

Checked whether the strongest severity-sweep result (delta_G peak at cutoff 0.10, +0.0053 LPIPS)
is a genuine landscape/endpoint difference or an artifact of the arbitrary 250-step GD recovery
horizon, before pursuing retraining-based levers to grow the effect. No retrain -- same seed0-4
gaussian/mask/gm checkpoints, cutoff fixed at 0.10, single trajectory per image recorded at steps
{0,25,50,100,250,500,1000} (new `gd_recover_multi` in `eval_fourier_recovery.py`, bit-identical to
separate single-step calls, unit-tested before submission). **Result: the gap is real and persists
to 1000 steps (4x the original horizon) — never reverses or vanishes — but it peaks at 250
(+0.0053) and shrinks ~40% by 1000 (+0.0032), and gaussian partially catches up given more compute**
(matches GM's old 250-step target by step 500). So the 250-step headline number overstates the
converged/steady-state gap (closer to +0.003-0.004). MSE for gaussian/GM rises sharply past step
250 (overshoot/drift) even as LPIPS keeps marginally improving -- 250-500 steps, not 1000, is the
best common stopping point if pixel fidelity matters at all. **Critical caveat**: qualitative grids
show NONE of the three models achieve recognizable object-identity recovery at cutoff 0.10 at any
step tested (including GM's best-case wins) -- outputs are generic blurry color blobs across the
board, so this whole comparison is "which model fails less," not "which model succeeds." Extending
the recovery horizon is therefore **not** a lever that grows the effect (it shrinks it and adds
MSE-side cost) -- retraining-based levers (ratio sweep, mask-prob tune, scale) remain the more
promising next moves for enlarging the effect size, pending a variant proposal per standing
discipline. Full table/plots/16 grids/writeup in pi-updates.md 2026-07-19 and
`documentation/convergence_2026-07-19/`.

## Scope discipline

Same as CLAUDE.md: no jump to IN-1K-scale confirmation runs without passing the current stage's
gate. Proxy/sanity results are filters, not publishable claims on their own (same rule as
diff-EqM's proxy-scale discipline).
