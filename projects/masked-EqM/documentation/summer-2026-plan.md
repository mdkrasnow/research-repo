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
  normalized_gap=0.449 — underperforms isolated mask, cause (dilution vs fourier drag) unresolved.
- Step 3 (fourier) + step4 arm B (gaussian+mask 2-way) launched 2026-07-06 (jobs 28888836,
  28888839) specifically to resolve the mixture question.

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

1. Mixture recipe: does gaussian+mask alone (2-way) recover most of isolated mask's gain, or does
   any dilution hurt? (Answering now via job 28888839.)
2. Is fourier corruption a net-positive ingredient on its own, independent of mixture dilution?
   (Answering now via job 28888836.)
3. Does the masked-recovery gain transfer to generation quality (FID) at all, or is it purely a
   denoising-task-specific effect? (Phase 3.)
4. What mask_prob / fourier_cutoff sweep, if any, is worth doing before locking phase 4's recipe?

## Scope discipline

Same as CLAUDE.md: no jump to IN-1K-scale confirmation runs without passing the current stage's
gate. Proxy/sanity results are filters, not publishable claims on their own (same rule as
diff-EqM's proxy-scale discipline).
