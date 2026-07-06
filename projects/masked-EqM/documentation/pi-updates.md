# PI Updates — masked-EqM

## 2026-07-06 draft (mechanism finding — hard-constrain ceiling invalid for EqM)

**Trigger**: bug/finding invalidating a planned eval design (mid-project, not a full result).

Context: building 3-arm masked-recovery eval (floor=gaussian-trained, ceiling=hard-constrained
oracle, treatment=mask-trained) per structured-start-state plan.

Finding: hard-constrain ceiling (reset visible pixels to ground truth every sampler step) does
NOT function as a meaningful oracle for EqM. Root cause: EqM's target scaling `c(gamma) -> 0`
as state nears data manifold (`transport.py:122-126`) — near true/visible pixels the model's own
learned dynamics already drift ~0 regardless of whether we force-constrain. So hard-constrain and
raw treatment converge (0.1403 vs 0.1434 masked-region MSE, within noise) — this is an architecture
property, not a bug (verified via RNG-seed fix; both runs use identical mask/noise, gap did not
open up).

Action taken: dropped hard-constrain as the positive control. Substituted VAE-encode/decode
roundtrip (skip corruption/model, measure only VAE's own reconstruction floor) as an
architecture-independent positive control — gives real headroom number (0.0149 masked MSE),
confirms metric is not saturated.

Revised eval design: floor (gaussian-trained treatment score) vs treatment (mask-trained /
mixture-trained score) vs oracle (VAE roundtrip, architecture-independent). No ceiling-via-hard-
constrain arm.

**Status of step2_mask_sanity gate**: still PENDING outcome metrics. gaussian_seed0_v5
(job 28842669) and mask_seed0_v5 (job 28842681) sanity training running on seas_gpu as of this
draft; masked-recovery eval to follow immediately on completion. mixture checkpoint already
evaluated (0.1403) but not yet interpretable without the isolated gaussian/mask numbers to compare
against.

**Ask of PI**: none blocking — proceeding per pre-registered plan. Flagging for visibility per
mechanism-finding trigger in AGENTS.md PI-update protocol.

## 2026-07-06 draft (step2_mask_sanity gate PASS — first real result)

**Trigger**: stage exit gate pass.

masked-recovery eval complete for all arms (n=256 held-out IN-1K val images, mask_prob=0.5,
RNG-matched seed across arms, gpu_test/seas_gpu, 1 epoch / 40000 steps each, EqM-B/2, single seed):

| arm | mean_masked_mse | normalized_gap |
|---|---|---|
| gaussian floor (baseline-trained) | 0.24237 | 0.0 (by definition) |
| mask treatment | 0.11313 | 0.568 |
| mixture treatment (1:1:1 gaussian/mask/fourier) | 0.14025 | 0.449 |
| VAE-roundtrip oracle (positive control) | 0.01488 | 1.0 (by definition) |

`normalized_gap = (floor_err - treatment_err) / (floor_err - oracle_err)`.

**Headline**: training EqM with Bernoulli-masked start-states (not just Gaussian noise) makes the
learned energy field ~2.1x better at recovering masked image regions than a gaussian-only-trained
model of the same architecture/scale/epoch count. Mask-trained closes 57% of the floor-to-oracle
gap; gaussian-only closes 0%. This is the first real outcome-metric evidence for the structured
start-state hypothesis (step 2 of CLAUDE.md build order).

**Caveat**: single seed, single scale (EqM-B/2, IN-1K, 1 epoch ~ sanity, not a paper-scale run),
single mask_prob (0.5, matches training). No FID / energy-ordering / generation-quality check yet
— this result is specific to the masked-recovery task, not generation quality broadly.

**Notable secondary finding**: the mixture arm (gaussian+mask+fourier, weight 1:1:1) captures LESS
of the gain (45%) than the isolated mask arm (57%), despite mask being one of its three components.
Consistent with dilution (each arm gets 1/3 of the gradient signal) rather than synergy. Was
launched ahead of isolated-arm results per user's explicit call (accepted attribution-confound
risk) — this is the first data point on that confound, not yet fully interpreted (need step3
Fourier-only isolated result to know if fourier or the dilution itself is responsible).

**Ask of PI**: is single-seed, single-epoch sanity sufficient signal to proceed to step3
(Fourier-only isolated arm) per the strict build order, or should mask arm be re-run with 2-3
seeds first to firm up the 2.1x number before committing more compute to the next build-order step?
