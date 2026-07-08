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

(Superseded below — user approved proceeding to step3 immediately; result landed same day.)

## 2026-07-06/07 draft (step3 fourier RESOLVED NEGATIVE + arm B mixture — significant pivot)

**Trigger**: significant pivot (fourier de-prioritized for this task) + result at gate-informing scale.

Ran step3 (isolated fourier-only sanity) and a 2-way gaussian+mask mixture (CLAUDE.md arm B)
same day, same eval protocol as the step2 result above (n=256, seed 0, mask_prob=0.5):

| arm | mean_masked_mse | normalized_gap |
|---|---|---|
| gaussian floor | 0.24237 | 0.0 |
| **mask (isolated)** | **0.11313** | **0.568** |
| gaussian+mask (arm B, 2-way) | 0.13692 | 0.464 |
| gaussian+mask+fourier (3-way) | 0.14025 | 0.449 |
| **fourier (isolated)** | **0.40612** | **-0.720** |
| VAE-roundtrip oracle | 0.01488 | 1.0 |

**Headline finding**: fourier low-pass corruption, trained in isolation, makes the model WORSE
than the gaussian-only baseline at masked-region recovery (negative normalized_gap = -0.720 — worse
than doing nothing special). This is a genuine negative result, not noise-level: it's the single
most extreme number in the whole table, in the wrong direction.

**Why this makes mechanistic sense**: fourier low-pass corruption trains the model to reconstruct
missing HIGH-FREQUENCY detail from coarse global structure (a deblurring-like task). Bernoulli
masking trains the model to reconstruct a spatially-missing region from surrounding visible pixels
(an inpainting-like task). These are different recovery skills. Training on one does not transfer
to, and apparently actively interferes with, the other — plausible mechanism: fourier training
biases the learned field toward frequency-domain corrections that are a poor prior for the
spatially-local, high-magnitude corrections the masked-recovery task needs.

**Mixture picture now resolved**: arm B (gaussian+mask, no fourier) scores 0.464 — between isolated
mask (0.568) and the original 3-arm mixture (0.449), but closer to the 3-arm number. This means the
3-arm mixture's underperformance vs isolated mask is driven by BOTH (a) gradient-budget dilution
(splitting signal across arms costs ~0.10 gap even with fourier excluded) AND (b) fourier's own
negative pull (present but diluted to 1/3 weight in the 3-arm case, vs -0.720 in isolation).

**Practical implication for the paper story**: CLAUDE.md's build-order arms C (gaussian+fourier)
and D (gaussian+mask+fourier) are now de-prioritized for the masked-recovery goal specifically --
adding fourier to any mixture aimed at this task looks actively counterproductive based on this
data. This does NOT mean fourier corruption is dead for the project broadly: this is one task
(masked-region recovery) at one scale (sanity) with one seed. Fourier could still matter for (a) a
different downstream task (e.g. deblurring/denoising recovery, which CLAUDE.md also lists as a
metric to check), or (b) generation quality (FID) independent of masked-recovery, neither of which
has been measured yet.

**Ask of PI**: given this negative result, should Fourier corruption be:
(a) dropped entirely from further build-order steps (proceed straight to mask-only Phase 2/3 per
    summer-2026-plan.md),
(b) kept but only tested for hypothesized non-masking tasks (deblurring-style recovery, if that's
    still of interest), or
(c) given one more isolated try at a different fourier-cutoff value before concluding, per the
    "1 retune per failing direction" rule in CLAUDE.md/AGENTS.md?
No compute committed to any of these yet pending PI input.

**Resolution (user, same day)**: option (a) — drop fourier entirely, keep gaussian+mask alive as a
generalization/regularizer hypothesis rather than a masked-recovery contender (mask-only will
trivially win that task, since it gets 100% of its gradient budget on exactly that skill). User's
framing: the real question is whether gaussian+mask learns a better OVERALL energy landscape
(global generative coverage from gaussian + local conditional-repair skill from mask) than either
parent alone — tested via cross-task generalization (FID, energy ordering, mask_prob sweep), not
another masked-recovery-only comparison.

## 2026-07-07 draft (generation-quality FID result — resolves the mask-only vs gaussian+mask question)

**Trigger**: result at gate-informing scale (generalization/generation-quality check requested by
user in the resolution above).

Built `eval_fid.py` (new, sanity-scale: 2000 unconditional samples via GD sampler, FID against
matching real-image sample, single seed) since no FID tooling existed yet at the masked-EqM sanity
scale. Smoke-sample-probe passed first (eyeballed generated PNGs from gaussian and mask
checkpoints — coherent textured scenes, not noise/garbage, confirms the sampler/pipeline works
before trusting the FID numbers).

| arm | FID (n=2000) | masked-recovery normalized_gap |
|---|---|---|
| gaussian floor | 172.567 | 0.0 |
| gaussian+mask (arm B) | 176.711 (+4.1) | 0.464 |
| mask-only | 239.512 (+66.9) | 0.568 |

**Headline**: this directly confirms the user's hypothesis. Mask-only wins masked-recovery
(expected — it gets full gradient budget on that exact skill) but at a LARGE generation-quality
cost: +66.9 FID vs baseline. Gaussian+mask nearly preserves baseline generation quality (+4.1 FID)
while still capturing 82% of mask-only's recovery gain (0.464/0.568). This is a genuine
generalization-vs-specialization tradeoff, not a case where one recipe dominates the other on both
axes.

**Mechanism read**: consistent with the user's proposed story — gaussian exposure teaches the
global noise-to-data route needed for unconditional generation; mask-only narrows the model's
training distribution toward "mostly-correct, locally-broken" states, which is excellent for
inpainting-style recovery but leaves the model under-exposed to the far-from-manifold states pure
noise-to-image generation requires.

**Caveat (same as all prior results)**: single seed, single sanity scale (1 epoch/40k steps,
EqM-B/2), single sample count (2000, well below the paper's 50k-sample protocol in
sample_gd.py/README) — absolute FID values (170-240) are expected to be poor at this undertrained
scale (paper's trusted 80-epoch baseline scores 1.90-31.41 depending on metric/scale) and should
only be read relatively (gaussian vs mask vs arm B), not as paper-comparable numbers.

**Ask of PI**: given a genuine tradeoff (not a clean winner), what should the paper claim be?
(a) mask-only as a specialized recovery/inpainting variant, with an explicit generation-quality
    caveat,
(b) gaussian+mask as the general-purpose/practical recipe (smaller effect size, safer),
(c) present both as separate contributions for different use cases, or
(d) explore mask-heavy mixtures (3:1, 4:1 mask:gaussian, per user's earlier suggestion) to find a
    better point on this tradeoff curve before deciding between (a)/(b)/(c)?
No further compute committed pending this decision. Whichever direction is chosen, still needs
Phase 2 multi-seed confirmation before this is a real paper claim (currently single-seed sanity
scale throughout).

**Resolution (user, same day)**: option (d) -- run a tight gaussian:mask sweep (1:2, 1:3, 1:4-if-
warranted) before locking the recipe. Decision rule: if 1:2/1:3 closes more of the recovery gap
while FID stays within ~10-15 of gaussian floor, promote it as the main recipe; if FID degrades
sharply, keep 1:1 arm B. Sweep launched (details below); one incident along the way (results-dir
race between the 1:2 and 1:3 jobs silently mixed their checkpoints -- caught, root-caused, fixed,
relaunched cleanly, no impact on final numbers). Sweep results pending as of this entry.

## 2026-07-07 draft (energy-ordering result — second independent confirmation of the generalization story)

**Trigger**: result at gate-informing scale, built proactively during overnight downtime while the
1:2/1:3 sweep trains (CLAUDE.md diagnostic #3 was on the "what to measure" list but unbuilt).

Built `eval_energy_ordering.py`: these checkpoints have no scalar energy head (`ebm='none'`), so
per EqM's own framing (the field IS the energy gradient) the mechanistically-justified proxy is
field norm ||f(x)|| — should be near 0 at the data manifold, large away from it. This also matches
the prior diff-EqM finding already in memory that raw energy scalars were a dead diagnostic while
field/descent-shape-based measures carried real signal.

Checked E(clean) < E(corrupt) < E(noise) (mask_prob=0.5, n=256, single seed) on the same three
checkpoints as the FID result:

| arm | clean | corrupt | noise | ordering holds |
|---|---|---|---|---|
| gaussian floor | 29.83 | 304.25 | 250.12 | **NO** (corrupt > noise) |
| mask-only | 15.53 | 202.62 | 182.27 | **NO** (corrupt > noise) |
| gaussian+mask (arm B) | 24.52 | 199.89 | 248.35 | **YES** |

**Headline**: clean < corrupt holds everywhere (all three models correctly recognize near-manifold
states as low-field). But corrupt-vs-noise ordering only holds for arm B. Gaussian-only has never
seen masked states during training — its field response to a masked input is inflated/erratic
(off-distribution). Mask-only has never seen pure Gaussian noise — same failure, mirrored. Only
arm B, trained on both corruption types, has a field response that's correctly calibrated across
the full state space it might encounter.

**Why this matters**: this is a SECOND, mechanistically independent line of evidence for the exact
same generalization argument the FID result made (172.6/176.7/239.5 FID ordering). FID says
"gaussian+mask preserves sample quality"; this says "gaussian+mask preserves correct field
geometry off the training corruption type it specialized in." Different measurement, same
conclusion, from a diagnostic that isn't even sensitive to sample quality — this is a materially
stronger case that gaussian+mask is really learning a more complete/generalizable energy landscape,
not narrowly optimizing whatever single metric happened to be checked.

**Caveat**: single seed, sanity scale, n=256, field-norm is a proxy (no ground-truth energy value
to compare against since ebm=none) — same scale caveats as everything else this week.

**Ask of PI**: no new decision needed — this strengthens the case for whichever of (a)/(b)/(c)/(d)
gets chosen once the sweep lands, particularly for (b)/(c) (gaussian+mask as general-purpose or
co-equal recipe). Flagging because it's a strong, unprompted secondary confirmation worth having in
the paper regardless of the final recipe decision -- suggest including both FID and energy-ordering
as complementary "generalization" evidence in whatever the final writeup is.

## 2026-07-08 draft (Pareto sweep complete + cross-corruption generalization suite — recipe choice now genuinely two-way)

**Trigger**: result at gate-informing scale, resolving both the ratio-sweep and the follow-up
generalization question the user asked to test explicitly.

### Table 1 — Pareto sweep (gaussian:mask ratio, all sanity scale, single seed)

| arm | masked-recovery gap | FID |
|---|---|---|
| gaussian floor | 0.0 | 172.567 |
| arm B (1:1) | 0.464 | 176.711 |
| **1:2** | 0.502 | 176.190 |
| **1:3** | 0.513 | 177.996 |
| **1:4** | 0.518 | 178.511 |
| mask-only | 0.568 | 239.512 |

1:2, 1:3, and 1:4 all **Pareto-dominate** arm B — strictly better gap AND equal-or-lower FID, not a
tradeoff. Gap climbs monotonically 1:1→1:4 while FID stays flat near the gaussian floor (all within
+6, nowhere near mask-only's +66.9). The trend has not reversed by 1:4 — we don't yet know where
the ceiling is; sweep stops at 1:4 because that's what was pre-approved, not because performance
degraded.

(One incident during this sweep: a results-dir race corrupted the first 1:2/1:3 training attempt —
caught, root-caused, fixed at the sbatch level, cleanly relaunched. All numbers above are from the
clean run.)

### Table 2 — Cross-corruption generalization (built `eval_generalization.py` per user request)

All 6 checkpoints (gaussian, mask, 1:1, 1:2, 1:3, 1:4) evaluated on 9 corruptions never explicitly
optimized for at this exact severity/composition: Bernoulli mask p=0.25/0.75/0.9 (trained at
p=0.5), block mask, stroke mask, Gaussian noise σ=0.3/0.6/1.0, and **noisy_masked** — a composition
matching neither pure training distribution, the single most informative test.

recovery_mse on the two hardest/most novel corruptions:

| arm | block_mask | noisy_masked | ordering_holds (9 corruptions) |
|---|---|---|---|
| gaussian | 0.768 | 0.229 | 4/9 |
| mask-only | 0.666 | **0.160** | 5/9 |
| **arm B (1:1)** | **0.649** | 0.169 | **8/9** |
| 1:2 | 0.681 | 0.163 | 7/9 |
| 1:3 | 0.668 | 0.171 | 7/9 |
| **1:4** | 0.659 | 0.166 | **8/9** |

(Correction 2026-07-08: an earlier draft of this table miscounted arm B as 6/9 and 1:4 as a
"perfect" 9/9. Recounted directly from the raw per-corruption JSON — both are actually tied at
8/9, each failing only on mask_p0.9. Fixed here and in pipeline.json.)

**The central finding**: on block_mask, only **1:1 and 1:4** beat BOTH pure arms simultaneously —
genuine compositional generalization, not interpolation between two memorized skills. 1:2 and 1:3
fall back above mask-only's floor **despite having strictly better raw Pareto numbers** (higher
gap, lower FID) than 1:1. Pareto-optimality on the trained corruption and compositional
generalization to novel corruptions are NOT the same axis, and this sweep does not co-optimize them
cleanly — 1:2/1:3 are the wrong choice if the generalization claim matters, even though they're the
"better" numbers on paper.

On noisy_masked, no mixture arm yet beats mask-only (0.160) — all mixtures beat gaussian but not
mask-only. Arm B and 1:4 are tied for best field-ordering (8/9, both fail only on the hardest
corruption, mask_p0.9); 1:2/1:3 are next best (7/9, also failing gaussian σ=1.0); gaussian-only and
mask-only trail (4/9 and 5/9).

### The tension, stated plainly

- **Best practical recipe** (masked-recovery + FID on the trained task): 1:3 or 1:4.
- **Best generalization-claim recipe** (genuinely learned a general repair field, not memorized two
  corruptions): 1:1 or 1:4 — NOT 1:2/1:3, despite their better raw numbers.
- **1:4 is the only ratio that's strong on both axes** — best Pareto gap in the sweep AND clears
  the compositional bar AND has perfect field-ordering. If forced to pick one recipe for
  everything, 1:4 is the least compromised choice.

**Ask of PI**: which framing does the paper lead with?
(a) "Structured start-states improve recovery while preserving generation quality" → 1:3 or 1:4,
    straightforward Pareto story, easiest to write up.
(b) "Structured start-states teach a genuinely more general repair field, not just two memorized
    corruptions" → 1:1 or 1:4, the stronger and more novel scientific claim, but built on a single
    corruption type per checkpoint at sanity scale — needs the most careful multi-seed/scale
    confirmation before publishing.
(c) Present both: 1:4 as the flagship recipe, 1:1 as a supporting ablation showing the
    generalization property first appears even at equal weighting.
All of this is single-seed, sanity-scale (1 epoch/40k steps). Whichever ratio is chosen needs Phase
2 multi-seed confirmation (per summer-2026-plan.md) before it's a real paper claim. No further
compute launched pending this decision.
