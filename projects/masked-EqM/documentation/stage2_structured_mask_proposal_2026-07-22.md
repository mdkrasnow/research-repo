# Stage 2 variant proposal: structured-mask corruption

Mandatory pre-code variant proposal (AGENTS.md template), written before any Stage 2 code was
committed. Triggered directly by the Stage 1 gate-fail conclusion (2026-07-21): neither more
inference-time recovery steps nor more training epochs grows the cutoff-0.10 Fourier delta_G
effect, so the next lever per the pre-registered plan is a training-time change to the mask
corruption itself, not more compute on the existing elementwise-mask recipe.

## Variant name
`structured_mask` — spatially contiguous mask corruption (large blocks, medium patches, irregular
connected regions, minority elementwise), replacing the pure elementwise Bernoulli mask used in
all masked-EqM work to date.

## Hypothesis
The existing `mask_corrupt` erases latent pixels independently at random (elementwise Bernoulli,
`mask_prob=0.5`). Independent-pixel dropout is a texture/high-frequency corruption: even at
`mask_prob=0.5`, every small neighborhood retains some visible pixels, so the model can plausibly
solve the recovery task with local interpolation rather than semantic completion. This may explain
why training on it produces only a small, non-growing zero-shot transfer benefit to a genuinely
different corruption family (Fourier low-pass, which removes *global* frequency content, not local
texture). A mask corruption built from large contiguous missing regions instead forces the model to
fill in structure with no local evidence in a sizeable neighborhood — closer, mechanistically, to
"reconstruct missing global/coarse structure from partial evidence," which is also what Fourier
low-pass recovery requires (coarse structure present, detail/high-freq missing... inverted: here,
detail present, whole regions missing). The shared demand is *inference under structured, not
independent, missingness*. Hypothesis: training against structured missingness generalizes better
to the Fourier task than training against independent-pixel missingness, because it can't be solved
by local interpolation alone.

## Failure mode addressed
Stage 1's flat/non-growing delta_G across 1/2/3/5 epochs (max +0.0034, never exceeding the +0.008
promotion bar) combined with Stage 1's own robust, large delta_M — meaning G+M reliably beats
mask-only, but only marginally and unreliably beats gaussian-only. This is consistent with the
elementwise mask task being "too easy" / too locally-solvable to teach anything Fourier-relevant
beyond what gaussian-only already gets from the base EqM objective. Structured (contiguous) mask
corruption is the most direct one-parameter-family fix that keeps everything else (build order,
CLAUDE.md Step 2/4 framing, choke point) unchanged.

## EqM compatibility argument
No change to the EqM training objective, target, or loss (`transport.py:167-217` untouched). This
variant only changes how `x0` (the training-time start state) is constructed in `Transport.sample`,
exactly the same choke point as the existing `mask`/`fourier`/`blur`/`downsample` corruptions
(`transport/corruption.py`). `structured_mask_corrupt` has the identical mathematical form as
`mask_corrupt` — `z0 = m*x1 + (1-m)*eps`, same masked-fraction control via `mask_prob` — only the
*spatial correlation structure* of `m` changes (contiguous instead of i.i.d. Bernoulli per pixel).
Since `path.py`/`training_losses` only require `x0` to match `x1`'s shape/dtype and be finite, and
`m` remains a valid {0,1}-valued keep-mask satisfying `E[mask_prob]` calibration by construction,
this is a strictly lower-risk change than any of the CLAUDE.md "high-risk loss" categories (no new
loss term, no EBM-style energy, no Jacobian penalty, no cosine/hinge objective) — it is a **pure
data/corruption-distribution change**, the same risk class as the already-adopted `mask`/`blur`/
`downsample` corruptions.

## Loss definition
Unchanged EqM base loss (`terms['loss']` in `Transport.training_losses`), applied identically
regardless of `corruption_mode`. No auxiliary loss term is introduced.

## Corruption definition
`z0 = m ⊙ x1 + (1-m) ⊙ ε`, `ε ~ N(0,I)`, same as `mask_corrupt`. `m` (spatial keep-mask, shared
across the 4 latent channels) is drawn per-sample from a 4-way categorical mixture over mask
*families*, matching the exact structure specified in the Stage 2 directive:
- **block** (weight 0.35): 1-2 axis-aligned rectangular regions, random aspect ratio (log-uniform
  in [0.5, 2]), random placement, total zeroed area ≈ `mask_prob` of the latent grid.
- **patch** (weight 0.35): non-overlapping `4x4`-cell coarse grid (MAE-style), random subset of
  cells zeroed until `mask_prob` fraction of grid area covered.
- **region** (weight 0.20): one irregular 4-connected region grown via random flood-fill from a
  random seed pixel until `mask_prob` fraction of grid area covered.
- **elementwise** (weight 0.10, minority): the original i.i.d. Bernoulli mask, kept as a small
  minority so the model isn't trained on contiguous structure exclusively (avoids overfitting to
  "always contiguous" and keeps a link back to the original task).

No Fourier-domain operations anywhere in this corruption (spatial-domain only, per the Stage 2
directive's explicit constraint) — keeps this variant orthogonal to the frozen Fourier evaluation
by construction, not just by intent.

Implementation: `transport/corruption.py` — `_block_mask_single`, `_patch_mask_single`,
`_region_mask_single`, `make_structured_mask` (categorical dispatch, per-sample), and
`structured_mask_corrupt` (applies the corruption formula). Wired through
`Transport.__init__`/`.sample` (new `corruption_mode="structured_mask"` branch, and a
`structured_mask_weight` mixture arm for `corruption_mode="mixture"`), `create_transport`,
`train_utils.parse_transport_args` (`--structured-mask-weight`), and `train.py`'s
`create_transport(...)` call site. All existing corruption modes/defaults are byte-identical to
before this change (verified: `gaussian` path untouched, new params default to 0/unused).

## Expected diagnostics if working
- Trained mask-recovery LPIPS (structured-mask task, analogous to existing `mean_mr_lpips`) shows
  the structured-mask-only and gaussian+structured-mask arms learning a real, non-trivial recovery
  skill (LPIPS well below a shuffled/no-op baseline), i.e. training is not broken or trivial.
- delta_G on the frozen cutoff-0.10 Fourier eval, comparing gaussian+structured-mask 1:1 vs
  gaussian-only, exceeds the analogous elementwise-mask delta_G at 1 epoch (+0.0034, not
  significant) by a clear margin, ideally reaching the pre-registered Stage 2 promotion bar
  (delta_G >= 0.010).
- Per-image win rate vs gaussian-only exceeds Stage 1's 63-65% ceiling, moving toward or past 75%.
- FID for the gaussian+structured-mask arm stays within +15 of matched gaussian-only (generation
  quality not destroyed by the harder training corruption).

## Expected diagnostics if failing
- delta_G stays flat/small (comparable to or worse than Stage 1's already-failed range), suggesting
  the "local-interpolation vs semantic-completion" hypothesis is wrong, or that the shared demand
  between structured-mask recovery and Fourier low-pass recovery is weaker than hypothesized.
- Structured-mask-only or gaussian+structured-mask FID regresses sharply beyond +15 of matched
  gaussian-only (the harder corruption destabilizes generation, mirroring the FID cost seen for
  mask-only historically in the mixture-ablation work, per `summer-2026-plan.md` open question 3).
- Trained structured-mask recovery LPIPS is unexpectedly *worse* than the original elementwise task
  at the same `mask_prob` and, combined with a null delta_G, suggests the harder corruption is
  simply harder without being differentially useful.

## Minimal test
CIFAR-scale sanity smoke first (per CLAUDE.md CIFAR-sanity-rule): confirm `structured_mask_corrupt`
runs without shape/NaN errors, produces visibly non-trivial (not degenerate all-zero or all-one)
masks across the 4 families at the target `mask_prob`, and a short training run's loss is finite
and comparable in scale to the existing elementwise-mask CIFAR run. This CIFAR check answers "does
the code run, is the mask distribution sane" — it does NOT answer whether the IN-1K Fourier-transfer
hypothesis holds (per CLAUDE.md CIFAR-sanity-rule, CIFAR cannot answer that). The actual test of the
hypothesis is the full IN-1K-scale Stage 2 comparison specified below.

## Promotion rule
Per the pre-registered Stage 2 gate (summer-2026-plan.md / pi-updates.md 2026-07-21 entry):
1. mean delta_G >= 0.010 (higher bar than Stage 1's 0.008, since Stage 2 is a fresh single-epoch
   comparison, not a longer-training extension).
2. 3/3 screening seeds beat both gaussian-only and structured-mask-only.
3. Per-image win rate vs gaussian-only > 75%.
4. FID within +15 of matched gaussian-only.
5. Structured-mask recovery (trained task) is strong (clearly better than a no-op/shuffled
   baseline, consistent with the model actually learning the corruption's inverse).

## Kill rule
Any of: delta_G stays below Stage 1's already-failed range; FID regression beyond +15; structured-
mask recovery does not show a working, non-degenerate trained skill; or the same failure pattern
(small win rate, small/non-significant delta_G) repeats as Stage 1. On kill: write postmortem, do
NOT auto-launch the optional narrow-ratio follow-up (2:1/1:1/1:2) or model scaling — that follow-up
is only in-scope if structured masking improves the Fourier effect but harms generation, per the
original Stage 2 directive; a flat null on delta_G itself ends the branch, escalate to user for next
direction.

## Scope limits (explicit, per Stage 2 directive)
- 1-epoch controlled comparison only, 3 matched seeds x 3 arms (gaussian-only, structured-mask-only,
  gaussian+structured-mask 1:1) — no multi-epoch extension unless this gate passes.
- No Fourier-domain operations in the corruption itself (spatial-domain only, satisfied by
  construction above).
- Frozen Fourier evaluation (cutoff 0.10, 250 steps, same 1024-image manifest, same cached
  latents/sampler/labels) is completely unchanged — reusing `eval_fourier_recovery.py`/
  `.sbatch` verbatim, no parameter or subset changes.
- Qualitative grids only for a promoted candidate (per directive) — not built speculatively before
  the gate is evaluated.
