# Frozen projected-EqM inference experiment

## Question and scope

This is a frozen-field inverse-problem evaluation: can existing EqM vector
fields reconstruct held-out masked images when observed latent coordinates are
analytically reinserted after each existing sampler update?  It does not train
new models and does not interpret vector fields as exact scalar energies.

## Discovered implementation map

* `transport/corruption.py:mask_corrupt` constructs masked VAE-latent states.
* `eval_masked_recovery.py:gd_recover` is the sampler reference; its update is
  `state + step_size * model_output` (not an inferred energy-gradient sign).
* `eval_masked_recovery.py` supplies the EMA loading convention.
* `eval_generalization.py` supplies the prior structured-mask context.
* `frozen_prior_constraint.py` applies the projection immediately after that
  exact update; NAG look-ahead uses projected history.

The representation is Stable-Diffusion VAE latent space, with scale `0.18215`.
Internally `V=1` means observed/fixed and `V=0` means missing/free.  The
corrupted state is `V*clean + (1-V)*fill`; hard projection is
`V*clean + (1-V)*proposal`; soft correction is
`proposal + rho*V*(clean-proposal)`.

## Frozen checkpoints and selection

The manifest is `configs/frozen_prior_constraint_checkpoints.json`.  It uses
40k-step EqM-B/2 ImageNet-1K checkpoints, EMA when present: Gaussian seeds
0–2, Bernoulli seeds 0–2, and verified 1:1 Gaussian+Bernoulli seed 0 from
training job 31549055.  The former 1:1 mixed seed-1/2 paths no longer exist;
the available `job29233398` checkpoint is 1:2 and is excluded rather than
silently substituted.  Hence mixed-versus-specialist comparisons are explicitly
seed-0 paired-image evidence, not checkpoint-seed robustness evidence.

## Locked data and masks

Pilot and final manifests are immutable JSON files:

* `frozen_prior_constraint_pilot_locked_v4.json` — 256 disjoint ImageNet-val
  images across Bernoulli, block, combined, and irregular masks at 50% visible.
* `frozen_prior_constraint_final_locked_v4.json` — 2,000 final images per
  mask-family/visibility condition, with visibility 0.35, 0.50, and 0.65.

All conditions share dataset index, VAE encode seed, fill-noise seed,
initialization seed, and mask hash. Combined masks use a 20% block hole and a
Bernoulli visibility probability chosen for the target final visibility.

## Evaluation and validation

The evaluator writes resume-safe JSONL shards with checkpoint provenance,
paired-mask identifiers, sampler configuration, model-space regional MSE,
full/missing-composite LPIPS, missing-pixel PSNR, observed-pixel MSE, runtime,
and peak GPU memory. Hard-projection runs fail closed if the maximum observed
latent deviation exceeds `1e-6`.

Focused tests cover mask polarity and composition, deterministic manifests and
shards, GD/NAG sign equivalence, NAG projected history, hard/soft projection,
pilot/final disjointness, and synthetic regional metric separation.  Final
jobs additionally save deterministic clean/corruption/reconstruction/mask
artifacts for qualitative panels.

## Scheduler workflow

`slurm/jobs/frozen_prior_constraint_bundle.sbatch` contains resumable smoke,
pilot, and final stages without arrays, avoiding FASRC array-index submit-cap
pressure. `slurm/jobs/frozen_prior_constraint_aggregate.sbatch` performs
fail-closed aggregation after final shards complete.  Live state and recoveries
are recorded in `frozen-prior-constraint-run-status.md`.

## Limitations

Hard projection makes observed latent error exactly zero by construction; it
does not guarantee zero observed pixel error after VAE decoding.  Results must
be described as plug-and-play prior-field / observation-constrained inference,
not exact posterior-energy composition or proof that the EqM field is
conservative.
