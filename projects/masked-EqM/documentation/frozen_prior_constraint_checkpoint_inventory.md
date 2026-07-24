# Frozen-prior constrained-inference checkpoint inventory

## Pre-registration

This experiment reuses frozen vector-field EqM checkpoints only.  Its primary
set is the recoverable, architecture-matched EqM-B/2 ImageNet-1K 40,000-step
set in `configs/frozen_prior_constraint_checkpoints.json`: three Gaussian-only,
three Bernoulli-mask-only, and three 1:1 Gaussian+Bernoulli checkpoints.
The 1:1 ratio was selected from prior work before this evaluation; it is not
being selected by combined-mask performance.

The `0040000.pt` files are 2,086,843,408 bytes.  The evaluator loads `ema`
when present, exactly as `eval_masked_recovery.py` does.  The checkpoint
inventory deliberately excludes the later Stage-2 structured-mask checkpoints:
they do not include a matched Bernoulli-only parent and answer a different
training question.

## Discovered implementation map

* Training-time mask corruption: `transport/corruption.py:mask_corrupt`;
  it uses `m*x1 + (1-m)*eps` and one mask shared by latent channels.
* Model representation: Stable-Diffusion VAE latents, scaled by `0.18215`.
* Existing recovery evaluation and EMA loader: `eval_masked_recovery.py`.
* Validated sampler: `eval_masked_recovery.py:gd_recover`; GD updates
  `xt = xt + stepsize*out`; NAG applies its look-ahead before the same update.
* Existing held-out block/stroke utilities: `eval_generalization.py`.
* Existing SLURM convention: `slurm/jobs/eval_masked_recovery.sbatch`.

`V=1` means visible/fixed throughout the new evaluator.  Projection is in
latent space after the repository sampler step: `V*clean + (1-V)*proposed`.
The vector-field checkpoints are never described as exact energies.
