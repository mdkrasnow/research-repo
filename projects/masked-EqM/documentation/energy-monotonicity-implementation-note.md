# Checkpoint-only energy monotonicity evaluation

## Repository interfaces

- `models.py::EqM.forward(x, t, y, get_energy=True)` is the only model stack used.
  `none` returns the trained vector field. `dot` forms `z·f(z)`, differentiates it,
  and returns that gradient; its public energy value is the negative scalar
  `-z·f(z)`. `direct` forms one scalar per sample and returns `-grad(E)`.
- `transport/transport.py::Transport.training_losses` samples
  `z_t=(1-t)x0+t*x1` through `ICPlan`, then trains against
  `(x1-x0)c(t)`. Thus the trained/model-returned field is the sampling direction
  from noise to data, not the gradient of a decreasing physical energy.
- `train.py` trains on ImageNet `ImageFolder` images after deterministic center
  crop plus a random horizontal flip, normalizes pixels to `[-1,1]`, and encodes
  them with `stabilityai/sd-vae-ft-ema`, scaled by `0.18215`. Evaluation removes
  the flip and uses deterministic VAE posterior draws.
- Checkpoints are dictionaries containing `model`, `ema`, `opt`, `args`,
  `epoch`, and `step`. The standard evaluation policy is EMA. The evaluator
  requires EMA for every selected checkpoint unless explicitly configured
  otherwise.
- The longer-training comparison uses `EqM-B/2`, ImageNet-1K, 256px images,
  Gaussian corruption, class labels, and `uncond=True`. Even though time is
  ignored under `uncond=True`, the exact gamma is passed to the model.

## Sign convention used by the evaluator

The requested raw effective-field definitions are preserved:

- `none`: the trained vector output;
- `dot`: `grad_z sum(z*f(z))`;
- `direct`: `grad_z E(z)` (the negative of the field returned by
  `EqM.forward`).

Because the repository trains the model-returned field toward `(x-epsilon)c(t)`,
the raw `none` and `dot` line integrals should increase toward data, whereas the
native `direct` scalar and its gradient should decrease. The evaluator records
`expected_direction` in every checkpoint result and converts the raw line
integral to a canonical decreasing energy only by a fixed, repository-derived
sign (`-1` for `none`/`dot`, `+1` for `direct`). This sign is written to
`config.json` before metrics are computed and is never chosen from results.

The dot scalar validation uses the explicitly requested `z·f(z)` scalar. The
direct validation uses the native scalar head. Neither validation uses the
sign-adjusted canonical energy.

## Integration points

`energy_monotonicity/evaluate_energy_monotonicity.py` owns checkpoint discovery,
the fixed held-out latent/noise bank, field extraction, line integration,
metrics, paired image-cluster bootstrap, caching, plots, examples, and report
generation. It imports the existing model registry and checkpoint loader.
Focused mathematical tests live in
`tests/test_energy_monotonicity.py`. Cluster execution is provided by a
checkpoint-only SLURM wrapper; it neither invokes `train.py` nor writes to any
checkpoint path.
