# Paired geodesic manifold-adherence benchmark

This is a checkpoint-only, **single-seed preliminary** comparison at the frozen
epoch-15 dot/direct checkpoints.  It is deliberately not a three-seed
confirmation and cannot emit the experiment's proposed pass/fail claim.

## Frozen conventions

- Scalar definitions are exactly the existing implementation: dot is
  `sum(z*f(z))`, direct is the native scalar head.
- Signs are fixed before evaluation: dot `-1`, direct `+1`.  A failed on/off
  ordering aborts; the evaluator never flips a sign.
- The solver has 33 points, an open-uniform cubic B-spline, eight free interior
  controls, linear initialization, three seeded restarts, and restart selection
  solely by the Riemannian kinetic objective.
- The metric calibrates clean held-out images to 1 and random held-out midpoints
  to 1000.  A non-positive linear metric aborts rather than being silently
  clamped.  The exponential metric is intentionally not used in the primary
  preliminary run; add it only as a separately labelled secondary configuration.

## Required bank contract

`bank.pt` is an immutable, disjoint held-out artifact containing:

`calibration_latents`, `calibration_labels`, `reference_images`,
`endpoint_latents`, `endpoint_labels`, `endpoint_images`, and `pairs`.

The caller must create it from ImageNet validation only, with no image appearing
in more than one bank. `pairs` must be same-class, have no reused endpoint, and
be selected by frozen DINOv2 distance between the 50th and 90th within-class
percentiles. This strict input contract is checked before any checkpoint load.

The preliminary sizing is intentionally modest (for example 256 calibration
images, 512 references, 64 pairs); the eventual primary protocol uses
2,000/5,000/1,000 pairs.  The summary records a paired 10,000-resample endpoint
bootstrap, but explicitly labels it **descriptive** because there is one model
seed. It reports DINOv2 and Inception features, manifold excess, precision,
D-RMSE, and detour. It does not treat paths as independent trained models.

`none` is not scalarized: in the full extension it is a model-free linear/slerp
control and a clearly-labelled gradient-norm secondary control, never a primary
energy comparison.
