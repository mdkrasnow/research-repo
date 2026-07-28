# Held-out EqM Energy Calibration Test — implementation note

## Reused implementation

- `energy_monotonicity/evaluate_energy_monotonicity.py`: `CheckpointRecord`,
  checkpoint manifest/hash validation, EMA model loading, frozen-bank metadata,
  atomic output helpers, and cache format.
- `energy_monotonicity/core.py`: `get_effective_field` (the validated canonical
  field source), `trapezoid_line_integral`, and paired image-cluster bootstrap.
- The completed evaluator output is `/n/home03/mkrasnow/energy_monotonicity_full`.
  Its bank is `evaluation_bank.pt`, SHA-256
  `45a6dfd993fda9cc0c73794a7d4c249baf4ae932b2220298cbd19c8e9c64aa60`, with
  2,048 held-out ImageNet validation images and two stored Gaussian draws/image.
  Its epoch-1/8 effective-field caches are the reuse source, subject to manifest,
  bank, grid, sign, EMA, precision, and field-definition checks.

## Confirmed source conventions

`transport/transport.py::Transport.get_ct` implements
`c(gamma) = 4 * min(1, 5 * (1 - gamma))`.  The epoch-8 checkpoint metadata says
Gaussian corruption, EqM-B/2, 256px, class labels, `uncond=True`, and EMA weights.
The checkpoint manifests are `configs/energy_monotonicity_{none,dot,direct}.json`.

The validated `get_effective_field` helper returns raw fields and records fixed
canonical signs `none=-1`, `dot=-1`, and `direct=+1`.  Thus the calibration
canonical gradient is positive in the `epsilon-x` direction and corresponds to
an energy increasing from clean to noise.  `dot` is always differentiated from
the per-sample scalar `sum(z * f(z))`; its raw vector output is never used as its
energy gradient.  `direct` requires exactly one scalar/sample.

## Planned additions

`energy_monotonicity/evaluate_energy_calibration.py` will consume the frozen
bank/caches, derive the exact clean-anchored target curve using the analytic
integral of the source schedule, calculate NECE and diagnostics, run the shared
cluster bootstrap, validate scalar-versus-line agreement, and emit the required
tables, parquet files, plots, examples, report, and resumable cache metadata.
Focused mathematical tests will be added to `tests/test_energy_calibration.py`;
an evaluation-only SLURM wrapper will leave all model weights untouched.
