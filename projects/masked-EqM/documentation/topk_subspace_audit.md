# Phase 0 audit — Stage A top-k high-gain subspace diagnostic (2026-08-10)

Audit before building the Stage-A experiment (top-k subspace confirmation, gate before
any Gauss-Newton work). Per protocol: reuse working infra, do not silently duplicate it.

## 1. What already exists (reused verbatim)

- `experiments/direct_energy/matched_replay_jacobian_diagnostic.py`: `load_model`,
  `block_groups`, `is_backbone_group`, `build_pool`, `probe_direct`/`probe_none` (real
  pre-clip grad_norm ranking), `select_spike_control`, `replay_union`,
  `paired_self_minus_other`, `matched_interaction_bootstrap`, `ratio_row`, `field_of`,
  `total_norm`. All unchanged.
- `experiments/direct_energy/longitudinal_jacobian_diagnostic.py`: `evaluate_stage`
  (progress-matched replay at one stage), Table 1/3/4/7/10 construction, tail-quantile
  helper `_quantiles`/`tail_row`. Reused for the late-stage matched replay + own/cross
  amplification tables; Stage A adds a NEW pass on top rather than re-deriving these.
- `fb_direct/exact_hvp.py`: `exact_field_vjp`/`field_vjp_none` (canonical VJP,
  `A(x,r)=||J^T(r/||r||)||`), `field_jvp_direct`/`field_jvp_none` (matrix-free JVP,
  double-backward R-op trick), `power_iteration_theta_sigma1` (single-vector top
  singular value + left/right vector, validated 2026-08-10). All validated against an
  explicitly materialized Jacobian + numpy SVD/finite-difference at FP64 CPU
  (`tests/test_fb_direct_exact_hvp.py`, 12/12 PASS before this addition).

## 2. New this run

- `fb_direct/exact_hvp.py::block_subspace_iteration_theta` — orthogonal iteration
  (block power method) for the top-k singular SUBSPACE of J restricted to a fixed
  parameter subset. Alternates: apply J to a k-column orthonormal theta-space block,
  QR-orthonormalize the k output-space columns (gives U estimate), apply J^T to that,
  QR-orthonormalize again (gives V estimate + the R-factor diagonal as the singular
  value estimate). Deterministic init (seed), early-stops on relative max-singular-value
  change < tol. Chosen over per-vector deflation specifically because A9/A7 require
  subspace-level correctness under near-degenerate spectra (deflation is unstable there;
  orthogonal iteration converges the column SPACE, not individual vectors).
- Validated (`tests/test_fb_direct_exact_hvp.py::test_block_subspace_matches_explicit_svd`,
  FP64 CPU) against an explicitly materialized Jacobian + numpy SVD on
  `t_embedder.mlp.{0,2}.weight` (chosen because a 2-tensor MLP's singular values are
  plausibly close, i.e. exactly the near-degenerate regime the spec warns about): compares
  BOTH top-k singular values (rel_err < 1e-3) AND the subspace projector distance
  `||U_k U_k^T - Uhat_k Uhat_k^T||_F < 1e-2` (not individual-vector cosine, which would be
  invalid under degeneracy) plus explicit `||V^TV-I||`/`||U^TU-I||` orthonormality checks
  (< 1e-8). Result: **PASS both arms** (see test output).

## 3. Checkpoints (unchanged from `longitudinal_jacobian_audit.md`)

Only the LATE, progress-matched pair is used for Stage A (per spec A2/A4/A9, "at
minimum" late; earlier diagnostics already established early/mid are null/weak on the
existing A-based interaction, so compute is spent where the signal is):

| stage | direct checkpoint (step) | none checkpoint (step) | mismatch |
|---|---|---|---|
| late | `fwrev_ep80_lambda0_job37780076/.../2825000.pt` (2,825,000) | `longer80_none_seed0_ckpt50k_job36632776/.../2800000.pt` (2,800,000) | 0.885% |

## 4. Tail-severity criterion (A2) — pool sizing, criterion UNCHANGED

Spike/control selection rule is preserved EXACTLY (`select_spike_control`: top
`spike_frac` by real pre-clip grad_norm = spike, 40th-60th percentile band = control).
Only pool size changes per tail level, per Phase-1 discipline (widen the pool, not the
percentile):

| tail level | spike_frac | pool_size for K>=target | target K | actual K |
|---|---|---|---|---|
| top 2.5% | 0.025 | 2560 (reuse late-stage pool from job 38150502's design) | 64 | 64 |
| top 1.0% | 0.010 | 4800 | 48 | 48 |
| top 0.5% | 0.005 | 6400 | 32 | 32 |
| actual clip events | n/a | reuse the top-2.5% pool (2560); locked training clip threshold `max_grad_norm=6.87141` (from `test_gradient_clip_calibration.py`, pipeline final_metric, PASS run) | all recoverable | reported honestly (may be < 32; a live 2560-batch offline pool over a static checkpoint is not the same distribution as 3.2M live training steps, so the count of batches in this pool that exceed the LIVE clip threshold is an honest but conservative proxy, not a reconstruction of the actual historical clip-rate trajectory — documented, not hidden) |

Controls are NOT re-selected by residual magnitude at any tail level (same
40th-60th-percentile-by-grad_norm rule throughout).

To keep the job within the compute budget, the SAME 2560-batch pool ranked at Phase-0's
late stage is reused for the top-2.5% and "actual clip events" rows (spike selection is
just a different top-N cut of the same ranking; no re-ranking cost). The 1.0%/0.5% rows
need a WIDER pool (4800 / 6400 batches) ranked fresh under both models, since a tighter
percentile of the SAME 2560-pool would not reach the required K — this is the "widen the
pool" instruction applied literally per tail level.

## 5. Energy-head parameter grouping (A3)

`block_groups()` already tags `energy_head.*` (direct) / `final_layer.*` (none) as
non-backbone groups. `J_h` = Jacobian restricted to `[p for n,p in model.named_parameters()
if groups[n]=="energy_head"]` (direct) — printed at run start (tensor names/shapes/count),
per A3 requirement. `final_layer` is NOT the same architectural object as `energy_head`
(model-specific, as already documented in `matched_replay_jacobian_diagnostic.py`'s
`A_head` scope note) — none's head-parameter subspace pass is a SEPARATE, not
cross-model-comparable, quantity; only the "is the head-specific effect present in none
too, and how much" comparison (in the sense of relative spike/control ratios, not raw
values) is meaningful, exactly mirroring how A_head vs A_backbone was already handled.

## 6. sv_k / iteration budget

`sv_k=8` (k=8 top modes) primary, `sv_k=16` opportunistic if the k=8 pass's wall-time
allows (per A4 "if computationally cheap, also test k=16"). `num_iters=15` per subspace
pass (same default budget as the already-validated single-vector `power_iteration_theta_sigma1`,
which converged in 5 iterations on real B/2 batches per job 38150502's smoke test — a
block pass of similar size is expected to converge in a comparable number of iterations).

## 7. Disk / output budget

No `--out` JSON write attempted from compute nodes this session (holylabs quota has
failed silently-recoverable multiple times, per prior diagnostics' scope notes) — full
results are printed unconditionally to stdout, tee'd to the sbatch log, which is copied to
`slurm/logs/` on the home filesystem (small text, no quota risk). Per-batch data is kept
compact (row-oriented dicts, medians not full distributions except in the explicit
Table A2/A3/A4/A7 quantile/spectrum fields) to avoid the prior large-JSON-dump failure
mode.
