# WFB-EqM: implementation plan + Stage 0/1 record (2026-08-11)

## Question

Does mixed input-parameter Jacobian whitening (WFB) causally eliminate the growing
scalar-energy EqM training instability (documented `direct` vs `none`, see
`postmortem-stageA-topk-subspace-2026-08-11.md` for the full prior investigation) while
preserving useful field learning?

## Repository forensics (Section 6)

- `M := d(field)/d(theta) = d^2 E/(dz dtheta)` is ALREADY computed by existing
  `fb_direct/exact_hvp.py` primitives: `exact_field_vjp(model,xt,t,y,v)` = `M^T v`,
  `field_jvp_direct(model,xt,t,y,params,v)` = `M p`. No new Jacobian machinery needed.
- No `--direct_backward_mode` flag exists yet in `train.py`; the natural hook point is
  alongside the existing `--exact-fwrev` boolean (train.py:332-382), which already
  branches between `loss.backward()` and `exact_fwrev_backward`. WFB training integration
  (Stage 2+) will add this as a sibling backward-mode selector, gated the same way
  (`ebm == 'direct'` required).
- Diagnostic infra (`build_pool`/`select_spike_control`/`rank_pool`/`probe_direct`/
  `load_model`/`block_groups`/`is_backbone_group`) in `experiments/direct_energy/
  matched_replay_jacobian_diagnostic.py` is reused verbatim for Stage 1 -- no raw batch
  tensors are pickled anywhere; pools are rebuilt deterministically from a fixed seed
  each run (see that module's docstring).
- Checkpoints (true root netscratch, not holylabs -- see `longitudinal_jacobian_audit.md`):
  `CKPT_DIRECT_LATE=/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/fwrev_ep80_lambda0_job37780076/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/2825000.pt`
  -- well past the ~1.5M-step instability onset, in the elevated (~24x grown) clip-rate
  regime.
- sbatch idiom: copy `slurm/jobs/topk_subspace_diagnostic.sbatch` (seas_gpu, 16h,
  `python -u` unbuffered per this session's earlier buffering lesson, `SBATCH_EXPORTS`
  env-var mechanism, unit tests run before the diagnostic).

## Implementation design (Section 15)

Added to `fb_direct/exact_hvp.py` (Stage 0 commit `da03875`):
- `mixed_gram_mv(model,xt,t,y,params,v)`: `A v = M(M^T v)`, reusing `exact_field_vjp` +
  `field_jvp_direct`.
- `estimate_lambda_max` / `_estimate_lambda_max_generic`: matrix-free power iteration for
  `lambda_max(A) = sigma_1(M)^2`, generic core injectable for synthetic-operator testing.
- `lanczos_inv_sqrt_apply` / `_lanczos_inv_sqrt_apply_generic`: k-step reorthogonalized
  Lanczos approximation of `(A + lambda I)^{-1/2} r` via the tiny tridiagonal's explicit
  eigendecomposition (Saad 2003 Ch. 13). Distinguishes a genuine numerical failure
  (non-finite A-apply) from a LUCKY breakdown (Krylov subspace found A-invariant early --
  exact, not approximate, answer in fewer than k steps -- classical symmetric-Lanczos
  behavior, not a bug) from a trivial zero-residual input.
- `compute_wfb_gradient`: orchestrates `g_wfb = M^T(A+lambda I)^{-1/2}r`,
  `lambda = rho*lambda_max(A)`, always also returning the hypothetical raw gradient
  `M^T r` as a diagnostic (Section 9/18 requirement -- the "raw diagnostic gradient in
  WFB" trace). Fails loudly (raises) only on a genuine numerical failure, never silently
  degrades to a fallback.
- `compute_field_direct`: field value alone, no theta-graph -- used to confirm the
  backward-operator swap does not change the model's forward output.

## Stage 0 result: PASS (commit `da03875`)

21/21 FP64 CPU tests pass, including two new WFB-specific tests:
- `test_wfb_operators_match_explicit_jacobian`: explicit-Jacobian toy test on a real tiny
  model (n_out=64, n_params=384) -- `mixed_gram_mv`, `estimate_lambda_max`, and
  `lanczos_inv_sqrt_apply` (k=n_out, exact Krylov recovery) all match an explicitly
  materialized/eigendecomposed Jacobian to machine precision (rel_err ~1e-14 to 1e-5);
  norm bound `||g_wfb|| <= ||r||` holds; field is unchanged by the backward-operator swap.
- `test_wfb_singular_mode_gain`: synthetic M with a hand-built extreme spectrum
  (sigma = [50, 20, 5, 1, 0.5, 0.2, 0.1]) confirms per-mode WFB gain
  `sigma_i / sqrt(sigma_i^2 + lambda)` exactly, including 50x suppression of the extreme
  top mode at `rho=1e-3`.
- `test_wfb_zero_residual_breakdown`: zero residual reported as an explicit breakdown,
  not silently substituted.

Gate: PASS. Proceeding to Stage 1.

## Stage 1 design (Section 8)

`experiments/direct_energy/wfb_stage1_diagnostic.py` -- zero-update (no weight updates),
frozen `CKPT_DIRECT_LATE`, real batches. Batch selection uses the ORDINARY pre-clip
grad_norm criterion (`probe_direct`, i.e. `exact_fwrev_backward`'s gradient norm) --
independent of WFB itself, so there is no selection-bias circularity in asking "does WFB
suppress batches an unrelated criterion already flagged as hard."

Pool: 1280 batches (single tail level, top 2.5% spike + 40th-60th-percentile control band,
`num_control=32`), plus an `actual_clip_events` group (threshold=6.87141, locked training
`max_grad_norm`). `rho=1e-4` (spec default), `k=8` (spec production default) for the main
population; a `k=2,4,8,12` convergence check runs on only the 4 most-severe spike batches
(bounding compute -- full-parameter-count Lanczos is heavier than Stage A's head-only
subspace estimator, since each Lanczos step costs one full-model VJP + one full-model JVP
over ALL ~130 parameter tensors, not a restricted subset).

Metrics collected per batch: `r_norm`, `g_raw_norm`, `g_wfb_norm`, head/backbone split of
both, `lambda_max`/`lam`/`T_eigmax` (cross-check agreement), Lanczos breakdown status,
`cosine(g_raw, g_wfb)`, wall-clock, peak memory.

Gate to Stage 2 (Section 8, criteria A-E): operator correctness (A, already passed at
Stage 0); WFB materially suppresses known real spike gradients, i.e. WFB spike/control
ratio substantially closer to 1 than the raw spike/control ratio (B); suppression not
caused by numerical failure/zeroing, i.e. breakdown reasons are benign and WFB does not
collapse to near-zero uniformly on control batches too (C); k=8 sufficiently converged
vs k=12 (D); runtime/memory cost plausible for a short training experiment (E). Evaluated
against job output once complete -- not scripted as an automatic pass/fail, per this
session's established practice of the agent making the explicit gate call from printed
evidence (see Stage A postmortem for precedent).
