# WFB-EqM Stage 3: no-Adam Forward-Backward Gauss-Newton training

**Status**: proposed + implemented, smoke-testing before full run (2026-08-13).

## Variant proposal

**Variant name**: Stage 3 no-Adam FBGN/WFB trainer (Armijo backtracking line search).

**Hypothesis**: WFB (alpha=0.5) and FBGN (alpha=1.0) are trainable directions once the
step size is chosen by the SAME geometry that produced the direction (a local quadratic
model + backtracking line search), rather than by Adam's coordinate-wise, geometry-blind
adaptive rescaling -- which the Stage 2 v5/D factorial showed can re-inflate a
conditioning-bounded direction back to an unsafe magnitude.

**Failure mode addressed**: ARM D (WFB + reset-Adam) showed a properly-conditioned
gradient DIRECTION (Stage 1: bounded parameter-gradient gain) still degraded once Adam's
per-coordinate normalization matched its raw magnitude to ARM A's typical step size --
i.e. Adam's adaptivity acted as an independent source of geometry-blind step-size
inflation, orthogonal to (and potentially undoing) whatever the WFB/FBGN preconditioner
was protecting against.

**EqM compatibility argument**: The trainer operates on the SAME canonical residual
`r = field - ut` and the SAME `g_alpha = M^T(A+lambda I)^{-alpha} r` already proven
(Stage 0-2.6) to be the exact/damped-Gauss-Newton family for the EqM regression target.
No new loss term, no new operator -- only the OPTIMIZER changes (backtracking line
search on `g_alpha` directly, replacing Adam).

**Loss definition**: unchanged, `L = mean_flat((field-ut)**2).mean()` (identical to
`exact_fwrev_backward`'s `loss_main`).

**Mechanism**: Armijo backtracking line search (Nocedal & Wright, *Numerical
Optimization*, 2006, Algorithm 3.1) -- the standard globally-convergent globalization
strategy for Newton-type descent directions. At each step: compute the locally-optimal
linearized step size `eta_alpha* = r.q_alpha / ||q_alpha||^2` (Stage 2.6b's closed form,
`q_alpha = M g_alpha`), then backtrack (halving) from `eta_alpha*` until the sufficient-
decrease (Armijo) condition holds, or reject the step if no such `eta` is found within
`--max-backtracks`. No momentum, no per-coordinate rescaling -- the ONLY adaptivity is
this scalar, theoretically-grounded step-size search.

**Controls (mandatory per project rule)**:
- **Negative control (floor)**: `alpha=0.0` (raw/direct `M^T r`, provably unbounded
  per-mode field-update gain, sigma_i^2) run through the IDENTICAL backtracking harness.
  Tests whether proper globalization ALONE -- with no WFB/FBGN preconditioning --
  already fixes the instability. If alpha=0 also trains cleanly here, that falsifies
  "the preconditioner is necessary" and redirects the story to "Adam specifically was
  the problem." If alpha=0 still thrashes/stalls/has a low accept rate (expected, given
  Stage 2.5's alpha=0 field gain measured in the tens of thousands), that confirms the
  preconditioner is doing necessary work, not just the line search.
- No positive-control oracle is available at this scale (no ground-truth optimal
  direction); WFB vs FBGN vs direct, all under the identical globalization, is itself
  the informative three-way comparison.

**Expected diagnostics if working**: high accept rate (most steps find a sufficient-
decrease `eta` within a few backtracks), monotonically-or-noisily-decreasing held-out
probe loss over the run, `eta_used` roughly stable (not collapsing toward the smallest
backtrack floor every step).

**Expected diagnostics if failing**: near-zero accept rate (line search can never find
a decrease -- direction is not locally useful beyond first order), probe loss flat or
increasing, `eta_used` collapsing to the backtrack floor every step (direction technically
descdescent but the model is so nonlinear at that curvature scale that any real step size
overshoots).

**Minimal test**: this IS the minimal test -- single-GPU, no DDP, k=96 (Stage 2.6a's
established adequate-convergence point on this checkpoint), 300 steps per arm
(~5h/arm at k=96's measured ~50-60s/step), same frozen `CKPT_DIRECT_LATE` and
`GLOBAL_SEED` across all three arms for a paired comparison.

**Promotion rule**: if WFB or FBGN shows a clearly higher accept rate AND net-negative
probe-loss trajectory than the alpha=0 negative control over 300 steps -> proceed to a
longer/DDP paper-scale confirmation run (per this project's CIFAR/proxy-scale
discipline: this diagnostic-scale result is a FILTER, not publishable on its own).

**Kill rule**: if all three arms (including WFB/FBGN) show near-zero accept rate or
flat/worsening probe loss -> the backtracking-line-search globalization itself is
insufficient at this checkpoint's curvature scale; would need either a genuine
Levenberg-Marquardt adaptive-damping scheme (increase `rho`/lambda on repeated
rejection, not just shrink `eta`) or reconsider whether this checkpoint (already deep in
the pre-instability regime) is simply too nonlinear locally for ANY first-order-model-
based step-size rule -- would require testing from an EARLIER, less-degraded checkpoint.

## Implementation

`fb_direct/exact_hvp.py`: unchanged (reuses `compute_wfb_gradient`, `field_jvp_direct`,
`compute_field_direct` verbatim).

`experiments/direct_energy/wfb_stage3_lm_trainer.py` (new): single-GPU trainer
implementing the algorithm above. `slurm/jobs/wfb_stage3_lm_trainer.sbatch` (new):
`ALPHA` env var selects the arm; `RUN_TAG`/`GLOBAL_SEED` for pairing.

Sanity-checked the closed-form `predicted_delta_L` and Armijo accept condition exactly
(rho=1.0-equivalent agreement) on a linear toy model before spending GPU time (matches
Stage 2.6b's validation pattern).

## Plan

1. Smoke (`--max-steps 20`, all three alphas) -- confirm no crashes, finite losses,
   sane accept/reject behavior, before committing to the full 300-step runs.
2. Full run (300 steps x 3 arms, paired seed) overnight.
3. Analyze: accept rate, probe-loss trajectory, `eta_used` stability, per Promotion/Kill
   rules above.
4. Postmortem + `pipeline.json`/`results_variants.tsv` update regardless of outcome.
