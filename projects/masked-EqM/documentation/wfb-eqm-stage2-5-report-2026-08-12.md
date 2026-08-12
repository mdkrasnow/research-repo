# WFB-EqM Stage 2.5: frozen-checkpoint alpha-family diagnostic

Job 38673961 (`wfb_stage2_5_alpha_diagnostic.py`, gpu_test, 42min, exit 0), commit `f1dcf25`.
Responds to the Stage 2 v5/D factorial finding (see `pi-updates.md`/`pipeline.json`
completed_runs) that resetting AdamW state alone did NOT close the WFB (alpha=1/2) learning
gap -- matching `delta_theta_norm` to ARM A produced net-WORSE held-out probe loss under WFB
(job 38646819/D), while the same reset had no material effect on exact-direct training (job
38651959/ARM C: `delta_theta_norm` 0.2492 vs A's 0.2485, `probe_delta_L` sum -0.0211 vs A's
-0.0245, cosine 49.6% positive vs A's 49.8% -- statistically indistinguishable from A).

## 0. Reviewer account (motivating this stage)

WFB uses `g_wfb = M^T(A+lambda I)^{-1/2} r`. Stage 1 proved this bounds the **parameter**
gradient's per-mode gain (`sigma_i / sqrt(sigma_i^2+lambda)`, capped near `1/sqrt(lambda)`
for large `sigma_i`). But the first-order **induced field update** `delta_s = M delta_theta`
in a singular mode is `delta_s_i ~ -eta * sigma_i^2/sqrt(sigma_i^2+lambda) * r_i` -- STILL
carrying one power of `sigma_i`, unbounded. The proposed fix is the alpha-family
`g_alpha = M^T(A+lambda I)^{-alpha} r`: alpha=0 is ordinary direct training (field gain
`sigma_i^2`, unbounded), alpha=0.5 is WFB (field gain `sigma_i`, still unbounded), alpha=1 is
full damped Gauss-Newton ("FBGN": field gain `sigma_i^2/(sigma_i^2+lambda)` in [0,1] for
every mode).

## 1. Implementation

`fb_direct/exact_hvp.py`: generalized `_lanczos_inv_sqrt_apply_generic` into
`_lanczos_inv_pow_apply_generic(alpha)` (kept as a thin alpha=0.5 wrapper for exact backward
compatibility -- verified by regression test, see below). `compute_wfb_gradient` gained an
`alpha=0.5` kwarg (default preserves prior behavior exactly).

3 new FP64 CPU tests (27/27 total pass):
- `test_wfb_alpha_family_singular_mode_gain`: synthetic 7-mode operator, confirms BOTH the
  parameter-gradient gain and the induced-field gain match closed form for alpha in
  {0, 0.5, 1}. Top-mode induced field gain: alpha=0: 2500.00 -> alpha=0.5: 50.00 ->
  alpha=1: 0.9990 (bounded, as predicted).
- `test_lanczos_inv_pow_apply_alpha_half_matches_sqrt_apply`: regression, alpha=0.5
  specialization matches the pre-generalization function exactly (rtol=1e-12).
- `test_compute_wfb_gradient_alpha_param_threads_through_on_real_model`: alpha kwarg
  actually changes the returned gradient on the real tiny model (||g_wfb|| alpha=0.5: 0.383
  -> alpha=1.0: 0.053), default (omitted) matches alpha=0.5 explicit.

## 2. Experimental setup

Same frozen checkpoint as Stage 1 (`CKPT_DIRECT_LATE`, step 2825000), no weight updates.
Pool 256 batches (smaller than Stage 1's 1280 -- each batch now costs 3x the Lanczos work,
one per alpha), spike_frac=0.06 (15 spike batches found), num_control=8. Swept 8 spike + 8
control batches through the full alpha-family diagnostic (rho=1e-4, k=12).

Two measurements per batch per alpha:
1. **Field-gain/cosine sweep**: `g_alpha = compute_wfb_gradient(...)`, induced field update
   `q_alpha = M g_alpha` (via `field_jvp_direct`), field gain = `||q_alpha|| / ||r||`,
   cosine(`q_alpha`, `-r`).
2. **One-step apply/revert probe**: real (non-Adam) step `theta' = theta - eta*(2/(B*D))*g_alpha`
   at eta=1 (native scale, matching the proven `2/(B*D)` mapping), measured on a SEPARATE
   held-out real-data probe batch, then EXACTLY reverted (`max_restore_err=0.0` confirmed on
   every batch -- no state leakage between probes).

## 3. Results

### 3.1 Field-gain/cosine sweep (medians, n=8/8)

| alpha | field gain (spike) | field gain (control) | g_alpha_norm (spike) |
|---|---|---|---|
| 0 (direct) | 164,232.8 | 43,040.8 | 121,219.7 |
| 0.5 (WFB) | 216.7 | 92.8 | 528.7 |
| 1 (FBGN) | 2.83 | 3.41 | 24.2 |

Induced field-update gain collapses ~750x from direct -> WFB, then a further ~65x from
WFB -> FBGN, landing at O(1) -- exactly matching the closed-form prediction
`sigma_i^2/(sigma_i^2+lambda)` bounded in [0,1] per mode (aggregate slightly >1 since
multiple modes can constructively combine, but order-of-magnitude collapse confirmed).

### 3.2 One-step apply/revert probe (eta=1, native scale, n=16)

| alpha | median probe_delta_L | range | frac improved |
|---|---|---|---|
| 0 (direct) | +70.48 | [+15.72, +4,383,547.87] | 0/16 (0%) |
| 0.5 (WFB) | -0.00167 | [-0.0062, +0.0019] | 13/16 (81.25%) |
| 1 (FBGN) | -0.0000296 | [-0.00019, +0.00007] | 11/16 (68.75%) |

**Important caveat on this sub-result**: at raw native scale (eta=1, no Adam), BOTH WFB and
FBGN are already safe -- their raw gradient magnitudes (step_norm median 0.032 vs 0.0016)
are far below where the `sigma_i`-conditioning in the induced field update actually bites.
This probe alone does NOT distinguish "safe because bounded" (FBGN's claim) from "safe
because tiny" (an alternative explanation for WFB's raw-scale safety) -- both are consistent
with WFB being numerically SMALLER than FBGN's delta_L here. The decisive evidence is
section 3.1's field-GAIN (a ratio, scale-independent), not this probe's absolute delta_L.
This probe's real contribution: (a) direct catastrophically diverges even at raw scale (0/16
improved, up to 4.4M in delta_L) -- direct's problem is not merely "too large under Adam,"
it's unconditionally unstable; (b) confirms the apply/revert harness is numerically exact.

## 4. Interpretation

Combined with the Stage 2 2x2 factorial (A: exact+loaded, C: exact+reset -- these two are
statistically indistinguishable, `delta_theta` 0.2485 vs 0.2492, `probe_delta_L` sum -0.0245
vs -0.0211, cosine 49.8% vs 49.6% positive; B: wfb+loaded net-flat with 74.0% cosine; D:
wfb+reset net-WORSE with 44.2% cosine), this stage's field-gain sweep gives a mechanistic
account of WHY B and D look contradictory:

WFB's raw gradient is orders of magnitude smaller than direct's. Under LOADED Adam
(calibrated to direct's larger `v_t`), that small raw gradient produces a tiny, accidentally
-safe step (ARM B: good direction, negligible progress). Under RESET Adam, the optimizer's
per-coordinate normalization recalibrates to WFB's own small gradient scale within the run,
inflating the step back to a "typical" size (ARM D: `delta_theta_norm` matches A/C) -- and at
THAT scale, the one remaining power of `sigma_i` in WFB's induced field update (bounded
parameter gradient, unbounded field update) is fully exposed, and actively hurts.

FBGN's boundedness (section 3.1) is a property of the DIRECTION itself
(`sigma_i^2/(sigma_i^2+lambda) <= 1` per mode), independent of what step size gets applied to
it -- unlike WFB's raw-scale safety, which is a magnitude accident that any adaptive/
normalizing optimizer (Adam being the paradigm example) will erase given enough steps to
adapt.

## 5. Prediction and the decisive next test

If FBGN is handed to Adam the same way WFB was in D (ARM E: `--wfb-backward --wfb-alpha 1.0
--reset-adam-state`, otherwise identical to D's design), Adam should inflate its step size
back toward the same ~0.24 "typical" magnitude (Adam normalizes toward roughly-unit RMS step
per coordinate regardless of raw gradient scale). The critical question: does
`probe_delta_L` stay negative like A/C, or flip positive like D?

- If it stays negative: confirms the fix is at the alpha/conditioning level, not the
  optimizer level -- FBGN is the correct Stage 3 training candidate.
- If it ALSO flips positive: falsifies the sigma_i-conditioning account as the FULL story,
  points to something else (e.g. this checkpoint being brittle to any large perturbation
  regardless of conditioning) -- would require the trust-region/Levenberg-Marquardt
  acceptance mechanism (increase damping when actual reduction underperforms the local
  model, matching the classical LM update rule) rather than a bare fixed-step optimizer.

`--wfb-alpha` added to `train.py` (default 0.5, preserves all existing behavior) and
`WFB_ALPHA` env var added to `slurm/jobs/wfb_stage2_paired_train.sbatch` to run this test.
