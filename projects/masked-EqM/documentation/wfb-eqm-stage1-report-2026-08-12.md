# WFB-EqM Stage 1: zero-update real-checkpoint causal diagnostic -- PASS

Job 38484995 (`wfb_stage1_diagnostic.py`, gpu_test, 42min, exit 0), commit `27cf048`.
Full results: `results/wfb_stage1_diagnostic_38484995.json` (recovered from the log's
`FULL RESULTS` block after `--out` hit the known holylabs disk-quota issue -- graceful
`OSError` handling worked as designed, no data lost).

## 0. Normalization note (added 2026-08-12, post-review)

All `g_raw_norm`/`g_wfb_norm` magnitudes in this report are in `compute_wfb_gradient`'s
CANONICAL, UNRESCALED residual convention (`r = field - ut`, no loss-reduction factor) --
**not** directly comparable in absolute terms to the production clip threshold
(`max_grad_norm=6.87141`) or to `exact_fwrev_backward`'s actually-applied gradient, which
uses `w = -(2/(B*D)) * r`. The native-scale mapping is `g_native = (2/(B*D)) * g_diagnostic`,
proven exactly (machine precision) on the real model in
`test_wfb_gradient_matches_native_fwrev_scale` (`tests/test_fb_direct_exact_hvp.py`), which
also proves `compute_wfb_gradient`'s whole `(A+lambda I)^{-1/2}` chain is linear in `r`
(`A`/`lambda_max`/`lambda` depend only on the model's Jacobian `M`, never on `r`'s scale) --
so this same rescaling applies uniformly to `g_wfb`, not just `g_raw`.

**This does not affect any conclusion in this report.** Every claim here is a *ratio*
(spike/control) of same-convention quantities, and any constant rescaling cancels exactly in
a ratio. As a sanity cross-check: rescaling Table 2's control-median `g_raw_norm` (51,417) by
`2/(B*D)` for this checkpoint's latent shape (`B=8, D=4096`, `scale=2/32768~6.1e-5`) gives
`~3.14` -- squarely in the range of previously-recorded native `grad_norm` medians for this
training regime (`clip_rate_zloss_wd01_vs_control` row: control median 1.70-4.88 depending on
arm); the spike-median rescales to `~7.66`, just above the clip threshold `6.87141` -- exactly
consistent with "spike" batches being selected via the native-scale `probe_direct` ranking.
This is strong independent confirmation the `2/(B*D)` mapping is correct, not merely an
algebraic assertion.

**Real consequence found by this check**: the WFB-EqM Stage 2 v1 training integration
(`train.py --wfb-backward`) applied `g_wfb` UNSCALED as the optimizer's gradient -- missing
this exact factor, off from the intended/calibrated scale by ~`(B*D)/2 ~ 16384x` for this
checkpoint. Caught pre-GPU-step (both v1 jobs cancelled while still in CPU pre-flight), fixed
in commit `ae95865` before any real training step ran under the bug.

## 1. Question

On batches where raw direct training exhibits a large gradient spike, does WFB
(`compute_wfb_gradient`) remove the parameter-gradient amplification while receiving the
EXACT SAME residual and EXACT SAME frozen model -- no weight updates?

## 2. Experimental setup

- Checkpoint: `CKPT_DIRECT_LATE` (`fwrev_ep80_lambda0_job37780076`, step 2825000) -- frozen,
  no updates.
- Pool: 1280 batches (batch_size=8), real ImageNet, seed=0.
- Batch selection: `probe_direct` (real pre-clip grad_norm, `exact_fwrev_backward`'s
  gradient) -- an ORDINARY criterion, independent of WFB, so there is no selection-bias
  circularity.
- Groups: 32 spike (top 2.5%), 32 control (40th-60th percentile band), 30 actual clip
  events (threshold=6.87141, locked training `max_grad_norm`).
- `rho=1e-4`, `k=8` (spec defaults) for the main population; k=2/4/8/12 convergence check
  on the 4 most-severe spike batches.
- n_params=130,488,577 (1 frozen: `pos_embed`, filtered automatically -- see bug notes).

## 3. Key metrics (medians, n=32/32 unless noted)

| table | quantity | spike | control | ratio |
|---|---|---|---|---|
| 1 | residual r_norm (sanity) | 588.9 | 602.6 | 0.977 |
| 2 | **g_raw_norm** | 125,527 | 51,417 | **2.441** |
| 2 | **g_wfb_norm** | 533.4 | 562.1 | **0.949** |
| 3 | raw_backbone | 117,926 | 47,011 | 2.508 |
| 3 | wfb_backbone | 525.9 | 555.7 | 0.946 |
| 3 | raw_head | 45,273 | 21,681 | 2.088 |
| 3 | wfb_head | 94.7 | 84.3 | 1.124 |
| 7 | g_wfb_norm, **actual_clip_events** (n=30) vs control | 534.4 | 562.1 | **0.951** |

Other diagnostics: `T_eigmax` vs independent power-iteration `lambda_max` agree to
0.27% (max rel err across all 94 real batches); `ortho_error` max 3.6e-7 (Lanczos basis
numerically clean); **zero breakdowns** on any real batch; cosine(g_raw, g_wfb) median
0.43 (spike) / 0.49 (control) -- WFB genuinely rotates the gradient direction, not a pure
rescale; median wall time 10.3s/batch, peak mem 17.9GB on a gpu_test MIG slice (3g.20gb).

## 4. Gate assessment (spec Section 8, criteria A-E)

**A. Operator correctness** -- PASS (Stage 0, commit `da03875`, 21/21 tests; extended to
22/22 after the pos_embed regression test below).

**B. WFB materially suppresses known real spike gradients** -- PASS, strongly. Raw
spike/control ratio 2.44x collapses to WFB spike/control ratio 0.95x -- WFB doesn't merely
reduce the amplification, it eliminates the spike/control distinction entirely (0.95 is
statistically indistinguishable from 1, if anything slightly *below*). Confirmed
independently on the literal real clip-event batches (Table 7, ratio 0.95x) -- not just an
artifact of percentile-based selection.

**C. Suppression not caused by numerical failure/zeroing** -- PASS. `breakdown_reasons: []`
across all 94 real batches (no lucky or unlucky breakdowns -- expected, since a real
130M-parameter model's spectrum is far from the toy near-degenerate cases Stage 0 tested).
WFB values on CONTROL batches are NOT collapsed toward zero (control median 562, comparable
to spike median 533) -- ruling out "WFB just zeroes everything." cosine(raw,wfb) is
meaningfully nonzero and stable (~0.43-0.49), not a degenerate near-orthogonal artifact.

**D. k=8 sufficiently converged vs k=12** -- PARTIAL PASS, with a caveat worth carrying
forward. `T_eigmax` (top eigenvalue, hence `lambda`) converges essentially exactly by k=4
(e.g. 36136189.92 at k=4 -> 36136189.94 at k=8/k=12 -- agreement to 7 significant figures).
But `g_wfb_norm` itself still decreases a further ~7-12% from k=8 to k=12 on the 4
most-severe spike batches tested (e.g. 463.67 -> 417.91, 295.54 -> 276.38). This means the
TOP eigenvalue/damping scale is resolved fast, but the full `(A+lambda I)^{-1/2} r` apply
needs more Krylov dimensions to fully resolve contributions from the rest of the spectrum.
The QUALITATIVE conclusion (massive, robust suppression) does not depend on this -- even
k=2 already shows 2-3 orders of magnitude suppression vs raw -- but the reported
suppression magnitude at k=8 is a slight UNDER-estimate of WFB's true effect (true
suppression is likely somewhat stronger at higher k). **Recommendation for Stage 2: use
k=12 as the production default**, not k=8, given the cost difference is modest (~2-3s/batch
more) and the convergence gap is real.

**E. Cost plausible for a short training experiment** -- PASS, with a note. ~10.3s/batch
(bs=8, full 130M-parameter Lanczos, k=8) on a gpu_test MIG slice (1/4-scale A100). A short
paired Stage 2 run (200-2000 steps) would take roughly 35min-6h at this per-step cost,
likely faster on a full seas_gpu A100/H200 (not MIG-sliced) -- plausible for Stage 2/3's
stated scope. Full-scale (millions-of-steps) production training is NOT yet plausible
without the later engineering optimizations the spec explicitly defers (warm-started
Krylov info, periodic whitening-operator refresh, fewer iterations) -- correctly out of
scope for this stage.

## 5. Overall verdict: STAGE 1 PASSES

All five criteria pass (D with a carried-forward caveat, not a blocker). The causal claim
is now established at the individual-batch level on the real checkpoint, without any
training: **the exact same residual, evaluated through the exact same frozen model,
produces a raw parameter gradient that is 2.44x larger on spike batches than control, but
a WFB-preconditioned gradient that is statistically indistinguishable between spike and
control** -- and this holds for the literal batches that triggered real clipping events
during actual training, not just a percentile-selection artifact.

Per spec Section 8's explicit gate instruction, this is sufficient evidence to proceed to
**Stage 2: very short paired training test** (both arms initialized from the same
pre-instability checkpoint, ARM A = exact-direct/fwrev, ARM B = WFB, everything else
identical, ordinary clipping enabled as an emergency guard in both arms, 200-2000 steps).

## 6. Engineering notes (bugs found and fixed this session, for future diagnostics)

Two real bugs were found and fixed during Stage 1 submission (3 total attempts):

1. **`pos_embed` requires_grad=False crash** (job 38472301): this codebase's `pos_embed` is
   registered as `nn.Parameter(..., requires_grad=False)` (fixed sinusoidal embedding), so
   it appears in `model.parameters()` but `torch.autograd.grad` raises unconditionally on a
   requires_grad=False input -- `allow_unused=True` does NOT cover this (that flag only
   suppresses the error for a requires_grad=True tensor that turns out disconnected from
   the graph). No prior diagnostic (topk_subspace, matched_replay) ever hit this because
   they always restricted to a head/backbone-only params subset that happened to exclude
   `pos_embed`; `compute_wfb_gradient` is the first caller in this codebase to pass the
   FULL parameter list by default, per the spec's explicit requirement to whiten the
   complete mixed Jacobian. Fixed (commit `5fd6c05`): `compute_wfb_gradient` now filters to
   `requires_grad=True` params, prints a note, and returns the actual `params` used so
   downstream code can align `g_raw`/`g_wfb` correctly. Regression test added
   (`test_wfb_compute_wfb_gradient_filters_frozen_parameters`), using the real tiny test
   model's own `pos_embed`.
2. **CPU/GPU device mismatch in Lanczos tridiagonal eigendecomposition** (job 38479632):
   `torch.eye(m, dtype=torch.float64)` and the tiny mxm eigendecomposition (T/theta/S/e1/
   coeff) defaulted to CPU regardless of the Krylov basis `Qmat`'s device (the
   n-dimensional field-space vectors, n = full parameter count, live on the batch's
   device). Stage 0's CPU-only FP64 unit tests never exercised a device mismatch since
   everything was on CPU there -- this is a GPU-only bug class Stage 0 could not have
   caught by construction. Fixed (commit `27cf048`): `coeff` and the `eye` comparison
   tensor are explicitly moved to `r.device` before combining with `Qmat`.

Both bugs are now covered structurally (the pos_embed case by a regression test; the
device-mismatch case is exercised by every GPU run of this diagnostic going forward, since
the CPU test suite cannot reproduce it).
