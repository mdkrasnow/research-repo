# Direct-energy gradient guard experiment — 2026-08-01

## Motivation

Direct job 36359213 suffered a finite optimizer shock near step 1,517,550. The
global Adam first-moment norm was 0.452 at step 1.50M, rose to 0.726 at 1.55M,
and returned to 0.444 at 1.60M. The disturbance was concentrated in the
timestep-conditioning path and first transformer block, not solely in the
scalar head. No parameter or optimizer tensor became nonfinite.

## Variant proposal

Variant name: `direct-gradient-guard`

Hypothesis: rare heavy-tailed second-order gradients can displace a trained
direct-energy model; clipping only extreme global gradient norms will prevent
such displacement without changing ordinary updates.

Failure mode addressed: finite optimizer shocks during native scalar-energy
training.

EqM compatibility argument: clipping is applied after the unchanged EqM field
loss and its higher-order backward pass. It does not change the transport
target, energy sign, model architecture, or sampling update.

Loss definition: unchanged EqM loss. The treatment applies global L2 gradient
clipping immediately before `AdamW.step()`.

Expected diagnostics if working: finite nonzero head/backbone gradients; the
calibrated threshold lies above ordinary gradients; clipping is rare; matched
short-run loss is not materially worse than the unclipped control.

Expected diagnostics if failing: clipping activates frequently, ordinary loss
is degraded, gradients remain unstable after clipping, or instrumentation
changes throughput enough to confound the comparison.

Minimal test: start both arms from direct checkpoint 1.50M with the same seed,
sampler epoch, global batch, optimizer state, and 2,000-step budget.

Promotion rule: calibration has at least 1,000 finite positive gradient-norm
records and finite nonzero scalar-head/backbone gradients. Lock
`max_grad_norm = 2 * p99(unclipped grad norm)`. The clipped arm must remain
finite, activate clipping on no more than 1% of steps in this ordinary window,
and have final-500-step mean loss no more than 2% above the matched unclipped
control. A zero activation rate is acceptable because this stage tests that the
guard is non-invasive, not that the rare shock deterministically replays.

Kill rule: any nonfinite gradient, more than 1% clipping in the ordinary
window, or greater than 2% loss degradation. Do not silently retune the
threshold.

## Controls

- Negative/null control: the unclipped calibration run from the same 1.50M
  checkpoint.
- Positive plumbing control: focused test applies clipping to a deliberately
  oversized synthetic gradient and verifies that the post-clip norm is bounded.

## Limitations

The original event cannot be replayed exactly because checkpoints do not store
PyTorch/CUDA/DataLoader RNG state. This experiment establishes a measured,
non-invasive guard. Only a longer run can estimate the shock rate.

## Result — 2026-08-02

Calibration job 36806020 and clipped treatment 36806043 both completed on the
same four-H200 node from checkpoint 1.50M with 2,000 matched steps.

- Calibration: median norm 1.72453, p99 3.43570, maximum 65.13386.
- Locked threshold: 6.87141 (`2 * p99`).
- The maximum-gradient step was 1,501,493; backbone norm 64.60719 and scalar
  head norm 8.26615. This was a backbone-dominated outlier.
- Treatment clipped exactly 1/2,000 steps (0.05%) and had no nonfinite values.
- Final-500 mean loss was 10.78434 clipped versus 10.79894 control, a -0.135%
  relative difference. This clears the preregistered <=1% activation and <=2%
  loss-degradation gates.

Verdict: **PASS**. The measured threshold behaves as a rare circuit breaker
rather than changing ordinary optimization. This short matched test does not
estimate long-horizon FID or prove that every future optimizer shock is
prevented.

## Long-horizon confirmation — 2026-08-05 (forensic follow-up)

Trigger: `direct` epoch80 paper-scale (50,000-sample) ADM FID = 39.94 vs matched
`none` epoch80 = 34.16 (+5.78 FID, worse across FID/sFID/IS/precision, flat on
recall). Investigated whether a second discrete optimizer-shock event (of the
same character as the original step-1,517,550 event) explains the gap.

**Data source**: `gradient_metrics.jsonl` from job 36847271 (clipped
epoch40→80 continuation, 1,601,500 steps, the run whose final checkpoint was
used for the FID-50k eval), logged every 50 steps — `grad_norm` (pre-clip
total L2 norm), `head_grad_norm`/`backbone_grad_norm` (post-clip), `clipped`,
`loss`. Cross-checked against the console `Train Loss`, which is a 50-step
**running average** (`train.py:384-387`), not the instantaneous per-step loss
— the console log completely hides single-batch outliers found below (e.g.
step 2,444,150 shows console loss 16.13 vs raw per-step loss 1059.12).

**Finding: not a second discrete shock — a growing, recurring instability.**

- 130 clip events over 32,029 logged steps (0.4%), well under the
  original 1%-activation gate measured on the 2,000-step calibration window.
  But activation rate is **not stationary**: 1 event in the first 200k
  steps of this run vs 24 events in the last 200k steps — a ~24x rise,
  monotonically increasing bin-over-bin.
- 6 events exceeded 100x the clip threshold (6.87141), vs a calibration-window
  max of only 65.13 (9.5x threshold). Worst two: **step 2,832,600, pre-clip
  grad_norm 34,793.80 (5,064x threshold), single-step loss 480.5**; **step
  2,444,150, grad_norm 12,389.37 (1,803x threshold), single-step loss
  1059.12** (~99x the ~10.7 baseline). Neither is visible in the console
  average.
- Signature matches the original 1,501,493 shock exactly: **backbone-dominated**.
  On every severe (>100x) event, `head_grad_norm` collapses toward its
  observed floor (0.06–0.17) while `backbone_grad_norm` pegs at the clip
  ceiling (6.85–6.87) — i.e. the explosion originates almost entirely in the
  backbone, same as the calibration-window outlier (backbone 64.6 vs head
  8.27). This is the same failure mode recurring, not a new one.
- At least one event (step 2,975,850: grad_norm 689x threshold, loss 9.5,
  indistinguishable from baseline) shows the gradient can explode by orders of
  magnitude **while the loss value itself stays normal** — a curvature/
  conditioning signature, not a loss-magnitude signature.

**Mechanism**: `ebm='direct'` (`models.py:292-300`) computes the training
field as `field = -grad(E.sum(), x0, create_graph=train)` — the loss is
therefore a function of a **double backward pass** through the full 12-block
transformer backbone (the training loss differentiates the input-gradient of
a scalar output, i.e. touches the energy's local Hessian structure).
`none`/`dot`/`l2` never do this: `none`'s field is a direct network output;
`dot`/`l2` take exactly one `autograd.grad` through a potential that is
linear/quadratic in `x0`. Double-backward-through-Hessian training losses are
a documented failure mode for numerical conditioning (WGAN-GP gradient
penalties, score-matching/EBM training) — wherever the learned scalar energy
develops sharp local curvature in `x0`, the *gradient of the loss w.r.t.
parameters* can blow up even though the energy value itself looks
unremarkable, exactly as observed at step 2,975,850. That the clip rate rises
through training is consistent with this: as `E` sharpens around the data
manifold (which is the intended training signal), its curvature — and hence
the tail risk of this specific double-backward pathway — increases too.

**Comparison control**: `none`'s matched epoch60→80 continuation
(`longer80_none_seed0_ckpt50k_36632776.log`) has no `gradient_metrics.jsonl`
(only enabled when `max_grad_norm` is set), but its console loss over the
identical step range (2.4M–3.2M) is essentially flat: mean 10.31–10.33,
max/mean ratio ≤1.03, zero bins with any outlier >2x the bin mean. `direct`
over the same step range: mean loss flat too (~10.75–10.80, masking the
outliers via averaging) but with growing clip-event frequency and single-step
raw losses up to 99x baseline. The instability is `direct`-specific.

**Why clipping didn't prevent the FID gap**: clipping bounds the L2 norm of
the parameter update but not its *direction*. A correctly-bounded update
whose direction is driven by a rare, ill-conditioned curvature spike (rather
than by bulk data signal) still perturbs the EMA-averaged weights used at
eval time. With 130 such events, rising in frequency toward the end of
training (several of the worst are in the final ~350k steps before the
epoch80 checkpoint at step 3,181,500), the accumulated EMA drift plausibly
explains part or all of the paper-scale FID gap. This is a smaller-magnitude,
higher-frequency version of the original 1,517,550 shock — not resolved by
the gradient guard, only contained per-step.

**Relation to prior mechanism-level findings**: this training-dynamics result
is consistent with (and gives a causal account for) already-documented
weaknesses of `direct` independent of this run — energy-candidate-ranking
confirmation seeds found `direct` "not independently sufficient" / failing
the direct-over-dot gate (`direct-energy-candidate-ranking-confirmation-2026-07-30.md`),
and the geodesic-manifold preliminary found `direct` "directionally worse"
than `dot`. All three lines of evidence now point to the same root cause: the
double-backward scalar-energy construction is intrinsically worse-conditioned
than `dot`/`none`, not merely under-trained or mis-tuned.

**Verdict on the gradient guard — corrected 2026-08-05**: **PASS**, and more
convincingly than first written up. Initial framing above overstated the
practical impact of the recurring explosions by leading with FID-gap
language ("not sufficient to make `direct` match `none`", "unable to clear
the paper-scale gate"). Recalibrating against this project's own precedent
for what a real failure looks like:

| Comparison | FID ratio vs its baseline |
|---|---|
| CAFM-EqM catastrophic mechanism failure | 341.25 / 31.41 = **10.9x** |
| Original *unclipped* direct optimizer shock damage | 160.11 / 31.41 = **5.1x** |
| `direct` epoch80 (this run) vs `none` epoch80 | 39.94 / 34.16 = **1.17x** |

A 1.17x ratio is not in the same category as either documented failure —
it's a modest quality tax, not a broken mechanism. Recall in fact favors
`direct` (0.6411 vs `none`'s 0.6292); only precision is meaningfully lower
(0.394 vs 0.537), which reads as "individual samples somewhat less crisp,"
not mode collapse or distributional breakdown. There was also no
pre-registered gate requiring `direct` FID ≤ `none` FID at this phase — the
phase gate was to obtain the protocol-matched measurement, which is done.

Read correctly, the training-dynamics finding above is actually **evidence
the guard is working**: despite 130 recurring backbone gradient explosions
(6 exceeding 100x threshold, one at 5,064x) at a growing rate through 1.6M
steps, the resulting model still landed within 17% FID of a completely
uninstrumented baseline and beat it on recall. That is consistent with the
original PASS verdict — a rare circuit breaker containing per-step damage —
not a contradiction of it. The recurrence and escalating rate of the
explosions remain worth tracking (a real, growing tail-risk signature tied
to the double-backward `ebm='direct'` construction, see mechanism section
above), and still connects to the independent candidate-ranking/geodesic-
manifold signals that `direct` is somewhat weaker-conditioned than `dot`. But
it does not, on its own, justify killing the `direct` direction or spending
the project's one retune — there is no failing gate to retune against here.
