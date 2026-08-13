# WFB-EqM Stage 3 audit: the completed three-way 300-step comparison

**Date**: 2026-08-13
**Status**: audit complete. Supersedes the interpretation in
`wfb-eqm-stage3-postmortem-2026-08-13.md` (which was written with only the FBGN
arm finished; see §4, which retracts its central causal claim).

## 1. Question

Before spending another multi-hour run on a repaired FBGN optimizer: what do the
three completed Stage 3 arms actually say, and is the held-out probe they were
judged on trustworthy?

## 2. Setup

All three arms: no Adam, no momentum, Armijo backtracking line search on
`eta_star = r.q/||q||^2`, 300 steps, batch size 8, single GPU, same starting
checkpoint and same seed.

- Start checkpoint: `CKPT_DIRECT_LATE` =
  `/n/netscratch/.../fwrev_ep80_lambda0_job37780076/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/2825000.pt`
- Trainer: `experiments/direct_energy/wfb_stage3_lm_trainer.py`
- Commit at submission: `7bc4e9d` (alpha 0.0 / 0.5), FBGN CG arm after the CG fix.
- Metrics fetched to `results/wfb_stage3/*.jsonl` (300 rows each).

## 3. Three-way table

| arm | job | probe start | probe end | net delta | accepted / rejected | median same-batch residual removed | median `eta` | median `R = actual/pred` | median `lambda_max` (first10 → last10 avg) | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| direct (alpha=0.0) | 38904393 | 10.6244 | 10.8324 | **+0.208** | 300 / 0 | 4.5 % | 3.8e-6 | **0.991** | 6.5e5 → 6.6e5 (1.0x) | 5:44:54 |
| WFB (alpha=0.5) | 38903397 | 10.6244 | 11.2145 | **+0.590** | 300 / 0 | 13.6 % | 2.5e-3 | **0.911** | 8.6e5 → 1.7e6 (2.0x) | 5:47:18 |
| FBGN (alpha=1.0, CG) | 38909657 | 10.6245 | 14.8811 | **+4.257** | 300 / 0 | 16.4 % | 1.19 | **0.528** | 1.7e7 → 6.1e8 (36x) | 2:33:20 |
| FBGN + LM adaptive damping (20-step smoke) | 38944527 | 10.7669 | 12.7593 | **+1.992** | 20 / 0 | 17.8 % | 1.26 | **0.453** | 2.1e7 → 3.6e7 | 0:25:39 |

Parameter displacement is not directly logged; `eta` × `||g||` is its per-step
proxy and is reported per-arm in the raw metrics.

### 3.1 The headline the FBGN-only postmortem missed

**Every arm worsens the held-out probe, and the harm is monotone in alpha.**
The negative control (alpha=0, raw `M^T r`, no whitening at all) also goes the
wrong way. So "FBGN diverges" is not a statement about FBGN's geometry alone —
it is the extreme point of a gradient that runs through the whole family, ordered
exactly as the arms' step aggressiveness is ordered.

### 3.2 The FBGN probe trajectory is NOT a divergence

| step | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 275 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe | 13.86 | 15.48 | 16.87 | 14.12 | 13.71 | 12.46 | 13.37 | 13.40 | 12.89 | 15.06 | 13.47 | 14.88 |

The damage is essentially **complete by step 25** (10.62 → 13.86, i.e. 78 % of the
total 300-step delta in the first 8 % of the run). After that the probe is a
stationary noisy band in [12.5, 16.9] with no upward trend — the last-150-step
values straddle the first-150-step values.

### 3.3 The LM adaptive-damping smoke did not help

`rho` moved only 1e-4 → 2e-4 over 20 steps and the probe still rose +1.99 in 20
steps (a steeper per-step rate than the fixed-`rho` 300-step run's +4.26/300).
Median `R` was 0.453, i.e. *worse* than the fixed-`rho` run's 0.528. The
reduction-ratio controller as configured barely engages, and the little it did
engage bought nothing. This is a data point against a damping-only repair, not
proof against it — §5 tests it properly.

## 4. Retraction: the postmortem's causal claim is not supported

`wfb-eqm-stage3-postmortem-2026-08-13.md` concluded:

> `lambda_max` ... trended up ~36x on average ... Every same-batch loss spike
> coincides EXACTLY with a `lambda_max` spike ... This is the mechanism.

Three independent reasons that does not stand as a causal account of the probe
damage:

1. **Timing decoupling.** `lambda_max` grew ~36x monotonically across 300 steps.
   The probe damage was done by step 25 and then flat (§3.2). A cause that keeps
   growing for 275 steps after its supposed effect saturated is not the cause of
   that effect.
2. **The damping is scale-invariant in exactly the way that would cancel it.**
   Damping was set as `lambda = rho * lambda_max` each step. Under a uniform
   rescaling `M -> cM` we have `A -> c^2 A` and `lambda -> c^2 lambda`, so
   `A(A+lambda I)^{-1}` — the operator that determines the FBGN direction and its
   induced field gain — is **invariant**. A growing `lambda_max` alone therefore
   changes nothing about the step, by construction.
3. **The `lambda_max` sequence was measured on different stochastic batches.**
   It is a per-batch quantity sampled along a trajectory, not a trajectory-level
   quantity. The same objection applies to reading the late `L_before` spikes
   (40.7 at step 285, 130.6 at step 292) as "the trajectory jumped to a high-loss
   region": those are fresh, never-before-seen batches, and `L_before` across the
   run has median 14.3 with a heavy right tail. They are draws from a
   heavy-tailed per-batch loss distribution, not a parameter-space excursion.

The postmortem's *backbone-vs-head localization* finding (backbone sigma_1 1.53x
vs head 0.95x, job 38942662) is unaffected — that was measured on a fixed batch
pool across two checkpoints and remains a valid description of where curvature
moved. What is retracted is only the claim that this growth **caused** the probe
damage.

## 5. Probe-integrity audit (§3 of the directive) — PASS, with one caveat

Inspected `wfb_stage3_lm_trainer.py` + `build_pool` +`build_probe_bank`.

**The probe is deterministic.** `build_pool` materializes the full list of
`(xt, t, y, ut)` tensors ONCE at job start and freezes them; the probe is a slice
of that frozen list. Concretely, everything the directive requires to be fixed is
fixed at pool-build time and never resampled:

| variable | where it is frozen |
|---|---|
| ImageNet examples + labels | `build_probe_bank`, seeded `torch.Generator` on the DataLoader |
| VAE latents | `vae.encode(x).latent_dist.sample()`, called once per pool index |
| `t` and the Gaussian `eps` (`x0`) | `transport.sample(x1)`, called once per pool index |
| corrupted latent `z` (= `xt`) and target `y` (= `ut`) | `transport.path_sampler.plan(...)`, stored `.detach()`ed |

`probe_loss_avg` then only does forward evaluations on those stored tensors, and
`load_model` leaves the model in `.eval()` (no CFG label dropout). So the reported
within-run trajectory `10.625 -> 14.881` **is** a like-for-like comparison of the
same field-regression examples. No repair needed; no re-evaluation of checkpoints
required before proceeding.

Empirical confirmation: alpha=0.0 and alpha=0.5 — two separate jobs — report
`probe_loss_initial` identical to all 16 significant figures (10.62436830997467).
The alpha=1.0 job differs in the 5th decimal (10.624534), consistent with a
different GPU/kernel selection, not with resampled data.

**Caveat (fixed going forward).** The probe slice is
`pool[max_steps : max_steps + n_probe_batches]`, so it moves when `--max-steps`
changes. That is why the 20-step LM smoke reports a different initial probe
(10.7669) — it is a *different* probe, not a worse checkpoint. Cross-run probe
comparisons are therefore only valid between runs with the same `--max-steps`.
All three 300-step arms share `--max-steps 300` and are mutually comparable.
Stage 3A pins the offset explicitly (`--train-steps 300`) so its probe is
byte-identical to the one the 300-step runs reported.

**Checkpoints available**: each 300-step arm saved `step100`, `step200`,
`step300` (`--ckpt-every 100`), all present on netscratch. Intermediate
checkpoints therefore exist and are used by Stage 3A; no re-run is needed to
manufacture them.

## 6. What the existing metrics can and cannot decide

The runs already log `rho_lm = actual_delta_L / predicted_delta_L`, which is
exactly the Levenberg–Marquardt reduction ratio, and it separates the arms
cleanly (0.991 / 0.911 / 0.528). Read naively that says the FBGN local model is
bad — H1.

**It cannot be read that way.** `R` is confounded with step length: FBGN takes
`eta ~ 1.19` full Gauss–Newton steps while direct takes `eta ~ 3.8e-6`. Any
smooth model is accurate in the limit `eta -> 0`, so a lower `R` for the arm
taking vastly longer steps is what H1 **and** H2 both predict. The logs never
varied `eta` at a fixed direction, and never evaluated any direction against data
other than the batch that produced it. Both gaps must be closed before choosing a
repair.

Note also that the arms' ordering by probe harm is exactly their ordering by
*fraction of its own minibatch residual each step removes* (4.5 % / 13.6 % /
16.4 %, max 79.6 % for FBGN). That is the signature H2 predicts — the more
completely a step solves its own 8-image problem, the more it costs on held-out
data — but it is equally consistent with H1, since those are also the longest
steps. It is a hypothesis to test, not a conclusion.

## 7. PASS/FAIL

- Probe integrity: **PASS** (deterministic; no repair required).
- Three-way audit: **complete**.
- CG-FBGN as a training algorithm: **FAIL**, confirmed, and now known to be the
  extreme point of a failure that affects all three arms including the negative
  control.
- Postmortem's `lambda_max`-causation claim: **RETRACTED** (§4).

## 8. Hypotheses eliminated

- Eliminated: "the probe measurement was noisy/regenerated corruption" — it is
  byte-deterministic.
- Eliminated: "FBGN specifically is broken while direct training is fine under
  this globalization" — direct also worsens the probe (+0.208).
- Eliminated (as a causal account): "`lambda_max` growth drove the divergence."
- Eliminated: "adaptive damping keyed to the reduction ratio is an obvious fix" —
  the 20-step smoke made `R` worse, not better. (Not yet falsified as a
  *mechanism*; properly tested in Stage 3A's `--lambda-sweep`.)

## 9. Next action

Stage 3A frozen-checkpoint mechanism discriminator
(`experiments/direct_energy/wfb_stage3a_mechanism_discriminator.py`,
`slurm/jobs/wfb_stage3a_discriminator.sbatch`), which closes both gaps in §6:
it scans `eta` at a FIXED direction (separating step length from direction) and
measures the eta-independent infinitesimal transfer `d_V = grad L_V . p` on an
independent trust bank (deciding H2 outright, since `d_V > 0` cannot be repaired
by any step size or damping).
