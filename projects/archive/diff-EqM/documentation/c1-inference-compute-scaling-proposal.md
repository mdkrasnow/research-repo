# C1 — Inference-Compute Scaling — Capability Proposal

Per CLAUDE.md "Research Process Rules for EqM / ANM". Mechanism check filled before code.
Capability experiment (not a training variant): adapts the variant-proposal template, swapping
"Loss definition" → "Measurement definition". **Frozen checkpoints only — no retraining.**

## Name
`c1_inference_compute_scaling`

## Hypothesis
EqM sampling is gradient descent on an explicit scalar energy `E` (models.py:257-272; field
`f = ∇_{x0} E`, equilibrium when `f → 0`). Vanilla EqM's field is only accurate in a thin tube
around the noise→data trajectory (Exp 2: vanilla MSE rises off-trajectory). Therefore, as you spend
**more** optimization steps, the vanilla trajectory eventually leaves the tube where its field is
trusted and the iterate wanders / overshoots — sample quality plateaus then degrades.

v10 (ANM) trains the field to be accurate off-trajectory (Exp 2 CONFIRMED, dose-responsive:
λ=0.3 ~2× the field-robustness gap of λ=0.1). Prediction: **ANM keeps descending toward a lower-energy
equilibrium for many more steps than vanilla; its quality-vs-compute curve stays monotone where
vanilla's turns up.** If true, ANM is not "+4 FID" — it gives EqM a property it didn't have:
**usable inference-time compute scaling** (more solver steps ⇒ better samples).

## Failure mode addressed
The current 4-experiment suite scores final-sample distribution (FID) at a single, tuned NFE. That
washes out the landscape benefit ANM actually produces. C1 measures the landscape directly: behavior
of the optimization as compute grows.

## EqM compatibility argument
Pure measurement on the native EqM sampler (`xt += f·η`, `t += η`, NFE = num_sampling_steps−1;
sample_gd_fixed.py:161-176). No objective change, no new loss, no reinterpretation of the target.
We sweep the existing `--num-sampling-steps` / `--step-mult` levers and read off energy + FID.
The scalar energy `E` is already exposed (`get_energy=True`) — equilibrium residual `‖∇E‖ = ‖f‖`
is the model's own convergence signal, not an imported one.

## Measurement definition
For each arm ∈ {vanilla (FID 31.41), v10 λ=0.1 (29.01), v10 λ=0.3 (27.09)}, frozen EMA ckpts:

1. **Quality-vs-compute curve.** Sweep NFE ∈ {25, 50, 100, 150, 250, 400, 600, 1000} at matched
   total integration (rescale η so the t-schedule endpoint is held fixed; also run a second sweep
   that holds η fixed and lets extra steps act as pure equilibrium refinement past t=1). 5K-sample
   FID per cell (deltas only — absolute 5K FID is inflated, per Exp 1).
2. **Energy / residual trace.** Every K steps log per-sample `E` and `‖f‖` (already cheap via
   `get_energy=True`). Curves: `E(step)`, `‖f(step)‖`, fraction of samples with `‖f‖` still
   decreasing.
3. **Stability.** Reuse Exp 1 bookkeeping: NaN rate, divergence rate (`‖xt‖ > thresh`), clip
   fraction — as functions of NFE.

Reuse `experiments/exp1_sampler_robustness/` harness (already has frozen-latent matched-arm sampling,
deltas-not-absolutes discipline, SIGPIPE-safe logging). **Exp 1's existing 80-cell 5K run already
contains the gd/ngd × NFE grid — first deliverable is a re-analysis of that data through the C1 lens
(near-zero new compute), before extending the NFE range upward.**

## Expected diagnostics if working
- Vanilla FID-vs-NFE is U-shaped: improves to ~nfe 100-150, then **rises** (overshoot). [Exp 1
  interim already shows vanilla worse than ANM at nfe≥100 and the crossover at low NFE.]
- ANM FID-vs-NFE monotone-decreasing or flat-low far past vanilla's turn-up point.
- ANM reaches **lower equilibrium energy** `E` and lower residual `‖f‖` at high NFE; vanilla's `‖f‖`
  bottoms out then climbs (leaving its trusted tube).
- Effect ordered by dose: λ=0.3 scales further than λ=0.1 than vanilla (matches Exp 2 dose-response).
- ANM divergence/NaN rate stays flat as NFE grows; vanilla's climbs.

## Expected diagnostics if failing
- Both arms plateau at the same NFE with the same floor → no scaling advantage; ANM gain is purely
  the fixed-NFE FID and the capability claim collapses.
- ANM also turns U-shaped at the same point as vanilla → off-traj robustness too small to matter for
  the sampler at B/2 (consistent with the B/2 NULL capability eval; escalate to λ↑ or XL/2 before
  killing).
- `‖f‖` traces identical between arms → energy landscape indistinguishable along the sampled path.

## Minimal test
Re-analyze the existing Exp 1 80-cell 5K CSVs (`documentation/exp1_data/*.csv`): plot FID-vs-NFE
deltas per arm, mark vanilla's turn-up, check ANM monotonicity. **Zero new compute.** If the
crossover/monotonicity signature is present → extend NFE upward (add cells at 400/600/1000) with the
energy-trace logging patch, λ∈{0.1,0.3}. ~1 GPU-day for the extension.

## Promotion rule
PROMOTE to a headline capability claim if, on ≥2 of 3 frozen arms and across BOTH samplers (gd, ngd):
vanilla FID-vs-NFE turns up (degrades ≥0.5 FID past its min) while ANM stays monotone/flat to ≥2×
vanilla's optimal NFE, AND ANM reaches strictly lower equilibrium `‖f‖`. Dose-ordering (λ=0.3 ≥
λ=0.1 ≥ vanilla) required as the mechanism control.

## Kill rule
KILL the inference-scaling claim (do NOT retune indefinitely — CLAUDE.md 1-retune cap) if the
re-analysis shows no crossover AND the one NFE-extension run shows ANM and vanilla turning up
together. On kill: capability story falls back to Exp 2 (mechanism) + fixed-NFE FID only → stays
workshop-tier; record in postmortem before proposing the next capability probe (C2/C3).

## Anti-laundering note
"Test-time compute scaling" is only claimed if the **energy** descends further AND FID improves
further — not from FID alone. A FID-only improvement at high NFE without a matching `E`/`‖f‖` trace
would be reported as "better high-NFE samples," not "compute scaling."
