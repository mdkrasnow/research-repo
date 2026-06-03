# C1 — Inference-Compute Scaling: findings

Pre-registered: `c1-inference-compute-scaling-proposal.md`. Script:
`experiments/c1_inference_scaling/reanalyze_exp1.py` (pure pandas, zero compute).
Input: `documentation/exp1_data/results.csv` (Exp 1 80-cell 5K sampler sweep, merged
job 18369465). Arms: vanilla (FID 31.41 @50K) vs anm = v10 λ=0.1 (29.01 @50K).
NFE ∈ {10,25,50,100,250}; step_mult ∈ {0.5,1,1.5,2}; samplers gd+ngd.

## Minimal-test result (existing data — partial, 2 arms, nfe≤250)

**POSITIVE — uniform quality gain (strengthens headline, not a new capability):**
- ANM beats vanilla on **18/18 converged cells** (both arms FID<100). 0 cells vanilla-better.
- Mean delta (anm−vanilla) = **−2.055 FID**, median −1.93. Consistent across BOTH samplers
  and EVERY nfe/step setting in the converged band.
- Interpretation: the ~2 FID gain is **robust across the whole sampler operating grid**, not an
  artifact of one tuned NFE/step. Good — but it is a *better-everywhere quality* result, the same
  class as the fixed-NFE FID. Does not by itself clear workshop→main.

**NEGATIVE — the inference-compute-scaling SIGNATURE is absent at nfe≤250 / step≤2.0:**
- `corr(step_mult, delta)` on nfe≥100 cells = **−0.037 ≈ 0**. ANM's edge does **not** grow as steps
  get more aggressive. (An earlier eyeball of nfe=100 cells, −2.10→−2.45, was cherry-picked; across
  the full converged band the edge is flat ~−2 FID.)
- Overshoot turn-up along the step_mult axis: vanilla mean rise-after-min **+0.29 FID**, ANM **+0.34
  FID**. ANM turns up *slightly more*, not less. **No crossover** where vanilla degrades while ANM
  stays monotone.
- The nfe=100 vanilla micro-U-turn (48.3→46.9→47.0, +0.08) is real but negligible; at nfe=250 both
  arms rise ~equally (vanilla +0.72, anm +0.64/+0.72) under bigger steps.

## Decision (pre-registered)

Promotion rule (proposal) NOT met: requires vanilla turn-up ≥0.5 past min while ANM monotone to
≥2× nfe, dose-ordering, lower residual. Existing data shows ~equal turn-up and flat corr.

BUT the kill rule requires BOTH (no crossover in re-analysis AND extension confirms turning-up
together). The existing grid never reaches the regime the hypothesis is actually about — **many
optimization steps past convergence** (nfe ≫ 250 / equilibrium refinement). The converged band here
is narrow (effective compute ≤500); high-NFE refinement is untested. Also residual `‖f‖` is missing
for vanilla in the recovered CSV, so the energy-descent half of the mechanism is unmeasured.

→ **Run ONE C1 extension** (the pre-registered decider), then promote-or-kill:
- NFE up to {400, 600, 1000} at step_mult 1.0 (the untested high-compute regime).
- step_mult up to {2.5, 3.0} at nfe 250 (push the overshoot axis past where it's flat).
- Add dose arm **v10 λ=0.3** (FID 27.09) → enables the dose-ordering control.
- Log per-step energy surrogate `‖f(xt)‖` (equilibrium residual) for BOTH arms → test whether ANM
  reaches lower residual / keeps descending where vanilla's bottoms out and climbs.
- gd + ngd, 5K samples (deltas only — absolute 5K FID inflated per Exp 1).

If at nfe≥600 vanilla FID turns up ≥0.5 past its min while ANM stays flat/monotone AND ANM residual
ends lower, dose-ordered → **promote** to a compute-scaling capability claim. If both flat / turn up
together → **kill** the capability claim; keep the uniform −2 FID gain as a supporting result and
move differentiation weight to C3 (OOD) / C2 (restoration).

## Honest framing note
As of the zero-compute analysis, C1 leans toward "uniform quality gain, NOT a scaling capability."
The extension is a genuine test of an untested regime, not confirmation of a visible trend. Report it
that way; do not launder the uniform gain into a scaling story without the high-NFE crossover.
