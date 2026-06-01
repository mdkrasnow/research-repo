# Exp 2 — Off-trajectory field robustness: results (2026-06-01)

**Question.** Does v10 (ANM) make the EqM field more accurate in a local L2 tube
around the vanilla noise→data trajectory — i.e. is the FID gain backed by a
measurable field-robustness mechanism, not just better samples?

**Setup.** Frozen vanilla (FID 31.41) vs v10 (`final.pt`, FID 29.01) EqM-B/2 on
5120 held-out IN-1K val latents. EMA weights, fp32, matched recipe. Repo transport
target `(x1−x0)·c(t)` (verified == training `_v10_pgd_hard_example_step`; radius-0
cosine +0.76 both ckpts → sign correct). Paired ANM−vanilla over identical
`(x1,x0,t,δ,label)`; 95% bootstrap CI over `sample_id` (n≈30k pairs/cell).
Jobs 17788287 (random, gpu_test, 1:28) + 17788329 (sampler, gpu, 1:43), both exit 0.

## Headline result — ANM is more field-robust off-trajectory (all SIG)

| perturbation | radius | dMSE (anm−van) | 95% CI | dCos |
|---|---|---|---|---|
| random_l2_orthogonal | 0 | −0.0164 | [−0.0191, −0.0138] | +0.00051 |
| random_l2_orthogonal | 0.02 | −0.0165 | SIG | +0.00050 |
| random_l2_orthogonal | 0.05 | −0.0192 | [−0.0220, −0.0163] | +0.00053 |
| random_l2_orthogonal | 0.10 | −0.0277 | [−0.0314, −0.0240] | +0.00060 |
| **sampler_endpoint_mined** (real v10 δ) | nat (‖δ‖=0.3, rel≈0.0035) | **−0.0368** | **[−0.0396, −0.0340]** | **+0.00109** |
| sampler_local_drift | nat | −0.0194 | [−0.0221, −0.0167] | +0.00055 |

Negative dMSE = ANM lower error = better. Positive dCos = ANM better aligned = better.

**Three findings, all in the predicted direction:**
1. **ANM beats vanilla at every radius**, every perturbation type — all significant
   (tight CIs, n≈30k pairs).
2. **Gap widens with radius**: random dMSE −0.0164 (r0) → −0.0277 (r0.1), dCos
   +0.00051 → +0.00060. ANM's field degrades *slower* moving off-trajectory.
3. **Largest gap at the actual mined perturbation** (−0.0368, ≈2.2× the r0 random
   gap; dCos +0.00109, ≈2× random-r0). ANM is most accurate precisely in the
   off-trajectory region its training mining targets — the cleanest possible
   mechanism-confirmation.

**Determinism / fairness cross-check:** `random_l2_orthogonal` numbers are
bit-identical between the two independent SLURM jobs (17788287 vs 17788329) — same
seed → same result, confirming both models saw identical inputs.

## Honest caveats
- **Small effect size.** dMSE ≈ −0.016 to −0.037 against vanMSE ≈ 10.3 (0.16–0.36%
  relative); dCos ≈ 5e-4 to 1.1e-3. Significant (huge n, tight CI) but modest —
  consistent with the NULL B/2 capability eval: v10's benefit at B/2 is real but
  small. Effect may be larger at XL/2 (untested).
- **Not pure radius-0 parity.** There is already a small r0 gap (−0.0164). So ANM
  has a slightly better field even on-path, with an *additional* off-trajectory
  component. Off-traj-specific part = gap(r0.1) − gap(r0) ≈ −0.011 MSE; mined-specific
  ≈ −0.020 MSE. The robustness mechanism sits on top of a small on-path gain.
- **Local proxy only.** This tests local field accuracy around the vanilla path; it
  does NOT prove global energy correctness or sampling optimality.

## Interpretation for the paper
Supports the workshop story: *"ANM does not merely improve final FID — it makes the
learned EqM field measurably more accurate under local off-trajectory drift, with the
largest improvement exactly at the hard-mined perturbations ANM trains on."* Pair with
Exp 1 (NFE/sampler robustness) and the FID result. State the effect size honestly;
flag XL/2 as the scale where the mechanism may amplify.

## Artifacts
`projects/diff-EqM/results/diagnostics/offtraj_random/` and `offtraj_sampler/`:
`per_sample_metrics.jsonl`, `aggregate_metrics.csv`, `paired_differences.csv`,
`sanity/first_batch_checks.json`, `plots/{mse,cosine,rel_norm_err}_vs_radius.png`,
`plots/*__tbin_heatmap_*.png`, `plots/field_norm_hist_{radius0,offtraj}.png`.
Diagnostic: `experiments/diagnostics/offtraj_field_robustness.py` (README there).
