---
name: wfb-eqm-fbgn-stage3
description: Stage 3 of the fb_direct scalar-energy optimization track — certified CG-FBGN reduces every one of its own minibatches yet worsens the held-out EqM field objective; audit, retraction, and the H1-vs-H2 discriminator
status: active
---

# WFB-EqM Stage 3: FBGN optimizes each minibatch but worsens the population field

Thread: `projects/masked-EqM/`, secondary track (fb_direct scalar-energy training
stability), not the primary Structured Start-State EqM question.

## Setting

EqM `--ebm direct` trains a scalar energy `E_theta(z)` with field
`s_theta(z) = -grad_z E_theta(z)` against the EqM target, squared error.
`M = d(grad_z E)/d theta` is the mixed input-parameter Jacobian; ordinary
training uses `g = M^T r`, `r = s_theta(z) - y`.

Stage 1 established the mixed-Jacobian amplification is real (raw gradient
spike/control 2.44x -> 0.95x under whitening). Stage 2/2.5 moved from the
square-root whitening `M^T(MM^T+lambda I)^{-1/2} r` to full Gauss-Newton
`p = -M^T(MM^T+lambda I)^{-1} r`, and a fixed-k Lanczos truncation bug was
replaced with residual-controlled CG.

## The 300-step result (2026-08-13) — all arms negative

No-Adam Armijo-backtracking trainer, 300 steps, batch 8, same start checkpoint
and seed. Deterministic held-out probe:

| arm | job | probe start -> end | delta | median R = actual/pred | median eta | own-batch residual removed |
|---|---|---|---|---|---|---|
| direct (alpha=0) | 38904393 | 10.6244 -> 10.8324 | +0.208 | 0.991 | 3.8e-6 | 4.5 % |
| WFB (alpha=0.5) | 38903397 | 10.6244 -> 11.2145 | +0.590 | 0.911 | 2.5e-3 | 13.6 % |
| FBGN (alpha=1, CG) | 38909657 | 10.6245 -> 14.8811 | +4.257 | 0.528 | 1.19 | 16.4 % |

All at accept_rate 1.000 with zero skips. **The raw-direct negative control also
worsens the probe** — the failure runs through the whole alpha family, ordered
exactly as step aggressiveness is ordered. FBGN is its extreme point, not a
special defect.

LM adaptive damping (job 38944527, 20-step smoke) did not help: rho moved only
1e-4 -> 2e-4, probe +1.99 in 20 steps, median R fell to 0.453.

## Two things that are NOT true (durable corrections)

1. **"lambda_max growth caused the divergence" is RETRACTED.** FBGN's probe
   damage is ~78 % complete by step 25 and then a flat noisy band, while
   lambda_max grew for 275 more steps — timing decoupled. And damping was set as
   `lambda = rho * lambda_max`, so under `M -> cM` the operator
   `A(A+lambda I)^{-1}` is **invariant by construction**: uniform curvature
   growth cancels and cannot change the step. The lambda_max sequence and the
   late `L_before` spikes (40.7, 130.6) were also measured on different fresh
   stochastic batches — heavy-tailed per-batch draws, not a parameter-space
   excursion. (The backbone-vs-head localization, backbone sigma_1 1.53x vs head
   0.95x, is unaffected and still stands.)

2. **The logged reduction ratio cannot decide the mechanism.** `rho_lm` is
   exactly the LM reduction ratio and separates the arms cleanly, but it is
   confounded with step length: FBGN takes `eta ~ 1.19` full GN steps, direct
   takes `eta ~ 3.8e-6`. Any smooth model is accurate as `eta -> 0`, so a low R
   for the arm taking far longer steps is what BOTH hypotheses predict.

## The open question and how it gets decided

- **H1** Gauss-Newton local-model failure (FBGN drops `sum_i r_i grad^2 r_i`)
  -> repair is genuine LM damping / shorter steps.
- **H2** stochastic minibatch over-solving: the local model is *right* and the
  minibatch is *wrong* -> repair is a larger GN model batch and/or independent
  acceptance; damping cannot help.

Decided by `experiments/direct_energy/wfb_stage3a_mechanism_discriminator.py`
(frozen checkpoints, no training, apply/evaluate/revert with bitwise restoration
asserted), which closes both gaps: an eta-scan at FIXED direction, and the
**eta-independent** infinitesimal transfer `d_V = grad L_V . p` on an
independent trust bank. `d_V >= 0` decides H2 outright — by first-order Taylor,
no step size and no damping can rescue an ascent direction.

## Stage 3A verdict (job 38985448, 2026-08-13) — H2 is causal

Answered at `start`, the healthy checkpoint FBGN training actually began from
(and ONLY there — see the gotcha about fbgn100/300 below):

| quantity | value |
|---|---|
| FBGN cosine with independent-batch gradient `C_V` | **0.00108** |
| raw-direct minibatch gradient `C_V` | 0.0395 |
| FBGN `d_V` descent rate (eta-INDEPENDENT) | **4/8** |
| trust bank worsened at `eta*/8` | **8/8**, +0.0629, paired SE 0.013-0.020 |
| `R_B` / `D_B`, `eta*` -> `eta*/8` | -10.685 -> +0.910 / 3.397 -> 0.459 |
| damping x10 / x100 -> `C_V` | -0.0031 / -0.0083 |

- **H1 is present but is a trust-region artifact.** `D_B` falls as O(eta) and
  `R_B` -> 1, which is what a CORRECT linearization does; `eta* ~ 1.3` was simply
  far outside the trust region. Nothing is wrong with the GN model itself.
- **H2 is what survives eta -> 0**, and it decided the question: the direction is
  not a descent direction for the population objective at all, so by first-order
  Taylor no step size and no damping can rescue it.
- **The pre-registered H1 repair failed its own falsifier.** Damping fixed the
  local model and simultaneously killed independent-batch descent.

Mechanism: at a converged checkpoint the B=8 GN system carries almost no
population signal, and `(A + lam I)^{-1}` whitening amplifies precisely the
small-curvature directions that only those 8 images constrain — a further 36x
alignment loss. **FBGN removes 66-74% of its own batch's residual while retaining
0.1% of the population gradient direction.** Certification makes this worse, not
better. That is the whole reconciliation of "100% Armijo acceptance + monotone
same-batch reduction" with "monotone probe damage": both true, about different
objectives.

Report: `projects/masked-EqM/documentation/wfb-eqm-stage3a-report-2026-08-13.md`.
Open branch: Stage 3B (job 39024136) tests whether `C(B) = cos(g_B, g_ref)` rises
faster than the sqrt(B) of pure noise-averaging. ~0.5 or below fires the
pre-registered KILL and the thread is written up as a negative result.

## Load-bearing gotchas

- **The probe is deterministic** — `build_pool` freezes `(xt, t, y, ut)` once
  (images, labels, VAE latents, `t`, Gaussian eps, corrupted latent, target) and
  the model stays in `.eval()`. Two separate jobs agree to 16 s.f. BUT the probe
  slice is `pool[max_steps : max_steps+8]`, so it **moves when `--max-steps`
  changes**: cross-run probe comparisons are only valid at equal `--max-steps`.
- **The FP64 CPU toy is linear in theta to machine precision at default init**
  (EqM/DiT adaLN-zero gates + zeroed output projection). Any test of
  theta-curvature, linearization defect, or second-order behaviour MUST call
  `perturb(model, std=0.02)` first, or it silently measures the 6.6e-16 roundoff
  floor and passes on nothing.
- `exact_field_vjp` negates the WHOLE accumulated `.grad` buffer, so a
  multi-batch gradient must harvest and sum per batch; naive accumulation across
  calls double-negates earlier batches.
- **Never read the FBGN mechanism off `fbgn100`/`fbgn300`.** `||g_V||` is
  74.6/192.2 there (vs 1.204 at `start`) and the raw-direct `C_V` is 0.98/0.88 —
  the model is broken enough that almost anything descends, so their apparent
  H1 signature measures distance-from-optimum, not mechanism. Only the checkpoint
  a failure BEGAN from can explain it.
- **A large reduction ratio `R_B` at a large step is not evidence of anything**,
  and neither is 100% Armijo acceptance: at `eta*` here `R_B = -10.685` while
  Armijo accepted every step for 300 steps.
- **CG's own reported residual understated the true one ~1.9x** (0.0086 vs
  0.0159). Always recompute `||r - (A + lam I)u|| / ||r||` from a fresh operator
  application before calling a direction certified.

Full write-up: `projects/masked-EqM/documentation/wfb-eqm-stage3-audit-2026-08-13.md`.
Superseded interpretation: `documentation/wfb-eqm-stage3-postmortem-2026-08-13.md`.
