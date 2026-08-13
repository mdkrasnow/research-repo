# Corrected-BTM scalar potentials via finite differences — implementation plan

**Date**: 2026-08-13 · **Branch**: `btm-fd-scalar` · **Status**: executing

## Hypothesis (pre-registered)

The poor late-training behaviour of explicit scalar EqM (`--ebm direct`) is caused
substantially by the **mixed input–parameter derivative** `d_theta d_x phi` required by
direct gradient matching, rather than by the nonexistence of a useful scalar autonomous
transport potential. Training the *corrected conservative BTM population solution* using
only scalar energy evaluations plus finite differences should retain transport
correctness while avoiding that derivative during training.

This supersedes the FBGN optimizer-rescue track, which Stage 3A closed:
the certified Gauss–Newton direction is essentially orthogonal to the population gradient
(`C_V = 0.00108`) and is not a descent direction at all, so no step size and no damping can
rescue it (`memory/topics/wfb-eqm-fbgn-stage3.md`). FBGN is now **closure only**.

## Theory the implementation respects

Primary reference **arXiv:2608.01692v2**, *Beckmann Transport Models: From Autonomous
Flows to One-Step Maps* (Lee, Coeurdoux, Potaptchik, Du, Albergo, Vanden-Eijnden).
Facts recovered from the v2 full text and hard-coded into the module:

- `div(nu b) = mu0 - mu1` with `nu = int_0^1 mu_t dt` (Theorem 1) — verified verbatim.
- §2.7: EqM's loss (eq. 16) pairs the **linear** interpolant with the schedule-scaled
  target `c_t (x1 - x0)`, which is `E[Idot | I_t]` for **no** interpolant; it coincides
  with the consistent loss exactly when `c_t == 1`. This is the inconsistency under test.
- Appendix H (eq. 57) self-stopping interpolant — transcribed exactly:
  `alpha_t = 1 - 2t/(1+tc)` for `t <= tc`, `(1-t)^2/(1-tc^2)` for `t > tc`.
  At the breakpoint both branches give `alpha = (1-tc)/(1+tc)` and
  `alpha_dot = -2/(1+tc)`, so the interpolant is `C^1`; `Idot_1 = 0`.
- **`tc` is NOT numerically specified anywhere in the paper** (checked the full v2 text,
  including Appendices H and K.2). It is therefore treated as a nuisance hyperparameter
  and swept on the toy over `{0.5, 0.7, 0.8, 0.9}`, then frozen before the image stage.
- Five-atom benchmark: `mu0 = N(0, I_2)`, `p = (0.30, 0.30, 0.15, 0.15, 0.10)`, EqM
  schedule `c_t = (1-t)^0.8`. Reference results: **BTM MAE 0.005, EqM MAE 0.102**.
  Atom coordinates are *not* published — we use five points on a circle of radius 3 and
  label this a faithful benchmark **variant**, not an exact reproduction.

## Arms

| code | name | trains | mixed derivative? |
|---|---|---|---|
| V | `btm_vector` | `(1/2d) E|b_theta(I_t) - Idot|^2` | no (unconstrained field) |
| G | `btm_scalar_exact` | `(1/2d) E|grad phi - Idot|^2` | **yes** |
| A | `btm_scalar_action_exact` | `(1/2d)E_nu|grad phi|^2 + (E_mu0 phi - E_mu1 phi)/d` | **yes** |
| D | `btm_scalar_fd_directional` | `(1/2) E [D_{h,u} phi - u^T Idot]^2` | **no** |
| F | `btm_scalar_fd_action` | `(1/2) E [D_{h,u} phi]^2 + (E_mu0 phi - E_mu1 phi)/d` | **no** |
| — | `eqm_legacy_{vector,scalar}` | old EqM target, negative control | vector: no / scalar: yes |

G and A share a population optimum but have different minibatch estimators; D → G and
F → A as `h -> 0` because `E_u[(u^T e)^2] = |e|^2/d` for normalized Rademacher `u`
(which is why the FD losses carry no explicit factor `d`).

`E = -phi`, so following `b = grad phi` is gradient **descent** on `E`. `E` is a transport
potential, **not** a calibrated `-log p_data`.

## Guarantees enforced mechanically

`fd.assert_no_double_backward()` patches `torch.autograd.grad` / `backward` to raise on
`create_graph=True`; both FD arms run their forward *and* backward inside it. The test
suite includes a positive control (Arm G must trip the guard), so the guard cannot pass
vacuously.

## Gate before image compute (pre-registered)

All six must hold for Arm D:
1. FD estimator numerically validated (calibration ladder, plateau chosen not minimum `h`);
2. training stable across seeds;
3. median basin-mass MAE close to corrected vector BTM;
4. decisively beats the inconsistent-EqM negative control;
5. no mixed input–parameter derivatives during training (guard, enforced);
6. the exact evaluation-time gradient field actually drives samples to the right modes.

Threshold: `MAE_D <= max(0.015, 2 * MAE_V)`.

## Files

```
experiments/btm/interpolant.py   BTM Appendix-H + linear + legacy-EqM target
experiments/btm/fd.py            central-difference estimator + autograd guard
experiments/btm/objectives.py    the five arms
experiments/btm/models.py        toy scalar/vector MLPs (matched param count)
experiments/btm/calibrate.py     eps ladder + plateau rule
experiments/btm/gradnoise.py     parameter-gradient variance per arm
experiments/btm/toy5.py          five-atom training + basin mass + weak residual
experiments/btm/toy_campaign.py  staged driver (calibration, tc sweep, main, grid, noise)
tests/test_btm_math.py           29 tests; hard gate before any cluster job
slurm/jobs/btm_toy_campaign.sbatch
```

## Implementation defects found and fixed during construction

- `directional_fd(fp32_subtract=True)` originally **downcast** float64 evaluations to
  float32, inflating the FD error ~20x at `eps=1e-3` and destroying the `O(h^2)`
  convergence signature. It now only *promotes* bf16/fp16 and never demotes fp64.
  Caught by the second-order-convergence unit test, not by inspection.
