# RC-ANM Step-1 verdict — P3 NON-PROBLEM (2026-06-13, GPU)

Job 22507558 (gpu_test, EqM-B/2 v10 IN-1K ckpt, inference-only, 40 batches x
32, eps sweep {0.25,0.5,1.0,1.5}, PGD 12x0.25). Full:
results/rcanm_step1_verdict.json. Pre-registration: preregistration-rcanm-step1.md.

## Result
| eps_ball | disp | unsafe_frac (field) | unsafe_frac (target) | r_inflate |
|---|---|---|---|---|
| 0.25 | 0.25 | 0.05 | 0.05 | 3.1 |
| 0.50 (v10) | 0.50 | 0.26 | 0.96 | 5.4 |
| 1.00 | 1.00 | 0.50 | 1.00 | 9.2 |
| 1.50 | 1.46 | 0.64 | 1.00 | 12.3 |

S1-PREMISE: TRUE — mining produces "unsafe-looking" endpoints, rising with eps.
S1-IMPACT: **FALSE** — gradient-impact (n=256): unsafe endpoints' training-
  gradient cosine-to-clean-batch = 0.047 vs safe 0.064, Welch p=0.40. Even at
  the most aggressive eps (64% unsafe, 12x inflation), unsafe endpoints do NOT
  carry worse training gradients.
BRANCH: **P3 NON-PROBLEM — certification bounds nothing at scale. STOP before
Step-2 training.**

## Why (the structural finding — this is the contribution-grade insight)
EqM's training target is (x1 - eps_adv) * c(gamma): it ALWAYS references the
real data point x1. Adversarial endpoint mining moves the NOISE endpoint eps,
making a harder regression example, but the gradient still teaches "predict the
direction toward the real x1." You cannot poison a regression-to-real-data
target by mining the noise endpoint — the target's data-direction is anchored
to x1, not to the endpoint's basin. r_target "corruption" (96-100%) measures
the angle of (x1-eps_adv) vs (x1-eps); both still point from a noise point to
the SAME real x1, so the learning signal is preserved.

Corollary: this is EXACTLY why v10/ANM helps EqM WITHOUT any certification
(diff-EqM FID 27.58 vs 31.41) — the geometry is intrinsically robust to
endpoint mining. Risk-control protects against a damage mode EqM's target
geometry already prevents.

Contrast (D2 contrastive): contrastive learning has NO fixed real anchor —
mining there DOES damage (probe 0.27 vs 0.95), because pushing/pulling mined
PAIRS corrupts the relative geometry that is the only signal. RC-HPM's
certification had a real target there (damage existed) but no utility channel;
RC-ANM has a utility channel (v10 works) but no damage to certify.

## Convergent conclusion across three scales
The "risk-control a mining method" thesis finds no problem to bound wherever
the base method actually works:
- D2 contrastive: hardness harms, but certification recovers no utility.
- RC-ANM 2D toy: oracle concept works, but not teacher-instrumentable (R2').
- RC-ANM scale (here): v10 works, and its mined endpoints don't poison the
  gradient -> nothing to certify (P3).
The cheap premise probe (G1.5/linchpin discipline) killed the Step-2 training
ladder spend BEFORE it happened. System worked as designed.

## What survives
1. STRUCTURAL INSIGHT (publishable): EqM's regression-to-real-data target is
   intrinsically robust to adversarial endpoint mining; contrastive objectives
   are not. Explains why ANM helps EqM uncertified and why certification adds
   nothing. A clean "when does adversarial mining need risk control?" result:
   only when the objective lacks a fixed real anchor.
2. The validated machinery (LTT/HB certified mining) + the contrastive-side
   safety story (D2: certify-negatives-only prevents the real damage there).
3. RC-ANM mainline (the "certified v10" utility thesis) is RETIRED at scale by
   premise. No Step-2 GPU.

## Decision (pre-registered P3)
STOP. No CIFAR/IN-1K RC-ANM training ladder (it would certify a non-problem,
the D2 lesson). Write up the structural insight + the negative. Any further GPU
only for a DIFFERENT question, human-gated.
