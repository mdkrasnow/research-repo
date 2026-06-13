# Pre-registration — RC-ANM Step 1: premise probe on a real EqM checkpoint (GPU, inference-only)

Written 2026-06-13 BEFORE the probe runs. Successor to RC-ANM R2' (toy teacher
too collapsed to instrument basin-safety; postmortem-rcanm-r4.md). Step 1 tests
the LOAD-BEARING premise on the target domain — the move the toy could not make
(R2') and the G1.5/linchpin discipline demands BEFORE any certified-training
GPU spend.

## Question
Does v10/ANM mining on a TRAINED EqM model actually produce UNSAFE endpoints
(wrong-basin / high teacher-disagreement) that a certification layer could
filter — and do those unsafe endpoints carry systematically worse training
signal? If v10's fixed eps_ball already mines ~no unsafe endpoints, RC-ANM
certifies a non-problem (D2 lesson: don't certify where there is no damage).

## Setup (inference-only; NO training)
- Checkpoint: trusted diff-EqM EqM-B/2 IN-1K 80ep. Use the EMA weights as the
  frozen teacher g AND as the mined model (single-checkpoint probe; the teacher
  IS the model at this snapshot, which is the RC-ANM deployment assumption).
  Prefer the v10 checkpoint if available (its eps_ball is the one we audit);
  else vanilla baseline (FID 31.41) as the mining target.
- Data: a few thousand IN-1K latents (VAE-encoded), batched; gamma ~ U(0,1) per
  the EqM training law; eps ~ N(0,I) endpoints.
- Mining: the existing transport.mine_negative / v10 PGD path (K=3, the v10
  eps_ball), plus an eps_ball SWEEP {0.25, 0.5 (v10), 1.0, 1.5} to map
  unsafe-fraction vs aggressiveness.

## Safety scores per mined endpoint (EqM-native; richer teacher = instrumentable)
- r_basin (proxy): short teacher-field rollout from x_t_adv; "unsafe" if it does
  NOT contract toward the same region as the un-mined x_t (flip form, D6). At
  IN-1K there is no class-label Voronoi oracle, so r_basin uses the teacher's
  own attractor consistency (mined vs un-mined endpoint converge to the same
  neighborhood under the teacher field). Reported with its limitation.
- r_field: 1 - cos(g(x_t_adv), g(x_t_orig)) — teacher-field disagreement.
- r_target: target-direction corruption (x1-eps_adv) vs (x1-eps).
- r_inflate: residual ratio res(x_t_adv)/res(x_t_orig).
- r_return: trajectory contraction toward the data manifold (||f|| shrinks).

## Measurements (descriptive + the two pre-registered gates)
M1 unsafe fraction: per eps_ball, fraction of mined endpoints with each score
   above a high percentile of its UN-MINED baseline (the score's null).
M2 gradient impact: for a random subset, compute the EqM training gradient
   contribution of safe vs unsafe mined endpoints (cosine to the clean-batch
   gradient; magnitude). Unsafe endpoints expected to have lower/negative
   alignment if they poison training.
M3 concentration: is unsafe-mass concentrated (E1.0 14.3x analog) at high gamma
   / high residual, or diffuse?

## Pre-registered gates -> branch
S1-PREMISE: at the v10 eps_ball (0.5), unsafe fraction (r_field or r_target
   above the 95th-percentile of the un-mined null) >= 5%. (a non-trivial set to
   filter.)
S1-IMPACT: unsafe mined endpoints have gradient cosine-to-clean-batch
   significantly lower than safe mined endpoints (Welch p<0.05, >=2 scores
   agree on the ordering).
Branches:
  P1 PREMISE-HOLDS: S1-PREMISE and S1-IMPACT both hold -> v10 mines a
     meaningful, training-relevant unsafe set -> proceed to Step 2 (CIFAR-mini
     certified-training RC-ANM ladder, GPU). The certification has a real target.
  P2 SAFE-AGGRESSIVE: S1-PREMISE fails at eps=0.5 but the eps_ball SWEEP shows
     unsafe fraction crosses >=5% at larger eps with damaging gradients ->
     v10's eps is conservative; RC-ANM's story is "certify -> mine MORE
     aggressively safely" (unlock headroom), not "fix v10". Proceed to Step 2
     with the aggressive arm as the treatment.
  P3 NON-PROBLEM: unsafe fraction < 5% at ALL eps with no gradient impact ->
     v10 mining is already safe; certification bounds nothing at scale. RC-ANM
     utility story collapses; the contribution narrows to the toy-scale safety
     guarantee + the methodology. STOP before Step-2 GPU; write up honestly.
Any no-branch -> STOP + postmortem.

## Compute / discipline
Single GPU, inference-only, minutes-to-1h. Job tracked in pipeline.json
active_runs per CLAUDE.md. No training, no FID. Step 2 (CIFAR-mini training +
FID) only after P1/P2 + the run is logged. rc_hpm/core.py + the diff-EqM ANM
code path reused; nothing in diff-EqM modified (read-only checkpoint use).
