# Pre-registration — RC-ANM (Risk-Controlled Adversarial Negative Mining), CPU

Written 2026-06-13 BEFORE any RC-ANM code runs. Mainline pivot per
documentation/pivot-rc-anm.md. Reuses rc_hpm/core.py (LTT/HB, validated G-1/G0)
and rc_hpm/eqm2d.py (Field, pgd_mine, basin_certify, get_ct, make_triplet,
voronoi_basin, reference_field, metrics). Inherits P-rules + A1-A4. Object
certified = the MINED ENDPOINT (not a contrastive pair).

## Constants
2D EqM toy: 8-mode unequal mixture (eqm2d MEANS/WEIGHTS), sigma=0.10, exact
get_ct. Field MLP 2->128->128->2 noise-unconditional. Training 4000 steps,
bs 128, Adam 1e-3. EMA teacher decay 0.999, snapshot frozen at step 1000
(calibration time). PGD: 3 ascent steps, rel_step 0.15. alpha=0.10, delta_r
0.05, m=250, m_fit=30. gamma bins eqm2d GAMMA_BINS; mining restricted to bins
where the teacher field is informative (t>=2/3 manifold-adjacent, MINE_BINS
spirit). 5 seeds.

## Teacher safety scores (pinned; each in [0,1], higher = MORE dangerous)
Computed on the FROZEN EMA teacher g, per candidate endpoint eps_adv at (x1, t):
- r_basin: descend frozen teacher field from x_t_adv to convergence; r_basin=1
  if attractor Voronoi-basin != source class, else 0. (the v10 failure mode)
- r_field: 1 - cos(g(x_t_adv), g(x_t_orig)) clipped to [0,1] /2 -> normalized
  teacher-field disagreement induced by mining.
- r_target: cos angle between mined target (x1-eps_adv) and clean (x1-eps);
  r_target = (1 - cos)/2 -> target direction corruption.
- r_inflate: residual ratio res(x_t_adv)/res(x_t_orig); r_inflate =
  clip((ratio - INFL_OK)/(INFL_MAX - INFL_OK), 0, 1), INFL_OK=3, INFL_MAX=20.
- r_return: short teacher GD trajectory (20 steps) from x_t_adv; r_return=1 if
  final distance-to-nearest-mode > initial (diverges), else fraction not
  contracted.
PINNED AGGREGATE (declared now, never tuned): the controlled risk functional
for calibration is **r_basin** (ground-truth-anchored, the load-bearing safety
property: a mined endpoint that flows to the wrong mode poisons training). The
other four are REPORTED diagnostics (descriptive), used by the accept/reject
arm as a composite r_soft = mean(r_field, r_target, r_inflate, r_return) only
as a NUISANCE selection knob (eligible for the single retune), NOT as the
certified functional. This keeps the guarantee grounded on a true label
(basin), exactly as toy-validated in D2; r_basin's labeler is exact in 2D
(analytic Voronoi), so eta=0 (no labeler-error inflation).

## Calibration (reuse LTT/HB)
Per gamma-bin, fixed-sequence over an eps_ball grid {0.1, 0.25, 0.5, 0.8, 1.2}:
risk = mean r_basin among ACCEPTED mined endpoints at that eps_ball (manifold-
adjacent t only). Certify the LARGEST eps_ball with HB p-value <= delta_r at
alpha. Adaptive-eps RC-ANM uses the per-bin certified eps_ball; accept/reject
RC-ANM uses the smallest-eps certified set with an r_soft threshold. ABORT
(no certified eps_ball) -> train un-mined (P7).

## Arms (5 seeds)
1. vanilla            no mining (baseline / negative-control floor).
2. fixed_anm          v10-style: fixed eps_ball=0.5, uncertified.
3. aggressive_anm     eps_ball=1.5, uncertified (DAMAGE control, must hurt).
4. rc_anm             per-gamma largest certified eps_ball (adaptive-eps) +
                      r_basin accept/reject on each mined endpoint (reject ->
                      un-mined endpoint).
5. oracle_safe_anm    accept only endpoints with TRUE basin == source (the
                      certification CEILING; toy only).

## Primary metrics (P3)
field MSE vs MC reference field (eqm2d.reference_field), 40x40 grid, gamma
slices {0.3,0.6,0.9} (PRIMARY). mode_coverage + recall_distance descriptive.
Margins = 2x vanilla seed-SD per metric.

## Pre-registered gate (G-RCANM)
(a) PREMISE: aggressive_anm WORSE than vanilla by >= 2x margin on field MSE.
    (uncontrolled mining damages the field — the thing RC-ANM bounds.)
(b) SAFETY: rc_anm NOT worse than vanilla by more than the margin
    (>= vanilla_MSE - margin; certification avoids the aggressive damage).
(c) UTILITY-PARITY: rc_anm <= fixed_anm + margin on field MSE
    (RC-ANM matches or beats fixed ANM) AND rc_anm mode_coverage >=
    fixed_anm coverage - margin.
(d) CERTIFICATION REALIZED: rc_anm realized r_basin <= alpha on fresh batches.
Branches:
  R1 BEST: (a)(b)(c)(d) all hold -> RC-ANM works at toy scale -> CIFAR mini
     (vanilla vs fixed/v10 ANM vs RC-ANM), then human-gated GPU for B/2.
  R2 SAFETY-ONLY: (a)(b)(d) hold but (c) fails (RC-ANM safe but loses utility
     vs fixed ANM) -> RC-ANM is a safety wrapper with a cost; report the
     validity-power tradeoff (spec E9 analog).
  R3 NULL: (a) fails (aggressive ANM does NOT damage at toy scale) -> the toy
     cannot exercise the premise; escalate (the 2D recall metric was noisy in
     E1.2) -> redesign toy or move premise test to CIFAR mini.
  R4 BROKEN: (b) fails (RC-ANM damages despite certification) -> calibration
     bug or r_basin is not the right functional -> STOP + postmortem.
Any no-branch result -> STOP + postmortem, no improvised branch.

## Reuse / archive discipline
rc_hpm/core.py + eqm2d ANM functions reused unchanged. Contrastive arms
(arm A, D1/D2/D3) ARCHIVED as negative baselines — no further compute. No GPU
on the contrastive path. CIFAR mini for RC-ANM is CPU-small first; B/2 GPU only
after R1 + explicit human approval (FID never auto, per CLAUDE.md).
