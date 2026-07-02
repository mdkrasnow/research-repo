# Pre-registration — RC-ANM iteration-2 (functional fix), CPU

Written 2026-06-13 BEFORE any v2 code runs. Successor to RC-ANM R4
(postmortem-rcanm-r4.md): concept VALIDATED by dose-matched oracle (oracle42
MSE 0.877 vs random42 1.285 at equal dose), but the implemented teacher-descent
flip-basin functional was degenerate on the weak 1000-step teacher (certified
eps=1.2, accepted 80%, damaged). v2 fixes ONLY the functional + teacher
snapshot; method, machinery, arms otherwise unchanged from preregistration-
rc-anm.md.

## Changes (the only deltas vs v1)
1. STRONGER TEACHER: snapshot the EMA at step 2500 (was 1000). Mining/training
   window = steps 2500-4000 (1500 steps), matching the dose-check teacher that
   made oracle_safe work.
2. BASIN PROXY (replaces 100-step descent flip-basin): SHORT teacher rollout +
   nearest-mode, validated against the analytic oracle BEFORE use.
   proxy_basin(x_t_adv) := nearest-mode( x_t_adv + sum_{k<R} eta * g(.) ),
   R = 15 steps, eta = 0.1 (short, low descent-noise). The certified functional
   is the FLIP form: r = 1 iff proxy_basin(x_t_adv) != lab AND
   proxy_basin(x_t_orig) == lab (mining-induced only; D6 lesson).
3. PROXY VALIDATION GATE (new, must pass before calibration is trusted):
   on labeled toy data with the step-2500 teacher, the proxy's accept decision
   (proxy_basin(x_t_adv)==lab) must AGREE with the analytic oracle decision
   (voronoi_basin(x_t_adv)==lab) on >= 70% of mined endpoints (balanced acc
   >= 0.65) at eps_ball 0.8, manifold-adjacent t. If validation fails, the
   teacher-based proxy is declared uninstrumented at toy scale -> R4 stands,
   escalate to CIFAR (richer teacher); do NOT tune the proxy to pass.

## Everything else inherited from preregistration-rc-anm.md
alpha=0.10, delta_r=0.05, m=250, EPS_GRID, gamma bins, 5 seeds, arms
(vanilla/fixed_anm/aggressive_anm/rc_anm/oracle_safe_anm), primary metric field
MSE vs MC reference + coverage, margins 2x vanilla seed-SD.

## Gate G-RCANM-v2 (same letters as v1)
(a) PREMISE: aggressive_anm worse than vanilla by >= 2x margin.
(b) SAFETY: rc_anm >= vanilla - margin (no net damage).
(c) UTILITY-PARITY: rc_anm <= fixed_anm + margin on MSE AND coverage >=
    fixed - cov_margin.
(d) CERT REALIZED: rc_anm realized proxy-flip-risk <= alpha.
PLUS proxy-validation gate (>= 0.65 balanced acc vs oracle).

Branches:
  R1 BEST: proxy-valid AND (a)(b)(c)(d) -> RC-ANM works at toy -> CIFAR mini.
  R2 SAFETY-ONLY: proxy-valid AND (a)(b)(d), (c) fails -> safety wrapper w/ cost.
  R2' CONCEPT-NOT-INSTRUMENTABLE: proxy-validation FAILS -> teacher proxy can't
     track oracle at toy scale; the channel exists (oracle) but the CPU proxy
     does not; escalate to CIFAR where the teacher is richer. (distinct from
     R4: not "broken", but "wrong instrument for the proxy".)
  R4 BROKEN: proxy-valid but (b) fails -> deeper bug -> postmortem.
Any no-branch -> STOP + postmortem.

## Discipline
No GPU. CIFAR mini only after R1 + human approval. rc_hpm/core.py + eqm2d
reused. v1 arms remain the comparison; v2 changes teacher snapshot + proxy only.
