# Pre-registration — D3 Certification Curriculum (no hardness), CPU

Written 2026-06-13, BEFORE any D3 code runs. Successor to D2 branch B2
(RC-HPM hard-pair-mining utility retired; F15 decomposition: certification
safe, hardness harmful). Inherits all standing protocol: preregistration.md +
A1-A4, preregistration-d2.md grid/generator, decision-tree v2.1 P-rules,
deviations -> deviations.md, rc_hpm/core.py + rc_hpm/ladder.py reused not
rewritten, monitor units per-batch.

## Hypothesis
**H-CC:** Certified pairs improve representation learning when selected by
CONFIDENCE / DIVERSITY / CURRICULUM SCHEDULE, but NOT by maximum hardness.
The useful ingredient is certification, not hardness (D2 F15: hardest
certified pairs -6..-24 vs no_mine; cert_random_k ~= baseline; both >> naive).

Primary research question: can certification be turned into UTILITY, or is it
only a safety guardrail? Main comparison: **cert_curriculum vs cert_random_k
vs no_mine** (NOT vs rc_hard — that is a settled foil).

## Constants (inherited)
alpha for the band = 0.40 (P2 4*alpha_0; the budget where certified-negative
supply is non-empty on high-headroom rungs, established in D2). delta_r=0.05,
m_test=250, m_fit=40, n_batch=64, k-=8, k+=2, tau=0.5. Student/probe protocol
EXACTLY as D2 (MLP d->64->32, Adam 1e-3, 1500 steps, bs 64; probe 1000/2000
fresh, 300 steps). 10 seeds per arm.

## Rungs
The same high-headroom band rungs from D2 (H > 2x floor AND S>0.10 @ alpha=0.40):
K20_s2.4_a0.0, K40_s2.4_a0.8, K10_s2.4_a0.0, K20_s2.0_a0.8, K40_s2.4_a1.6
(spanning rho_tail 0.19-0.59). no_mine/supcon/naive_neg reused from
d2_preflight.json + d2_utility_band.json where alpha-independent; mining arms
re-run at alpha=0.40.

## Arms (10 seeds each)
1. no_mine            baseline (reuse).
2. naive_hard         negative control (reuse d2 naive_neg).
3. rc_hard            D2 failed method, FOIL ONLY (reuse d2 rc_hpm).
4. cert_random_k      certified, random among certified negatives (reuse d2).
5. cert_conf_easy     certified negatives with the LOWEST same-class
                      probability rho_hat(s) (most-confident-clean), top-k- by
                      (1 - rho_hat) i.e. EASIEST-to-trust, not hardest.
6. cert_mid_band      certified negatives from the MIDDLE similarity band:
                      among certified, drop the top and bottom hardness
                      terciles, sample k- from the middle.
7. cert_diverse       certified negatives maximizing coverage: greedy
                      farthest-point selection in teacher-embedding space among
                      the certified set (k- picks that are mutually dissimilar).
8. cert_curriculum    schedule: start from cert_conf_easy; every C steps, if
                      realized batch risk (L-) over the last window <= alpha,
                      shift the selection band one tercile harder (easy->mid->
                      mid-hard), capped BELOW the top hardness tercile (never
                      enters the residual-FN tail). If risk exceeds alpha, step
                      back one tercile. Pure teacher-driven selection (theta-
                      free); risk evaluated on TEACHER labels of the batch
                      (synthetic) / on the gate-certified proxy elsewhere.

All weights stop-gradiented, teacher-side (R1/R2). Positives = aug view only
(D2 B3: positives channel not load-bearing; drop mined positives everywhere ->
this is the negatives-only debiased form). v- = (1 - rho_hat(s)) for the loss
denominator, unchanged. Curriculum changes ONLY which certified negatives enter
top-k-, never the weights.

## Curriculum schedule (pinned)
C = 150 steps per curriculum check. Window = last 150 steps. Terciles of the
CERTIFIED-negative similarity distribution (recomputed per rung at calibration,
fixed thereafter): T_low (easy), T_mid, T_high (hard, EXCLUDED ceiling).
Bands available to the schedule: {T_low}, {T_low,T_mid}, {T_mid}. Start at
{T_low}. Advance/retreat by the risk rule above. Selection within the active
band is farthest-point-diverse (ties to cert_diverse) to avoid collapse.

## Pre-registered gates
G-D3 (all four evaluated; outcome = which fire):
 1. SAFETY: realized risk (L- on fresh batches, 200) <= alpha for
    cert_curriculum on every band rung. (machinery sanity)
 2. UTILITY: cert_curriculum > no_mine by >= 2x seed-SE on >= 1 nontrivial
    rung (nontrivial = H > 2x floor, which all band rungs satisfy).
 3. MECHANISM: cert_curriculum > cert_random_k by >= 2x pooled seed-SE
    (does curriculum beat random-certified?).
 4. ANTI-HARDNESS: rc_hard < cert_curriculum on every band rung (confirms the
    pivot; expected from D2).

## Branch outcomes (pre-registered)
C1 BEST: gates 1+2+3 hold -> certification curriculum produces UTILITY ->
   new paper direction; THEN (and only then) consider the GPU EqM question.
C2 MIDDLE: gate 2 holds for cert_random_k OR cert_conf_easy but gate 3 fails
   (curriculum no better than random-certified) -> "certified pairs are a
   cheap safe regularizer"; simpler safety+mild-utility paper. Report which
   simple certified selector is best.
C3 WORST: gate 2 fails for ALL certified selectors (none beat no_mine by the
   margin) -> certification is ONLY a safety guardrail; write the
   bounded-harm / risk-dial / abort-visibility paper and STOP chasing utility.
Any result matching no branch -> STOP + postmortem, no improvised branch.

## Reporting
Primary table: {no_mine, cert_random_k, cert_conf_easy, cert_mid_band,
cert_diverse, cert_curriculum, rc_hard(foil), naive(floor), supcon(ceiling)}
x band rungs, 10 seeds, mean +/- seed-SE. Gate verdicts machine-written to
results/d3_verdict.json. Per-rung comparisons descriptive; the gate decisions
(2,3,4) are the confirmatory tests. Curriculum trajectory (active band over
training) logged for the designated rung.

## Compute
No GPU. 6 mining arms x 5 rungs x 10 seeds + reuse. Est. 1-2 h CPU background.
