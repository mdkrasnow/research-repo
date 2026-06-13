# Postmortem — RC-ANM toy ladder, branch R4 (2026-06-13)

Pre-registered gate G-RCANM (preregistration-rc-anm.md) returned **R4 BROKEN**:
RC-ANM damaged the field despite certification. STOP + postmortem (no
improvised branch). Results: results/rcanm_verdict.json.

| arm | field MSE | coverage | accept | flip-risk |
|---|---|---|---|---|
| vanilla | 0.837 | 0.275 | — | — |
| fixed_anm (eps .5) | 2.671 | 0.150 | — | — |
| aggressive_anm (eps 1.5) | 2.671 | 0.150 | — | — |
| rc_anm | 2.310 | 0.125 | 0.798 | 0.000 |
| oracle_safe_anm | 0.877 | 0.350 | 0.416 | — |

## Root cause: the implemented functional, not the method
gate (a) premise holds (aggressive damages, 2.67 vs 0.84); (c) parity holds
(rc ≤ fixed); (d) flip-risk realized ≤ α (0.0); but (b) FAILS (rc_anm 2.31 >>
vanilla). Three diagnostics pin the cause:

1. **The concept is validated by the positive control.** oracle_safe_anm
   (accept only endpoints whose TRUE Voronoi basin == source class; 42% accept)
   lands at MSE 0.877 ≈ vanilla AND the BEST mode coverage of any arm (0.350 vs
   vanilla 0.275). A perfect safety filter makes adversarial mining safe and
   mildly beneficial. RC-ANM's design is sound.

2. **The implemented safety score is degenerate at this teacher quality.**
   flip-basin risk = 0.000 at every eps_ball (smoke + run) -> calibration
   certified the LARGEST eps (1.2) for every gamma bin and accept/reject passed
   80% of endpoints. The 1000-step EMA teacher's gradient-descent basin is
   insensitive to the endpoint (everything flows to a few attractors), so a
   descent-based flip metric cannot distinguish safe from unsafe endpoints. The
   functional certified almost everything -> RC-ANM ~= aggressive -> damage.
   oracle_safe used the ANALYTIC basin of the mined point directly (no teacher
   descent) and worked; the teacher-descent proxy of it did not.

3. **The 2D toy cannot reproduce ANM UTILITY (E1.2 lesson, confirmed).** fixed
   ANM (= v10) damages the toy (2.67) but v10 HELPS at IN-1K (FID 27.58 vs
   31.41, diff-EqM). fixed == aggressive to 3 decimals (identical collapsed
   field) -> any mining at eps >= 0.5 collapses the tiny 2D field to the same
   broken solution. The toy is the wrong instrument for the "fixed/RC-ANM
   matches-or-beats" half of the gate: there is no ANM benefit to match here.

## What this licenses (iteration-2, NEW pre-registration — not a rescue)
The toy validated the CONCEPT (oracle-safe) and the PREMISE (aggressive
damages) but cannot validate the implemented functional or ANM utility. Two
fixes, both pre-registered before any rerun:

A. **Functional fix:** replace teacher-descent flip-basin with an
   oracle-MATCHING proxy that does not accumulate descent noise:
   - single-step teacher-field direction at x_t_adv vs the analytic basin
     geometry (does the teacher field at the mined point point toward x1's
     mode?), and/or
   - a STRONGER teacher snapshot (step 2500-3500, not 1000) so the basin is
     informative. Validate the proxy against the analytic oracle (must agree
     with voronoi_basin(x_t_adv)==lab on labeled toy data) BEFORE calibrating.
B. **Right instrument for utility:** the matches-or-beats-ANM half belongs at
   CIFAR mini, where fixed/v10 ANM is KNOWN to help and teacher signals are
   richer. The toy keeps only the SAFETY half (aggressive damages; certified
   filter avoids it; oracle-safe is the ceiling).

## Recommendation (escalation)
The 2D toy has delivered its informative verdict: concept good, premise good,
functional + utility-measurement need a different instrument. Two paths:
1. iteration-2 toy with the functional fix + stronger teacher (cheap, CPU) to
   get RC-ANM's IMPLEMENTED filter to track the oracle, THEN
2. CIFAR mini RC-ANM (vanilla / v10 fixed-ANM / RC-ANM) where ANM utility is
   real — the goal's step-5 CIFAR comparison. v10 fixed-ANM is the live,
   scale-proven mechanism; RC-ANM's value proposition is "v10's gain + a
   certified safety filter against the wrong-basin endpoints v10 mines blind".

Do NOT declare RC-ANM dead: unlike the contrastive arm (oracle-NULL, no
channel), RC-ANM's oracle control WORKED (safe + better coverage). The
channel exists; the CPU functional proxy is the gap. Per CLAUDE.md, GPU/CIFAR
needs explicit human approval; flagged in pipeline.json.

## ADDENDUM — dose-confound check (decisive): concept is REAL, not dose
Concern: oracle_safe (42% accept) might beat the other mined arms only because
it MINES LESS (MSE was monotone in accept fraction: 0%/42%/80%/100% ->
0.84/0.88/2.31/2.67). Ran a dose-matched control (results via
experiments/rcanm_dose_check.py, 5 seeds):

  oracle42 (basin-safe, 42% accept):  MSE 0.877 +/- 0.115   cov 0.350 +/- 0.105
  random42 (RANDOM,     42% accept):  MSE 1.285 +/- 0.285   cov 0.225 +/- 0.056

At IDENTICAL dose, basin-safe selection beats random by ~1.4 SD on MSE and
~1.2 SD on coverage. So WHICH endpoints you mine matters, not just how many:
the certification TARGET (basin-safety) carries real signal. oracle_safe ~=
vanilla (0.877 vs 0.837) while dose-matched random damages (1.285). The R4
failure is therefore localized entirely to the IMPLEMENTED proxy (teacher-
descent flip-risk, degenerate on the weak 1000-step teacher), NOT the concept
and NOT a dose artifact. This is the opposite of the contrastive arm-A result
(oracle-NULL, no channel): here the oracle WORKS. The channel exists; build a
proxy that tracks it.

## Final branch
R4 (RC-ANM damaged despite certification) with CONCEPT VALIDATED by a
dose-matched oracle. Pre-registered STOP + postmortem reached; iteration-2
spec (functional fix A + CIFAR-mini utility test B) recorded above; both
require human go (CPU functional fix is cheap; CIFAR/GPU per CLAUDE.md needs
approval). No improvised branch taken.

## v2 RESULT — R2' (concept real, CPU proxy not instrumentable)
Proxy-validation gate (preregistration-rc-anm-v2.md, experiments/
rcanm_v2_validate.py): short-rollout teacher proxy vs analytic-oracle basin,
balanced accuracy:
  teacher@1000 steps: 0.492  (chance)
  teacher@2500 steps: 0.529  (barely above chance)
Both FAIL the >=0.65 bar. Even a stronger toy teacher's GD field is too
collapsed (every point descends to a few attractors irrespective of the mined
endpoint), so a teacher-based basin proxy cannot track the analytic oracle at
2D toy scale. Per prereg this is **R2': the basin-safety channel EXISTS (oracle
+ dose-check prove it) but is NOT instrumentable by the CPU teacher proxy** —
distinct from R4 (not a bug; wrong instrument). Pre-registered action: escalate
to CIFAR mini, where the teacher (real EqM-EMA or rn18) is far richer and
field/basin signals are meaningful. Did NOT fish for a passing proxy (prereg
forbids). Terminal CPU state for the RC-ANM pivot.
