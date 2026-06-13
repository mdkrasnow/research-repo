# Pivot: RC-HPM (contrastive) → RC-ANM (Risk-Controlled Adversarial Negative Mining for EqM)

Date 2026-06-13. Mainline of the rc-hpm project pivots from contrastive
hard-pair auxiliary losses on EqM to **RC-ANM**: certified adversarial-endpoint
mining native to Equilibrium Matching. The contrastive-EqM arms are archived as
a negative result. The statistical machinery is preserved unchanged.

## Why the pivot (the CPU-tree evidence)

The CPU decision tree (documentation/experiment-decision-tree.md v2) produced an
unusually trustworthy negative on the contrastive-EqM bridge AND validated the
machinery that RC-ANM reuses:

- **G-1 PASS** (E0.0 bug-injection): the LTT/HB certified-mining pipeline fails
  loudly under 9 classes of injected bugs and is silent when clean. The
  machinery is correct.
- **G0 PASS** (E0.1): 20/20 seeds certified at α=0.10, 0 risk exceedances;
  vacuous-abort under a worthless gate. Calibration is valid.
- **G1 (E0.2′)**: uncontrolled hard-negative mining DEMONSTRABLY damages
  representations (probe 0.27 vs 0.95); certification prevents it. The DAMAGE
  PREMISE — the thing risk-control exists to bound — is real.
- **G1.5 PASS (E1.0)**: on CIFAR with a frozen teacher, gradient-weighted
  damage concentrates 14.3× in the hard tail. The premise holds in the wild.
- **G2 FAIL, both contrastive-EqM arms (E1.1/E1.2)**: the LOAD-BEARING kill.
  Even **oracle-pair** SupCon (true labels, no mining, no uncertainty) on the
  EqM field's penultimate activations did NOT improve field MSE (0.890 vs
  vanilla 0.837), with real headroom proven present (capacity floor 0.282;
  D1-#1b). If perfect pair information cannot improve the field, no certified
  approximation of it can. The contrastive→EqM channel does not exist at this
  scale. Arm A is dead headroom-proven; arm B (endpoint, certified) was
  measurable-but-inconclusive in the noisy 2D recall metric.
- **D1/D2 follow-on**: confirmed the contrastive gains axis is dead — at the α
  where certified-hard supply exists, hardness-mining NET-HARMS (D2 F15:
  rc_hpm −6..−24 vs no_mine; cert_random_k ≈ baseline). Hardness and
  certifiability are anticorrelated (the 14.3× tension): the hardest certified
  pairs are the residual false-negatives the gate cannot clean.

**Conclusion.** The machinery works (G-1/G0), the damage premise is real
(G1/G1.5), but the contrastive auxiliary loss has no productive channel into
the EqM field (G2 oracle-null). The mistake was the MECHANISM (contrastive
aux on activations), not the safety apparatus.

## What survives and seeds RC-ANM

1. Damage is real and certification bounds it (G1) — keep the safety apparatus.
2. EqM endpoint mining (ANM / v10 lineage) ALREADY WORKS at scale in diff-EqM:
   PGD ascent on the EqM regression residual at the noise endpoint, FID 27.58
   vs vanilla 31.41 at IN-1K, 3 seeds (memory: diff_eqm_v10_in1k_3seed). That
   is an EqM-NATIVE hard-negative mechanism with demonstrated utility — exactly
   the productive channel the contrastive arm lacked.
3. The D2 caution applies and is the design driver: hardest = least certifiable.
   v10 mines the HARDEST endpoint at a fixed eps_ball with NO certification.
   RC-ANM adds the missing safety layer: certify each mined endpoint before
   training on it, and choose the largest eps_ball whose certified risk ≤ α.

## RC-ANM method (EqM-native)

Mainline. Object certified = the MINED ENDPOINT (not a contrastive pair).

1. **Candidate generation** — ANM/PGD endpoint mining (reuse eqm2d.pgd_mine):
   PGD ascent on ‖f(x_t) − (x1−ε)c(γ)‖² w.r.t. the noise endpoint ε, against a
   FROZEN EMA teacher field (P5: theta-free, exchangeable).
2. **Teacher safety scoring** (frozen EMA teacher g), per candidate endpoint —
   EqM-native risks, each in [0,1], higher = more dangerous:
   - r_field: teacher field DISAGREEMENT at x_t_adv vs the un-mined x_t
     direction (1 − cos of teacher fields), normalized.
   - r_target: γ-conditioned target inconsistency — does the mined target
     (x1−ε_adv)c(γ) still point toward x1's basin under the teacher?
   - r_basin: endpoint/basin consistency on labeled toy data — descend the
     frozen teacher field from x_t_adv; wrong attractor basin = unsafe
     (the v10 failure mode: mined endpoint flows to the wrong mode).
   - r_inflate: LOCAL loss inflation — residual at x_t_adv far exceeds the
     un-mined residual beyond a sane factor (over-mining).
   - r_return: trajectory return/contraction — does a short teacher GD
     trajectory from x_t_adv CONTRACT back toward the data manifold, or diverge?
   The pinned aggregate risk r = the basin/target-grounded combination declared
   in preregistration-rc-anm.md before calibration.
3. **Calibration** — reuse LTT/HB: per γ-bin, choose the LARGEST eps_ball whose
   certified-endpoint risk ≤ α (fixed-sequence over an eps_ball grid), OR an
   accept/reject threshold on r. γ-conditional (P4), batch-level, drift-
   monitored, risk-ledgered — the same apparatus that passed G-1/G0.
4. **Training integration** — train EqM on certified adversarial endpoints
   only. Arms:
   - vanilla (no mining)
   - fixed-eps ANM (v10-style, fixed eps_ball, uncertified)
   - aggressive ANM (large eps_ball, uncertified — the damage control)
   - RC-ANM accept/reject (mine, accept only endpoints with r ≤ threshold;
     reject → fall back to the un-mined endpoint)
   - adaptive-eps RC-ANM (per-γ-bin largest certified eps_ball)
   - oracle-safe ANM (accept only endpoints whose TRUE basin == source class —
     the ⊕ ceiling, toy only)

## Experiments (replace E1.1/E1.2 contrastive bridge)

RC-ANM ladder on the 2D EqM toy (8-mode unequal mixture, exact get_ct):
arms above, primary metrics = field MSE vs MC reference + mode coverage.

Pre-registered gate (preregistration-rc-anm.md):
- aggressive ANM MUST damage (worse than vanilla by margin) — premise.
- RC-ANM MUST avoid damage (≥ vanilla − margin).
- RC-ANM MUST match or exceed fixed-eps ANM on field error / mode coverage.
If passed → CIFAR mini vs vanilla and v10 ANM (small, then human-gated GPU for
B/2 scale; NO GPU on the dead contrastive path).

## Status of the old path
Contrastive-EqM (arm A) and the contrastive ladder (E1.1/E1.2, D1, D2, D3) are
ARCHIVED as a negative result — kept as baselines and as the evidence trail for
this pivot. No further compute on contrastive-EqM. Mainline = RC-ANM.
