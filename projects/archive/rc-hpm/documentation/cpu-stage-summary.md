# RC-HPM CPU Stage Summary (Stages 0–1, 2026-06-12)

Full CPU portion of the decision tree (documentation/experiment-decision-tree.md
v2) executed end-to-end. Verdicts machine-written to results/*.json.

## Gate ledger

| node | gate | verdict | one-line outcome |
|---|---|---|---|
| E0.0 bug-injection suite | G-1 | **PASS** | 9/9 injected bugs detected, clean run silent (results/e0_0_verdict.json) |
| E0.1 calibration validity | G0 | **PASS** | 20/20 seeds certified at α₀=0.1, 0 exceedances; pooled L⁻=0.031, L⁺=0.018; ⊖ random gate 20/20 vacuous ABORT; ⊕ oracle clean |
| E0.2′ damage/utility | G1 | **K2-only** | naive hard-mine craters probe acc 0.27 vs 0.95; RC-HPM prevents damage at certified risk; NO utility gain (probe at ceiling; RINCE≈no-mine≈RC); cert-random-k == RC (hardness adds nothing, F15) → **harm-bounding + guarantee framing** |
| E1.0 premise (CIFAR) | G1.5 | **PASS** | damage density 14.3× concentrated in top hardness decile (≥3 required); decays 9.4/3.9/2.0 with γ — P4 structure confirmed; figures/e1_0_premise.png |
| E1.1 ARM A (disp-pair) | G2 | **FAIL** | ⊕ oracle-pairs NULL on field MSE (0.890 vs vanilla 0.837) → bounds the whole mechanism class; rc_hpm ties vanilla |
| E1.2 ARM B (endpoint) | G2 | **FAIL** | calibrated + certified (eps_ball 0.1, flip risk ≤ α) but worse-in-noise on recall distance; damage arm also in noise (margin 0.65) |
| E1.3 MNIST EqM-mini | G3 | **SKIPPED** | entry condition (a G2-passing arm) unmet — formal skip |

## Headline findings
1. **The certified-mining machinery is correct and validated** — the
   spec-faithful LTT/Hoeffding–Bentkus pipeline holds its guarantee across 20
   seeds, fails loudly under 9 classes of injected bugs, and abstains
   (vacuous) under a worthless gate. Portable to any future host task.
2. **Damage is real, certification prevents it**: uncertified hard-negative
   mining destroys representations (probe 0.27 vs 0.95); RC-HPM mines at
   certified risk with zero damage. No utility gain at toy scale → paper
   framing = harm-bounding + guarantee, per pre-registered branch.
3. **Premise verified in the wild** (CIFAR/rn18): pair-label damage
   concentrates 14.3× at the hard tail, exactly where mining mines.
4. **EqM bridge dead at toy scale**: with TRUE labels, contrastive structure
   on field activations does not improve the 2D EqM field — an oracle-null
   that bounds all certified variants. Postmortem:
   documentation/postmortem-g2-eqm-bridge.md. Scope caveat: 2D MLP toy;
   transformer-scale Dispersive-Loss gains are not contradicted.

## Amendments (all pre-run / pre-gate, logged in preregistration.md + deviations.md)
A1 calibration-budget arithmetic (m=250, α₀=0.10) — the P1 wall, hit exactly
   as the tree predicted.
A2 detector redesign (fold-reuse → data-flow guard; monitor → multi-statistic
   batch-level KS; q_mean of mined sets catches live-student drift 8/8).
A3 2D task de-saturation (8 unequal modes; mode-recall-distance primary).
A4 γ-windowing (mine where the teacher separates) + flip-risk functional for
   endpoint arm + teacher bandwidth 3.0 (measured).

## Next decision (human)
pipeline.json:needs_user_input set. Options:
(a) Stage 2 (GPU CIFAR) on the contrastive-CL standalone harm-bounding story —
    per tree, G4(b) corrupted-gate demo is the load-bearing experiment;
(b) redesign the EqM-bridge toy (oracle-null may be a 2D-scale artifact);
(c) park rc-hpm.
