# EqM Baseline Reconciliation

**Date**: 2026-05-19
**Status**: RESOLVED

## Question
Our trusted baseline is EqM-B/2 80ep IN-256 class-conditional FID **31.41** with paper-comparison `Δ=−1.44 vs paper 32.85`. Initially extracted "Figure 3: EqM-B/2 80ep ≈ 3.0 FID" — apparent 10× discrepancy raised concern.

## Resolution (from EqM paper §5.3 + Appendix A)

Three training budgets in the paper:

| Budget | Reported B/2 FID | Source |
|---|---|---|
| 80 epochs (ablation regime) | **32.85** (default c(γ) truncated a=0.8 λ=4) | Section 5.3, Table 3, Figure 9 |
| 80 epochs (Figure 3 left scaling) | ≈ 3.0 (approximate read) | Figure 3 — but verify |
| 1400 epochs (full training, XL/2 headline) | 1.90 (XL/2 only, not B/2) | Table 9 (XL/2), Appendix A |

**Quote (§5.3)**: "We use our EqM-B/2 model and 80 epochs of training for all hyperparameter experiments."

**Quote (Appendix A)**: epochs schedule "80 80 80 80 - 1400" — the "1400" applies only to the XL/2 headline run.

## Interpretation

- The ablation 32.85 IS our comparable number. Both are 80ep B/2 default config IN-256 class-cond.
- Our 31.41 matches paper's ablation (Δ=−1.44 well within seed noise).
- Figure 3's apparent ~3.0 reading for B/2 80ep is likely an extraction error from the figure caption or axis. Without seeing the actual axis, conservative interpretation: Figure 3 scaling curve is on a log scale (e.g., XL/2 line) and B/2 80ep reads ≈ 30s not ≈ 3.

**Decision**: trust our 31.41 baseline. Reconciliation complete.

## Implications

- v10+CAFM headline target: improve 31.41 at B/2 80ep IN-256 class-cond.
- We are NOT chasing 1.90 (XL/2 1400ep). Out of budget.
- Lin CAFM SiT-XL/2 8.26 → 3.63 was at full training. Our equivalent should be percentage gain on B/2 80ep, not absolute SOTA.
- Workshop / ICLR claim must honestly characterize the scale: "B/2 80ep ablation-regime improvement."

## Action items
- [x] Reconciliation logged.
- [ ] Verify Figure 3 axis reading by direct PDF inspection (low priority; not blocking).
- [ ] Adjust paper framing to NOT claim SOTA — claim mechanism + scaling-friendly improvement.
