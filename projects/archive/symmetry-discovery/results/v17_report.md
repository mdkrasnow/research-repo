# v17 MorphismGym — Results Report

## Phase 2 — Payoff
EqM-lite (field-matching, the target analog) is the PRIMARY gate. The shape-ID classifier is a calibrated-INSENSITIVE diagnostic (0B: oracle~=random; shape identity is robust to these morphisms) and is NOT gated.

### classifier / multi_independent — FAIL (higher better)
| BASE | ORACLE | RANDOM_VALID | DISC_SINGLE | **DISC_MULTI** | NO_ANCHOR |
|---|---|---|---|---|---|
| 0.1797 | 0.3372 | 0.4115 | 0.2865 | **0.3346** | 0.3516 |
- beats_base=True beats_random=False near/over_oracle=True anchor_essential=False

### classifier / single_rotation — PASS (higher better)
| BASE | ORACLE | RANDOM_VALID | DISC_SINGLE | **DISC_MULTI** | NO_ANCHOR |
|---|---|---|---|---|---|
| 0.5267 | 0.9642 | 0.6497 | 0.9525 | **0.9577** | 0.5605 |
- beats_base=True beats_random=True near/over_oracle=True anchor_essential=True

### eqm_lite / multi_independent — PASS (lower better)
| BASE | ORACLE | RANDOM_VALID | DISC_SINGLE | **DISC_MULTI** | NO_ANCHOR |
|---|---|---|---|---|---|
| 0.4659 | 0.0784 | 0.1066 | 0.0588 | **0.0617** | 0.3015 |
- beats_base=True beats_random=True near/over_oracle=True anchor_essential=True

### eqm_lite / single_rotation — PASS (lower better)
| BASE | ORACLE | RANDOM_VALID | DISC_SINGLE | **DISC_MULTI** | NO_ANCHOR |
|---|---|---|---|---|---|
| 0.0038 | 0.0037 | 0.0037 | 0.0036 | **0.0037** | 0.0037 |
- beats_base=True beats_random=True near/over_oracle=True anchor_essential=True

**Phase 2 verdict (EqM-lite PRIMARY): eqm_lite=True | classifier(diagnostic)=False**
- discovered_multi vs single: {'classifier/multi_independent': {'discovered_multi_vs_single': 'multi_better'}, 'classifier/single_rotation': {'discovered_multi_vs_single': 'multi_better'}, 'eqm_lite/multi_independent': {'discovered_multi_vs_single': 'single_better_or_tie'}, 'eqm_lite/single_rotation': {'discovered_multi_vs_single': 'single_better_or_tie'}}
**EqM/FID: RECOMMENDED (EqM-lite passed)** — FID is never auto-authorized; a clean EqM-lite pass is a RECOMMENDATION to integrate, pending EXPLICIT human approval (CLAUDE.md gating discipline).
