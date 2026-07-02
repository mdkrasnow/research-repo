# Deviations log

## D1 — 2026-06-12 — Pre-run amendment A1 (calibration budget arithmetic)
preregistration.md originally set m=20 calibration batches / α₀=0.05. Smoke
of the pipeline (before any gate was evaluated) showed m=20 cannot certify any
λ at α=0.05 even with zero observed loss (HB p ≥ 0.36 > δ_r) — the P1
arithmetic the tree warns about. Amended: m_test=250, m_fit=40, n_fold=32,000,
α₀=0.10 primary / 0.05 stretch / 0.20 loose. Derivation in preregistration.md
Amendment A1. Not outcome tuning: no gate criterion was evaluated against any
treatment before the amendment.

## D2 — 2026-06-12 — E1.0 teacher substitute
Spec E1.0 says "EMA embeddings"; no trained EqM EMA exists at CPU stage.
Pre-registered substitute: torchvision resnet18 ImageNet-pretrained
penultimate features (see preregistration.md E1.0).

## D3 — 2026-06-12 — Safety arms on supply-bearing rungs (B3/foil testability)
preregistration-d2.md pre-flight gate routes rungs failing (a) H>2xfloor OR (b)
S>0.10 to "naive_neg only". But B3 (H-B' retention) and the RINCE foil are
SAFETY questions needing only certified-hard SUPPLY (b), not headroom (a). The
H-S anticorrelation meant NO rung passed (a)+(b), so B3/foil were untestable
under the strict reading despite supply-bearing rungs existing (K40_s1.2_a0.8
S=0.573 with naive damage 0.395). Deviation: run the safety arm set on the
max-damage supply-bearing rung to test the pre-registered B3/foil/FP-pull
endpoints. NOT a new hypothesis — tests pre-registered endpoints on the rung
geometry the generator actually provided. experiments/d2_safety_arms.py.

## D4 — 2026-06-12 — Linchpin refocus (compute)
d2_supply_alpha_probe.py originally swept all 9 toy rungs x 3 alpha
(~2.25h sequential). The DECISION needs only: do high-H rungs (H>0.04) reach
S>0.10 at the loosest alpha? Refocused to high-H rungs x {0.20, 0.40},
parallel. Full S-vs-alpha table demoted to optional. No threshold changed.

## D5 — 2026-06-12 — Band evaluated at alpha=4*alpha_0 (supply-measurement bug)
The pre-flight measured certified-hard supply S at alpha_0=0.10 ONLY. At that
tight risk budget the LTT certifies almost nothing in the top similarity decile
(S~0), which FAKED an H-S anticorrelation and a premature "band absent (B2)"
lean. The pre-registered linchpin (d2_supply_alpha_probe) caught it: at the
pre-registered looser endpoints (P2 alpha-grid: 0.20, 0.40) the high-headroom
rungs have S=0.4-0.92. Correction: the band utility ladder runs at alpha=0.40
(P2 tertiary endpoint, 4*alpha_0), uniform across all high-H rungs so the
rho_tail band regression is unconfounded by alpha. The alpha-frontier sweep
(safety arms) separately maps the safety-supply tradeoff. This is the system
working: the linchpin existed precisely to prevent declaring B2 on a
throttled-supply artifact.

## D6 — 2026-06-13 — RC-ANM r_basin = FLIP risk (pre-gate, mirrors D2)
Smoke (before any gate): absolute basin-error rate of mined endpoints = 0.70,
but the UN-MINED baseline is 0.61 — a pure-noise endpoint has no canonical
Voronoi basin, so absolute r_basin measures teacher quality + geometry, not
mining damage, and certifies nothing (all eps_ball -> None). Refined the
certified functional to FLIP risk: r_basin=1 iff mined-wrong AND unmined-right
(mining-induced damage only; ~0 for un-mined by construction). Identical to the
D2 calibrate_arm_b flip-risk fix; the four diagnostic scores unchanged. No
threshold touched; functional definition refined pre-gate per the same lesson.
