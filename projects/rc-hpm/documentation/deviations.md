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
