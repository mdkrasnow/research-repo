# C3 — OOD detection via EqM field-residual: findings

Pre-registered: `c3-ood-energy-detection-proposal.md`. Job **18952117** (gpu_test, COMPLETED
exit 0, 8:55). Code: `experiments/c3_ood_energy/ood_energy_detect.py`. Data:
`documentation/c3_data_auroc_18952117.csv`, `results/c3_ood_energy/run_18952117/`.
Arms: vanilla (FID 31.41) / v10 λ=0.1 (29.01) / v10 λ=0.3 (27.09), frozen EMA, ebm='none'
→ score = field magnitude ‖f(x,y)‖. N=2000 in-dist IN-1K val, 2000 per OOD set.

## Result — NEGATIVE. Hypothesis refuted; C3 killed.

AUROC (in-dist vs OOD; higher = better OOD detector):

| OOD type | vanilla | v10 λ=0.1 | v10 λ=0.3 | Δ(λ03−van) | read |
|---|---|---|---|---|---|
| gaussian | 1.000 | 1.000 | 1.000 | 0 | **saturated** — far-OOD trivially separated, no headroom |
| uniform | 1.000 | 1.000 | 1.000 | 0 | saturated |
| wrong_label | 0.492 | 0.499 | 0.500 | +0.008 | **null** — ‖f‖ ~identical for right/wrong label (uncond zeroes t; field-norm is x-driven, not y-driven) |
| **patch_shuffle** | **0.819** | **0.770** | **0.726** | **−0.093** | only discriminative axis — **v10 WORSE, dose-ANTI-ordered** |

Pre-registered promotion rule: v10 AUROC > vanilla on ≥2 OOD sets, dose-ordered λ03≥λ01≥van.
**Observed the exact opposite** on the only set with signal: vanilla > λ01 > λ03 (monotone in the
wrong direction). Kill rule met → **C3 OOD-detection claim killed.**

## Mechanism (clean, not post-hoc laundering)

The field-norm gap that OOD detection relies on SHRINKS with mining dose:

| | vanilla | λ=0.1 | λ=0.3 |
|---|---|---|---|
| in-dist mean ‖f‖ | 13.05 | 13.58 | 14.43 |
| patch_shuffle OOD mean ‖f‖ | 26.69 | 24.12 | 23.82 |
| **gap (OOD − in)** | **13.64** | **10.55** | **9.39** |

v10's confirmed off-manifold field-robustness (Exp 2: v10 has SMALLER field error and is more
"data-like" at off-manifold points) directly **reduces** the off-manifold field magnitude → the
model is LESS surprised by corrupted inputs → a worse novelty detector. The very mechanism that makes
v10 better at generation (accurate off-trajectory field) makes it worse at OOD-via-field-norm. This
is internally consistent with Exp 2 and is a genuine (if negative) mechanistic finding, not a tuning
artifact. More mining → more robustness → smaller gap → lower AUROC. Monotone, dose-confirmed.

## Caveats (honest scope)
- `ebm='none'` checkpoints expose no scalar energy E; only ‖f‖ was testable. A model trained with an
  explicit energy head (ebm='l2'/'dot') might behave differently — but these are the trained models;
  the claim is about THESE checkpoints.
- gaussian/uniform saturation (AUROC 1.0) and wrong_label null mean patch_shuffle was the only
  informative cell. A single discriminative axis is thin, but it points clearly negative and is
  dose-confirmed, so escalating (more OOD sets / scale) is not warranted — the mechanism predicts the
  sign and matches.

## Decision
- **C3 → killed** per pre-registered rule. Does NOT satisfy the C2 gate (which required C3 ΔAUROC>0
  dose-ordered).
- Negative result is publishable as a one-line mechanism note: "ANM's off-manifold field-robustness
  trades away energy-based OOD-detection ability (dose-monotone)." Supports the Exp-2 robustness story
  from the opposite direction.
- C2 (restoration) now hinges ENTIRELY on C1 (inference-compute scaling). If C1 also negative, no
  capability probe cleared → differentiation falls back to the uniform FID gain + Exp-2 mechanism
  (workshop-tier), and C2 should NOT be launched (no positive evidence to justify it).
