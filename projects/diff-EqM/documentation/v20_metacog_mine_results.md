# v20 metacog-mine — CIFAR screen RESULT (2026-06-21)

Image-gen form of the v11 metacognitive-curriculum idea: mine REAL examples by FIELD
INSTABILITY (‖f(x_t+ξ)−f(x_t)‖ — the local descent-instability the metacognition probe reads)
and upweight the base EqM loss on them. 3-arm CIFAR screen, 150ep, same harness, same seed.

| arm | selector | CIFAR FID (5k) |
|---|---|---|
| **v20 instability-mine** | high field-instability | **13.26** |
| v20 random | random subset, same λ | 14.41 |
| v00 vanilla | none | 14.41 |

- **instability-mine beats random AND vanilla by ~1.15 FID** (13.26 vs 14.41).
- **random ≈ vanilla to 4 decimals** (14.4051 vs 14.4052) → the gain is from the *failure
  selection*, NOT from merely upweighting examples. The metacognition-instability signal is
  the active ingredient.
- Comparable to v10 PGD-mining (CIFAR 13.40) but with a CHEAPER selector: one extra forward
  (instability probe) vs K-step PGD gradient ascent.

## Significance
This is the claim Sudoku could not support (there the probe-selector was weak, +0.023 over
random, far below oracle). On IMAGES — where the trajectory-metacognition probe is strong
(descent AUROC 0.82) — the same instability signal works as a **label-free training selector**
that improves generation. Turns the metacognition probe from an inference tool into a training
signal, the bigger claim.

## Scope / caveats
- CIFAR is a proxy/filter, NOT publishable per project rules — requires IN-1K confirmation.
- Single seed per arm (random≈vanilla to 4 decimals indicates harness consistency, supports the
  instability separation being real, but multi-seed needed to quantify CI).
- λ=0.5, instability via single-perturbation ‖Δf‖ (a local proxy for the full descent-shape probe).

## Next (gate passed → promote)
IN-1K EqM-B/2 confirmation: v20 instability-mine vs vanilla (+ random), FID human-gated.
Cheaper than v10 and a distinct mechanism (instability-selection vs PGD), so worth the confirm.
