# Phase 3 Spec — Scaling Curves (S/2, B/2, L/2, optionally XL/2)

Status: IN FLIGHT — vanilla baselines + v10 smokes submitted 2026-05-27 ahead of Phase 2 gate (sunk-cost acceptable per user 2026-05-27 directive: "compute summer budget is kinda made up; we have access to as much compute as we need").

## Objective
Establish whether v10's B/2 80ep gain (2.40 FID vs vanilla 31.41) **holds, grows, or shrinks** as the EqM model scales. Three outcomes:
- **Constant in FID** → uniform gain across scale; methodological contribution at all sizes.
- **Growing with scale** → strongest paper story (mining matters more for bigger models).
- **Shrinking with scale** → narrower claim; B/2-regime ablation result.

## Pre-registered gate
- v10 must beat vanilla by **at least 0.5 FID** at S/2 AND L/2 (single-seed each, no Welch required for scaling figure).
- XL/2 stretch: any positive gain considered confirmation of "scales to paper's headline architecture".

## Configurations

### Model architectures (from EqM upstream)
| Model | Hidden dim | Layers | Heads | Params | GBS | GPU-h est (80ep) |
|---|---|---|---|---|---|---|
| EqM-S/2 | 384 | 12 | 6 | ~32M | 256 | ~96 |
| EqM-B/2 | 768 | 12 | 12 | ~130M | 256 | ~144 (done) |
| EqM-L/2 | 1024 | 24 | 16 | ~450M | 256 | ~200 |
| EqM-XL/2 | 1152 | 28 | 16 | ~675M | 128–256 | ~320 |

All trained 80 epochs, IN-1K-256, class-conditional, gd sampler eta=0.003 steps=250 cfg=1.0, 50K-sample FID.

### v10 hyperparameters (held constant across scale, per Phase 1 PASS recipe)
- λ (gamma) = 0.1
- K = 1 (FGSM-style)
- ε_rad = 0.3
- mining_lr = 0.05
- mine_every = 1

Reason: same recipe enables clean apples-to-apples scaling comparison. λ retune is a Phase 5 ablation (separate axis).

## Smoke gates (PASS before launching full)
For each new model size, smoke = 1 epoch with reduced GBS:
- Smoke PASS = exit 0 + loss finite throughout + no OOM + no DDP hang.
- XL/2 smoke also informs final GBS (default GBS=32 in smoke; full run uses largest GBS that fits → likely 128 on 4×A100-80GB).

## Active + planned jobs

### In flight
| Job ID | Run | Type | Partition |
|---|---|---|---|
| 16369650 | vanilla S/2 seed 0 | full baseline | seas_gpu |
| 16369651 | vanilla L/2 seed 0 | full baseline | gpu |
| 16371622 | v10 S/2 smoke | smoke | seas_gpu |
| 16371637 | v10 L/2 smoke | smoke | seas_gpu |
| 16371642 | v10 XL/2 smoke (GBS=32) | smoke | seas_gpu |

### To submit on smoke PASS
| Run | Est cost | Comments |
|---|---|---|
| v10 S/2 seed 0 full | ~150 GPU-h | Submit after 16371622 PASS |
| v10 L/2 seed 0 full | ~300 GPU-h | Submit after 16371637 PASS; may need 2-job resume |
| v10 XL/2 seed 0 full | ~480-640 GPU-h | Submit after 16371642 PASS; needs multi-job resume |
| vanilla XL/2 seed 0 | ~320-480 GPU-h | Phase 3 stretch baseline; defer until v10 XL/2 smoke informs GBS |

### Optional (post-Phase 3)
- 3-seed at each scale for figure error bars (~3× the above).

## Required diagnostics
Same per-step logging as Phase 1/2 (CLAUDE.md mandatory). Plus:
- Wall time per epoch per model size (for Phase 4/5 timeline planning).
- Memory peak (to verify GBS choice on smaller GPUs).

## Decision tree
- **All scales PASS (gain ≥ 0.5)** → Phase 4 SiT transfer. Update workshop paper §4 with scaling figure as primary result.
- **S/2 fails, L/2 passes** → mining helps at scale but not below; narrow to "B/2 and up".
- **L/2 fails, S/2 passes** → mining helps at small scale only; would need theory explanation (unlikely outcome).
- **Both fail** → mining is a B/2 80ep oddity. Phase 4 skipped; paper becomes negative result + CAFM postmortem (still publishable as workshop spotlight on regression-target adversarial training failure modes).

## Expected timeline
- Smokes: complete within ~24h (depends on queue).
- Vanilla S/2 baseline: ~24h after start.
- Full v10 S/2: launched on smoke PASS, ~30h after start.
- L/2 + XL/2: 2-4 days each.
- Phase 3 verdict: ~5-7 days from 2026-05-27 → roughly 2026-06-03 to 2026-06-05.
