# ASM CPU Ladder — Verdict (2026-06-09)

**Adversarial Symmetry Mining (ASM)** = mine the valid transform HARDEST for the current EqM model
(Madry inner-max in symmetry space), vs static v17 "which transform is valid?". Tested on a gated CPU
ladder — CPU decides whether GPU is warranted; no GPU on noise.

## Results

| stage | gate | verdict |
|---|---|---|
| CPU-A unit/validity | PASS | miner valid; all 13 families run; decoys 100% rejected; hardness finite (loss/comm/loss+comm) |
| CPU-B positive-control gap | PASS | ASM_loss mines `saturate` (the missing factor), beats random 0.403<0.413 on TinyEqM eqm_full. **Commutator signal picked pad_crop, NOT saturate** — did not ID the missing factor on a lightly-trained probe. |
| CPU-C full-CIFAR (3 seeds) | **FAIL** | SOLO (ASM<random) passed seed0 (+0.016) but FAILED seeds 1,2 (−0.005, −0.001). Mean +0.003 = noise. HYBRID also inconsistent (F/F/T). **base (no aug) beats ALL aug every seed** (0.605/0.606/0.642 lowest). static_v17 landed on a decoy (color_collapse) seed0. |

## Conclusion

**ASM does NOT beat random on full CIFAR — confirmed across 3 CPU seeds. NO GPU ASM launched.**

Two clean findings:
1. **CPU-B confirms the mechanism works WHERE THERE IS A GAP** — ASM (loss-hardness) recovers the missing
   factor (saturate), reproducing the gap15 story at proxy scale. Validity firewall holds (decoys rejected).
2. **CPU-C confirms there is no full-CIFAR win** — with no gap, hardness-mining ties/loses to random and
   all augmentation hurts the small proxy. The seed0 SOLO pass was an artifact; 3-seed killed it. This is
   the SAME verdict as the static v17 full-CIFAR ladder (discovery ties random) — making the adversarial
   objective adversarial did NOT change the full-CIFAR outcome.

**The commutator signal underdelivered** (CPU-B picked pad_crop not saturate); on a lightly-trained probe
it doesn't isolate the missing factor. Would need a well-trained field to test properly — not pursued
since CPU-C already gates out full-CIFAR GPU.

## Decision

- **gap15 (constructed-gap, real EqM, discovered saturate beats crop +3.4 FID) remains the flagship.**
- No GPU ASM ladder. No HP tuning (no real effect to tune).
- The ASM machinery is preserved (`asm_miner.py`, `asm_cpu_ladder.py`) and CPU-B shows it works on gaps —
  reusable if a future task has a real missing factor. ASM's value, like all discovery here, is
  GAP-CONDITIONAL: it earns its keep only when something is genuinely missing.

The gated CPU program did its job: **caught a noise-level full-CIFAR signal at ~zero GPU cost.**
