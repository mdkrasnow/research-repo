---
name: diff-EqM Stage B vanilla baseline verified
description: IN-1K-256 EqM-B/2 vanilla 80ep gives FID 31.41 (paper 32.85, Δ=-1.44). Pipeline reproduces paper; baseline trusted; aux-loss work unblocked.
type: project
originSessionId: 9bce0bb7-107d-4741-8833-aa999578b23b
---
**Baseline FID 31.41** on vanilla EqM-B/2 IN-1K-256, 80ep, 50K samples (verified 2026-05-13).
Paper FID = 32.85. We beat paper by 1.44 at epoch 76/80.

**Why:** Per CLAUDE.md research process rules added 2026-05-09, no auxiliary-loss work allowed until vanilla baseline verified against paper. Gate: |FID - 32.85| < 5. PASSED.

**How to apply:**
- Use this 31.41 (and final.pt at step 405000) as the v10 comparison baseline. NOT 32.85.
- Compare any DG-ANM/v10 variant against THIS run, same seed, same sampler config (gd, eta=0.003, steps=250, cfg=1.0).
- ckpt path: `projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/{0380000.pt, final.pt}`
- Config that reproduces paper: defaults (transport.py:122-126 hardcodes c(γ) interp=0.8 λ=4). No flags needed.
- Jobs: train 12010899 (timeout step 380000) + 12590805 (resume to step 405000); FID 12590806.
- All prior v01/v02/v06/v09 variants on EqM-B/2 saturated/collapsed because they violated EqM target geometry. Park unless EqM-compatibility argument written.
