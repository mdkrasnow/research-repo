---
name: diff_eqm_capability_ladder_null
description: ANM/v10 capability-acquisition ladder NULL — FID gain does not transfer to image-repair tasks
metadata: 
  node_type: memory
  type: project
  originSessionId: 74757461-ae3e-41e5-9da5-d3db38cca78e
---

Capability ladder (2026-06-07) tested whether v10 PGD hard-example mining unlocks
downstream image ops beyond its FID gain. Frozen-weight, identical-GD-sampler,
vanilla-s0-400k vs v10-s1-400k checkpoint comparison.

**Result NULL.** Rung1 (denoise/inpaint/compose, n=16) + Rung2 (held-out transfer
gray/lowres/blur/crop, n=32) both v10≈vanilla at noise level (ΔPSNR<0.03dB,
ΔLPIPS≤0.004; denoise EXACT parity all gammas). v10 FID gain (27.58 vs 31.41) does
NOT convert to conditional repair capability. Ladder KILLED per pre-registered gate.

Consequences: random-corruption control train (`--v10-random-control`/`V10_RANDOM=1`,
coded, committed 5999cee) NOT submitted — no signal to defend, 30h saved. Rungs 3-5
not run. v10 stays a generation/FID method, NOT capability-expansion — do not claim
downstream transfer in paper. needs_user_input set: accept null (recommended) vs
stronger probe (3-seed/n>=256/trained head).

Infra kept: `eval_capabilities.py` (denoise/inpaint/compose/restore modes),
`eval_rung2_transfer.sbatch`. Results: capabilities_19842241/, rung2_transfer_19843548/.
Postmortem: postmortem-capability-ladder-2026-06-07.md. See [[diff_eqm_v10_in1k_3seed]].
Caveat: crop/outpaint harness underpowered (margins stay grey both arms, inconclusive
for outpaint only). v10 seed0 lambda03 ckpts pruned/missing — used seed1 not best seed0.
