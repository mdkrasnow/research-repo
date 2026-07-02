# Postmortem — ANM capability-acquisition ladder (2026-06-07)

## Question
Does v10 (PGD hard-example mining on EqM regression target) *unlock new image
operations* — inpaint / colorize / super-res / deblur / outpaint — that vanilla
EqM cannot do, vs merely improving FID? Hypothesis: mining sharpens the field
OFF the data manifold → better zero-shot repair prior.

## Design (pre-registered gates)
Checkpoint comparison, frozen weights, identical GD inference path both arms
(only the checkpoint differs → any gap = learned-field repair prior).
- Rung 0: sampler sanity on vanilla.
- Rung 1: zero-shot denoise / inpaint / compose, vanilla vs v10.
- Rung 2: held-out corruption transfer (gray / lowres / blur / crop) — v10 mined
  ONLY on generation hard examples, never on these corruptions.
- Random-corruption control train + Rungs 3–5: GATED on Rung 1/2 signal.

Arms: vanilla seed0 `0400000.pt` (FID 31.41) vs v10 seed1 `0400000.pt` (3-seed
mean 27.58), step-matched.

## Result — NULL
**Rung 1** (n=16): denoise EXACT parity across γ∈{.3,.5,.7,.9} (ΔPSNR ≤0.1,
ΔLPIPS ≤0.005); inpaint hole-PSNR van 7.275 vs v10 7.382 (+0.11, noise); compose
both-class top5 0.125 == 0.125.

**Rung 2** (n=32): all 4 held-out corruptions v10≈vanilla. ΔPSNR <0.03 dB, ΔLPIPS
≤0.004 every family. clf-top5 swings cancel (gray +0.09 / blur −0.09 ≈ 3/32 imgs).
crop ≈ corrupt baseline → outpaint failed BOTH arms symmetrically (GD-restore
harness limit, not a differentiator).

## Conclusion
Capability-acquisition hypothesis **falsified at this probe**. v10's FID gain
(−3.83) does NOT translate to conditional image-repair capability. Denoise exact
parity + 4-family transfer parity = convergent evidence the off-manifold repair
prior is unchanged by mining.

Pre-registered Rung-2 gate (v10 > vanilla on ≥3 families) → FAIL. Per CLAUDE.md
"pre-registered gate wins": ladder KILLED.

Consequences:
- Random-corruption control train (30h B/2) NOT submitted — no v10>vanilla signal
  to defend. Code preserved (`--v10-random-control` flag, `V10_RANDOM=1` hook).
- Rungs 3–5 (mask-mining train, translation specialist, compositional) NOT run —
  gated on Rung 1/2, both null.

## Framing impact
v10 remains a **generation / FID method** ("first adaptive hard-negative mining
for regression-target generative models"), NOT a capability-expansion method. The
workshop story is unchanged and intact; do not claim downstream-task transfer.

## Caveats (honest bounds on the null)
- n=16/32, single ckpt pair (v10 seed1, not best seed0 27.21). 3-seed + larger n
  could tighten but is unlikely to move parity-level deltas into a gap.
- GD-restore is a zero-shot proxy, not a trained conditional inpainter; a
  fine-tuned head MIGHT differ — but that tests finetuning, not ANM.
- crop/outpaint harness underpowered (margins stay near-grey); inconclusive for
  outpaint specifically, but symmetric so non-differentiating.

## Reusable infra (kept)
- `experiments/eval_capabilities.py`: modes denoise / inpaint / compose / restore
  (restore = gray/lowres/blur/crop, clamp-based, identical path both arms).
- `slurm/jobs/eval_capabilities.sbatch`, `eval_rung2_transfer.sbatch`.
- Results: `results/capabilities_19842241/`, `results/rung2_transfer_19843548/`.
