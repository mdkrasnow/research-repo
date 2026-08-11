# Phase 0 audit — longitudinal direct-specific Jacobian amplification (2026-08-10)

Audit of infra before building the progress-matched longitudinal experiment (job 38136938
follow-up). Per protocol: do not silently substitute unmatched checkpoints.

## 1. Existing script / job under audit

- Script: `experiments/direct_energy/matched_replay_jacobian_diagnostic.py`
- sbatch: `slurm/jobs/matched_replay_jacobian_diagnostic.sbatch`
- Prior runs: 38126114 (failed pre-flight, torch 2.1.2 `torch.nn.attention` missing, fixed
  commit 5947c1c), 38127797 (completed, single late checkpoint, superseded — used the WRONG
  interaction statistic, see module docstring correctness note), 38136938 (completed,
  corrected statistic + early/mid/late direct sweep, `none` held FIXED at one late reference).

## 2. Spike / control selection rule (UNCHANGED going forward, per Phase 1 requirement)

- Pool: `pool_size` real ImageNet batches, fixed images/t/y/noise (seed=0) — IDENTICAL
  tensors reused across every model/checkpoint evaluated.
- Ranked by REAL pre-clip training grad_norm (`probe_direct`/`probe_none` —
  `total_norm(model, block_groups(model))` after `exact_fwrev_backward` / plain backward).
- Spike = top `spike_frac` (=0.025, i.e. top 2.5%) by that ranking.
- Control = middle band, 40th–60th percentile by the same ranking, first `num_control`
  batches of that band.
- This criterion is preserved EXACTLY. Only `pool_size` and `num_control` are increased
  (Phase 1: get more genuine spikes by widening the collection window, not by relaxing
  the percentile).

## 3. Prior `none` checkpoint handling (WHY held fixed, and why that reasoning was incomplete)

job 38136938's `none_held_fixed_reason` claimed: *"none's earliest surviving checkpoint here
is 2.45M with unknown epoch correspondence to direct's numbering."* This was checked against
only ONE directory:
`.../longer80_none_seed0_ckpt50k_job36632776/.../checkpoints/` (2450000.pt … 3200000.pt,
epoch80.pt).

That directory is the LAST of a chain of resumed training jobs for the `none` arm, not the
whole lineage. Listing the full chain on netscratch:

| none job dir | checkpoint range |
|---|---|
| `longer40_none_seed0_ckpt50k_job36359207` | 0750000.pt … 1600000.pt, epoch40.pt |
| `longer60_none_seed0_ckpt50k_job36597079` | 1650000.pt … 2400000.pt, epoch60.pt |
| `longer80_none_seed0_ckpt50k_job36632776` | 2450000.pt … 3200000.pt, epoch80.pt |

This is a single continuous lineage saving every 50k steps from 750k through 3.2M — `none`
DOES have checkpoints spanning direct's entire early/mid/late range. The prior "hold none
fixed" scoping decision was an infra-discovery gap, not a real data limitation. Fixed here.

## 4. Matched checkpoints (this experiment)

CORRECTION (2026-08-11, added after job 38265027 FAILED on `FileNotFoundError`): the `...`
abbreviated paths below never recorded the actual filesystem root, which caused a real
resubmission failure in the follow-up `topk_subspace_diagnostic.py` job. **True root:**
`/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/` (netscratch, NOT
`/n/holylabs/.../masked-EqM/results/` as was wrongly assumed by convention when resubmitting).
Full paths:
- direct: `/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/fwrev_ep80_lambda0_job37780076/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/<step>.pt`
- none: `/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer80_none_seed0_ckpt50k_job36632776/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/<step>.pt`

Direct checkpoints (unchanged, `fwrev_ep80_lambda0_job37780076`, exact steps as filenames):

| stage | direct checkpoint | direct step |
|---|---|---|
| early | `.../fwrev_ep80_lambda0_job37780076/.../1600000.pt` | 1,600,000 |
| mid   | `.../fwrev_ep80_lambda0_job37780076/.../2175000.pt` | 2,175,000 |
| late  | `.../fwrev_ep80_lambda0_job37780076/.../2825000.pt` | 2,825,000 |

Progress-matched `none` checkpoints, chosen as the nearest available step by absolute value
(ties broken toward the LOWER step number, arbitrary but documented):

| stage | none checkpoint | none step | step mismatch |
|---|---|---|---|
| early | `.../longer40_none_seed0_ckpt50k_job36359207/.../1600000.pt` | 1,600,000 | **0.000%** (exact) |
| mid   | `.../longer60_none_seed0_ckpt50k_job36597079/.../2150000.pt` | 2,150,000 | 25,000 / 2,175,000 = **1.15%** |
| late  | `.../longer80_none_seed0_ckpt50k_job36632776/.../2800000.pt` | 2,800,000 | 25,000 / 2,825,000 = **0.885%** |

Caveats carried forward honestly:
- Both lineages use the SAME data pipeline / effective batch size (both are continuations of
  the same `fwrev`/`longer*` sweep infra), so step count is a reasonable proxy for "amount of
  gradient-descent progress," but the two arms are NOT the same run with a `ebm` flag flipped
  mid-training — they are independently-initialized, independently-optimized lineages that
  happen to share hyperparameters and data order seed. Step-matching controls for training
  DURATION, not for stochastic path divergence between the two arms.
- `none`'s `epoch40`/`epoch60`/`epoch80` labels don't align 1:1 with `direct`'s implicit epoch
  boundaries (direct's own epoch structure isn't independently re-verified here); step count is
  used as the matching key per the instruction ("prefer matching by optimizer/global step
  rather than merely epoch label"), which is exactly why this table reports raw steps, not
  epoch labels.

## 5. Existing K (job 38136938) vs Phase 1 target

- `pool_size=960`, `spike_frac=0.025` → `n_spike = round(0.025*960) = 24`.
- `num_control=24`.
- Target this run: `K>=48`, prefer `K=64`, SAME `spike_frac=0.025` → requires
  `pool_size = 64/0.025 = 2560`. `num_control` raised to 64 to match (each spike should have
  at least one matched ordinary control; symmetric K keeps the bootstrap balanced).

## 6. Batch-bank format

Live probe (unchanged design decision from the superseded diagnostic, documented there):
exact historical spike-step reconstruction would require deterministic DataLoader sampler
replay from epoch 0, impractical. `build_pool()` constructs one shared pool of real
ImageNet batches (fixed VAE-encoded latents + transport-plan noise, same seed), reused
verbatim across every checkpoint/model evaluated — NOT independently regenerated per
model, which is what makes cross-model comparison on identical tensors valid.

## 7. Current `A` calculation / parameter grouping / bootstrap (unchanged, already validated)

- `A(x,r) = ||J_theta^T (r/||r||)||` via `exact_field_vjp` (direct) / `field_vjp_none` (none),
  both independently unit-tested against double-backward / `torch.autograd.grad` references
  and finite-difference ground truth (`tests/test_fb_direct_exact_hvp.py`, 8/8 PASS, FP64 CPU,
  re-run inside every sbatch as a pre-flight gate).
- `block_groups()` / `is_backbone_group()`: backbone = `blocks.*`, `x/t/y_embedder`,
  `pos_embed` (shared architecture across direct/none); head = `energy_head` (direct) /
  `final_layer` (none), not cross-model comparable.
- `matched_interaction_bootstrap`: nonparametric bootstrap over 4 independent groups
  (direct-spike, direct-control, none-spike, none-control), resampling medians, computing
  `interaction = (Delta_source) - (Delta_recip)`. This is the CORRECTED statistic (see
  script's module-level correctness note) — kept as-is, only the input K grows.

## Conclusion / what changes in this run

1. `none` is NO LONGER held fixed — progress-matched `none` checkpoints exist at all three
   stages (Table above), essentially exact at early (0%) and near-exact at mid/late (~1%).
   This removes the biggest confound in job 38136938's design.
2. `pool_size`: 960 → 2560, `num_control`: 24 → 64, to reach K=64 genuine spikes at the SAME
   `spike_frac=0.025` criterion (Phase 1: do not relax the tail definition).
3. Everything else (canonical residual identity, `A` via exact VJP, matched-replay design,
   corrected interaction statistic) is unchanged infrastructure, now exercised at 3
   progress-matched stages instead of 1 fixed reference + 3 direct-only stages.
