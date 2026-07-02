# Experiment 3 — Fidelity-Diversity & Mode Coverage

> **✅ DONE 2026-06-05 — VERDICT: STRONG_SUCCESS (no diversity tax).**
> ANM l03 FID 26.88 vs vanilla 31.27 (−4.38, disjoint 95% CIs) on 49996 identical ids.
> **recall FLAT 0.7185→0.7193** (diversity preserved), coverage +0.072, density +0.044,
> weak-class bottom-quartile FID −5.61, 91% classes improve. Caveat: single seed.
> Full writeup: `documentation/exp3-fidelity-diversity-results.md`. Data: `results/exp3_metrics_out/`.

Does ANM's FID gain (IN-1K-256, EqM-B/2, 80ep: vanilla **31.41** → ANM λ=0.3 **27.09**)
**preserve or improve diversity / class coverage / mode coverage**, or does ANM
merely sharpen samples while dropping modes? **FID alone is not the answer.**

Primary arm = ANM **λ=0.3** (best FID). Control = vanilla.

## Design (confound control)

Vanilla and ANM share an **identical** schedule: same balanced label at index `i`,
same per-sample noise seed `base_seed+i`, same sampler (gd), NFE 250, step 0.003,
cfg 1.0, EMA weights, VAE, sample count. Schedule hash is checked equal across arms.

Single feature space for FID/KID/PRDC = **pytorch_fid InceptionV3 pool3 2048-d**
(identical to the trusted FID run that produced 27.09/31.41). PRDC is vendored
(`prdc_vendored.py`, Naeem et al. 2020) so it runs on those same features with no
pip/network dependency.

Reference is **fixed and cached once** (`build_reference.py`) — the trusted FID
sbatch re-shuffles its reference every run; Exp3 does not.

## Files

| file | role |
|---|---|
| `schedule.py` | build/save/load balanced label + per-sample seed schedule; `schedule_hash` |
| `sample_scheduled.py` | DDP generation, per-index seed+label, `{i:06d}.png` + manifest, NO filtering (fork of `eqm-upstream/sample_gd.py`) |
| `build_reference.py` | one-time fixed reference: agg feats, per-class μ/Σ, real classifier histogram |
| `features.py` | pytorch_fid Inception feats; resnet50 IMAGENET1K_V2 preds; Inception Score |
| `prdc_vendored.py` | pure-numpy precision/recall/density/coverage |
| `metrics.py` | FID, KID, PRDC, bootstrap CIs (stratified), per-class, classifier histogram, weak-class, verdict |
| `eval_fidelity_diversity.py` | orchestrator → CSV/JSON + deltas + verdict + README |
| `plots.py` | required plots |
| `submit_exp3.sh` | build schedule → rsync → submit reference→generate×2→metrics chain |
| `test_metrics_local.py` | CPU plumbing test of the metric layer (no GPU/cluster) |

SLURM: `slurm/jobs/exp3_reference_features.sbatch`, `exp3_generate.sbatch`, `exp3_metrics.sbatch`.

## Commands

Local plumbing test (no GPU):
```bash
python projects/diff-EqM/experiments/exp3_fidelity_diversity/test_metrics_local.py
```

Smoke (cluster, ~1000 samples, PLUMBING ONLY — numbers not valid):
```bash
bash projects/diff-EqM/experiments/exp3_fidelity_diversity/submit_exp3.sh --smoke \
  --vanilla-ckpt projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt \
  --anm-ckpt    projects/diff-EqM/results/imagenet1k_80ep_v10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints \
  --out exp3/smoke
```

Full 50K run (λ=0.3 best ANM arm vs vanilla):
```bash
bash projects/diff-EqM/experiments/exp3_fidelity_diversity/submit_exp3.sh \
  --vanilla-ckpt projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt \
  --anm-ckpt    projects/diff-EqM/results/imagenet1k_80ep_v10_b2_lambda03_seed0 \
  --num-classes 1000 --samples-per-class 50 --out exp3/full_lambda03_vs_vanilla
```
Chain: reference (~build once) → generate vanilla + generate anm (4×A100 seas_gpu,
~50K gd samples each ≈ same wall as the FID-eval runs) → metrics (1 GPU).
**Add an `active_runs` entry per submitted job to `pipeline.json` (job-tracking protocol).**

## Verdict logic (printed + in README/json)

- **success**: FID↓ AND KID↓ AND recall ≥ vanilla−max(0.005, 1 bootstrap SE) AND
  coverage ≥ vanilla−max(0.005, 1 SE) AND classifier TV-to-requested not worse by
  >0.02 AND conditional top-1 not down >0.01.
- **strong_success**: success + ≥2 of {recall↑, coverage↑, bottom-quartile FID↓, ≥55% classes improve}.
- **failure**: FID↓ but a diversity/coverage/class guard fails (sharpening-while-narrowing).
- **ambiguous**: FID and KID disagree, or partial → repeat with new seed schedule / larger budget.

## Reliability warnings

- Smoke FID/KID/PRDC are pipeline checks only.
- Per-class FID @ 50/class is NOISY (flagged); lead with `feature_distance_class` and
  pooled bottom-quartile (~12.5K samples).
- Bootstrap FID CI uses resampling-with-replacement and is mildly biased upward at
  small N; reliable at 50K. Diversity CIs (recall/coverage) are the gating ones.
- Classifier histogram interpreted vs the cached real-reference histogram.
- No sample filtering / rejection sampling.
- Never compare these IN-1K numbers to CIFAR FID (different harness).
