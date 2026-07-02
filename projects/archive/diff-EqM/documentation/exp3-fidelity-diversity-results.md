# Exp 3 — Fidelity-Diversity & Mode Coverage — RESULTS (2026-06-05)

**Status: ✅ DONE. Verdict: STRONG_SUCCESS (no diversity tax).**

## Question (pre-registered)
Does ANM's IN-1K FID gain *preserve/improve* diversity, class coverage, and mode
coverage — or is it just sharpening samples (fidelity bought with diversity)?
FID alone is **not** sufficient evidence; verdict weighs recall/coverage + per-class
histogram + weak-class bottom-quartile.

## Setup (parity-controlled)
- Arms: **vanilla EqM** (`stage_b_vanilla_in1k_80ep_seed0/.../0380000.pt`, FID-trusted 31.41)
  vs **ANM EqM v10 λ=0.3** (`imagenet1k_80ep_v10_b2_lambda03_seed0/.../final.pt`).
- IDENTICAL sampler/NFE/step/EMA/CFG/VAE/labels/seeds: gd, 250 steps, stepsize 0.003,
  cfg 1.0, EMA weights, sd-vae-ft-ema, EqM-B/2, 1000 classes.
- Shared deterministic schedule (balanced labels + per-sample seeds), hash
  `83a8ede763e1b318` — both arms identical z and y at every index.
- Fixed seeded ImageNet-train reference (one cached subsample, controls the
  "re-shuffled reference" confound the trusted FID sbatch has).
- Features: pytorch_fid InceptionV3 pool3 (2048-d) — same as trusted FID.
- PRDC: vendored pure-numpy (Naeem et al. 2020). KID: poly-kernel MMD.
- Both arms scored on the **same 49996 sample ids** (4 zero-byte vanilla PNGs from
  an interrupted file move dropped from both arms; immaterial at 50K scale).

## Results
| metric | vanilla | ANM (λ0.3) | Δ | reads as |
|---|---|---|---|---|
| FID ↓ | 31.27 | **26.88** | −4.38 | 95% CIs disjoint (31.64–32.28 vs 27.27–27.85) |
| KID ↓ | 0.03156 | 0.02591 | −0.0057 | confirms FID |
| precision ↑ | 0.581 | 0.604 | +0.023 | fidelity up |
| **recall ↑** | 0.7185 | 0.7193 | +0.0008 | **diversity FLAT — no tax** |
| density ↑ | 0.433 | 0.477 | +0.044 | local density up |
| **coverage ↑** | 0.443 | 0.515 | +0.072 | **mode coverage up big** |
| bottom-quartile (weak-class) FID ↓ | 62.80 | 57.19 | −5.61 | weak classes gain MORE |
| classifier TV→requested ↓ | 0.181 | 0.162 | −0.019 | better class balance |
| conditional top-1 ↑ | 0.433 | 0.483 | +0.050 | more on-class |
| frac classes ANM better (feat dist) | — | 0.913 | — | 91% of classes improve |

## Interpretation
- **No diversity tax.** Recall is the diversity axis; it is flat. Coverage and
  density both rise. The FID gain is therefore quality, not mode-dropping.
- **Helps where it should.** Weak-class bottom-quartile FID improves MORE than the
  mean (−5.61 vs −4.38), and 91% of classes improve — broad, not one lucky bucket.
  Consistent with hard-example mining targeting the model's weak regions.
- **Better class adherence.** classifier TV-to-requested down, conditional top-1 up:
  generated images land in their requested class more often.

## Paper use
Closes the obvious reviewer attack on the FID claim ("you just sharpened samples /
lost variety"). Pairs with Exp 1 (sampler/NFE robustness) and Exp 2 (off-trajectory
field robustness): ANM improves quality, holds diversity, is robust to sampler budget,
and the mechanism is measurable.

## Caveat
Single seed at B/2. Phase 2 (3-seed Welch t, p<0.05, mean ≥1 FID gain) still required
for the paper-final claim. This is the per-seed evidence the no-diversity-tax story
rests on.

## Artifacts
- `results/exp3_metrics_out/` — aggregate_metrics.{json,csv}, class_metrics.csv,
  delta_class_metrics.csv, classifier_histogram.csv, samples_manifest.csv,
  features_{vanilla,anm}.npy(+.stems.npy), plots/, README.md
- Generated images (holylabs): `mkrasnow_eqm/exp3/full_lambda03_vs_vanilla/gen/{vanilla,anm}/`
  (50000 PNGs each), reference cache `mkrasnow_eqm/exp3/reference/`
- Code: `experiments/exp3_fidelity_diversity/`
- Jobs: 18964347 (reference, exit 0) + 18964349 (anm gen, 50K PNGs) + 19120911 (metrics, exit 0)
- TSV row: `results_variants.tsv` round=`exp3_fidelity_diversity` STRONG_SUCCESS
- PI update: `pi-updates.md` 2026-06-05 section (drafted, user-send only)

## Infra cascade (5 failures before clean run — see debugging.md 2026-06-05 entry)
home quota exit-53 → moved exp3 to holylabs+symlink; reference deleted → rebuilt;
gen log-ENOSPC (cosmetic, data already written); holylabs ydu_lab GROUP quota EDQUOT
→ split metrics read(holylabs)/write(home); feats/stems misalignment IndexError (4
corrupt PNGs) → fixed extractor-aligned stems + score-on-common-id-set.
