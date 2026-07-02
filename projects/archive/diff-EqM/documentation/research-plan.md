# Research Plan — DG-ANM for EqM

## One-line Summary
Find a formulation of geometry-guided adversarial negative mining that actually helps EqM training. The original formulation (PGA mining at t=1 + feature-normal projector + margin-hinge on `‖field‖`) loses to vanilla by ~2× FID on CIFAR-10 and is provably inert at realistic margins. We now search the *formulation* space, not the hyperparameter space.

## Current state (2026-04-23)
- **Vanilla EqM CIFAR 150ep 10K-FID 3-seed**: 12.54 ± 1.15 (from `cifar_seed_study_COMPLETE`).
- **DG-ANM v01 CIFAR 150ep 10K-FID 3-seed**: 21.66 ± 9.95 — worse in mean and 8× higher variance (from `cifar_dganm_resume_*`).
- **Diagnostic (`diag_dganm_signal.py`)**: with fresh UNet at near-data perturbed inputs, median `‖field‖` ≈ 47 and min ≈ 30. `ReLU(margin − ‖field‖)` at margin=5 produces gradient norm = 0 exactly. Even at margin=50 the hinge fires on only a fraction of samples and saturates within a few epochs of training — hence the observed `neg_loss ≈ 0` on 99%+ of logged epochs in the 150ep runs.
- **Prior IN-100 "improvement" (DG-ANM 112.58 vs vanilla 121.24)**: now considered a single-seed artifact. 3-seed CIFAR seed study shows DG-ANM variance alone (σ ~10 FID) is larger than the prior IN-100 gap.

## Why the v01 formulation fails (literature synthesis)

**Lit review** (see memory: 2026-04-23 lit-review, summary below):

1. **Margin-hinge on velocity norm is structurally wrong.** `‖v_θ(x,t)‖` diverges as t → 0 (score blows up near data); no fixed margin can simultaneously produce signal across the t-schedule without saturating. Evidence: "What is AT for DMs?" (arXiv 2505.21742) argues DMs require *equivariance*, not invariance, of the velocity under perturbations; Du 2021 (ICML) and VeCoR 2025 both use *relative* (contrastive) objectives specifically to avoid this.
2. **Mining at hardcoded t=1 is wrong.** At t=1 the marginal is ≈ pure Gaussian; PGA on `‖field‖` there finds a region the model handles trivially. Luo et al. 2023 ("Diffusion Contrastive Divergences", arXiv 2307.01668) show noise-conditional negatives (t ~ U(0,1)) dominate fixed-t variants.
3. **Pixel-space PGD ignores the perceptual manifold.** Metric Flow Matching (Kapusniak NeurIPS 2024) and VeCoR (arXiv 2511.18942) show semantically-preserving perturbations beat raw L∞ PGD by large margins.
4. **The explicit feature-normal/tangent projector may be redundant.** Pidstrigach (NeurIPS 2022) proves learned scores already decompose into tangential and normal components implicitly.

## Variant search (current autoresearch)

Instead of sweeping scalars, the autoresearch loop now proposes and benchmarks *code variants* — each a concrete reformulation motivated by a specific paper. Each variant is a drop-in `experiments/dganm_variants/v{NN}_<name>.py` with a shared harness (`_common.py`, `run_variant.py`, `variant_pilot.sbatch`).

| Variant | Change from v01 | Citation |
|---|---|---|
| v00_vanilla | No mining. Sanity baseline. | — |
| v01_current | Unchanged; frozen reference to beat. | — |
| v02_score_repulsion | Two-sided cosine contrastive on velocity; drops hinge. | Jiang 2025 (VeCoR), 2505.21742 |
| v03_noised_negatives | Mine at `t ~ U(0,1)`, not t=1. Keeps hinge. | Luo 2023 (DCD) |
| v04_ebm_contrastive | InfoNCE softplus on velocity-norm energy. Relative, cannot saturate. | Du 2021 (CD), DCD 2023 |
| v05_drop_geometry | v01 minus P_N/P_T projector. Pure ablation. | Pidstrigach NeurIPS 2022 |
| v06_diffusion_recovery | Short reverse-ODE reconstruction error at perturbed x_t. Self-calibrating. | Kim 2024 (CTM), Kong 2024 (ACT) |

Two additional variants (not yet implemented) identified by the lit review:
- **v07_fokker_planck_consistency** (Lai 2023 FP-Diffusion, ICML): drop mined negatives entirely; add FP residual as auxiliary loss.
- **v08_latent_negative_mining**: mine in frozen encoder latent space (DINOv2 or backbone features) rather than pixel space.

## Proxy and gates

- **Proxy**: CIFAR-10, 25 epochs, 5K FID. ~1 h/seed on 1 A100.
- **Pilot promotion**: single-seed pilot beats v00_vanilla by ≥1 FID → run 2 more seeds at pilot.
- **Pilot elimination**: single-seed pilot >5 FID worse than v00_vanilla → eliminate (do not burn more seeds).
- **Confirmation (Stage A.5)**: promoted variant × 3 seeds × 150 epochs × 10K FID; mean must beat v01_current by ≥1 FID AND be within 3 FID of vanilla.
- **Stage B unlock**: only after Stage A.5 confirmation passes.

## Key Files (new)
- `program.md` — governance for variant search (IMMUTABLE during loop)
- `experiments/dganm_variants/_common.py` — shared harness (IMMUTABLE)
- `experiments/dganm_variants/v*.py` — variant trainers (each a drop-in)
- `experiments/run_variant.py` — CLI dispatcher (IMMUTABLE)
- `configs/variants/v*.json` — per-variant configs
- `slurm/jobs/variant_pilot.sbatch` — generic pilot runner (IMMUTABLE)
- `results_variants.tsv` — new results log (TSV)
- `results_hp_archive.tsv` — archived HP-tuning history

## Key Files (legacy, preserved but not active)
- `experiments/train_cifar_dganm_unet.py` — original v01 trainer; frozen reference
- `experiments/train_cifar_eqm_unet.py` — original vanilla trainer; frozen reference
- `experiments/train_imagenet.py` — IN-100 trainer for Stage B (gated on Stage A.5)
- `experiments/diag_dganm_signal.py` — the diagnostic that proved v01 is inert
