# Stage A.5 Audit — CIFAR-10 Reproducibility Bug

## TL;DR
Our CIFAR-10 80-epoch vanilla EqM run got **FID 497.55**; the EqM paper reports **FID 3.36** on the same dataset. The root cause is almost certainly that **we used the wrong architecture**: we trained a transformer (EqM-S/2, patch=4), but the paper's CIFAR result uses a **U-Net** from a completely different codebase. Secondary concerns (sampler hyperparams) stack on top of that.

This is a reproducibility bug, not a scaling issue. Fix this before any Stage B/C compute.

## Evidence

### What the paper did (Appendix B.1, verified via web fetch)
- **Architecture**: "non-transformer architectures…U-Net"
- **Codebase**: Flow Matching (Lipman et al., 2024) — facebookresearch/flow_matching
- **Hyperparameters**: "same hyper-parameters as the original codebase"
- **Sampler**: kept `a`, `b` hyperparameters of `c(γ)` from ImageNet experiments, searched over gradient multiplier `λ`, step size 1
- **Result**: FID 3.36 (vs. FM 2.09, FM-uncond 3.96)

### What we did
- `projects/diff-EqM/configs/full_80epoch_vanilla.json` + `full_80epoch_vanilla_paperparams.json`:
  - Model: `patch_size=4, hidden_size=384, depth=12, num_heads=6` → **EqM-S/2 transformer**
  - Training: 80 epochs, bs=256, lr=1e-4 (paperparams) or 5e-5
  - Data: CIFAR-10 direct pixel space, no VAE
- `projects/diff-EqM/eqm-upstream/`: contains **only** the SiT-style transformer models (EqM_S/B/L/XL in `models.py`). No U-Net code. The upstream we vendored is the ImageNet-focused branch.
- Training entry point `experiments/train_dganm.py` literally documents: "Architecture: EqM-S/2 (depth=12, hidden=384, patch=2, heads=6)". It was designed for a CIFAR proof-of-concept with a small transformer, not for matching the paper's CIFAR number.

### Why this produces catastrophic FID
- A transformer with ~6M params trained for 80 epochs on 32×32 CIFAR with patch_size=4 (64 tokens) is far from converged. The paper implicitly knew transformers are a poor choice at this resolution/dataset and used U-Net instead.
- CIFAR-10 SOTA is reached by UNet-based models (NCSN++, DDPM, FM UNet). Transformers at CIFAR-32 without a VAE have always been weak in the literature.
- Both our vanilla and DG-ANM runs landed at FID ~497 — a pathological floor consistent with a wrong-architecture stack, not a method difference.

## Ranked root causes

| # | Cause | Evidence | Confidence |
|---|-------|----------|-----------|
| 1 | **Wrong architecture**: transformer instead of U-Net | Paper explicitly says "non-transformer…U-Net"; our configs are EqM-S/2 | HIGH |
| 2 | **Wrong codebase**: our upstream is the ImageNet branch (SiT-style models only) | `eqm-upstream/models.py` has no UNet; paper uses Flow Matching repo for CIFAR | HIGH |
| 3 | **Sampler**: we may not be using the same `a`, `b`, `λ`, step-size-1 GD sampler the paper used on CIFAR | Our `sample_gd.py` exists but unclear if CIFAR configs invoked it with paper-matching hyperparams | MED (separable) |
| 4 | **Data pipeline**: normalization, cropping, flip augmentation may differ from FM's CIFAR pipeline | Not inspected yet | LOW (unlikely to cause 100× FID gap) |
| 5 | **Eval pipeline**: FID reference stats, number of samples, cleanfid vs. pytorch-fid vs. TF-FID | `fid_results.tsv` shows 50K samples but eval implementation not verified against FM's | LOW-MED |

Causes #1 and #2 alone are sufficient to explain FID 497 vs. 3.36.

## Fix plan

### Option A (recommended): vendor the Flow Matching CIFAR recipe
1. Clone `facebookresearch/flow_matching` into `projects/diff-EqM/fm-upstream/` (or similar).
2. Identify their CIFAR-10 U-Net training entry point, config, and hyperparameters.
3. Port the EqM objective onto their U-Net training loop — swap the FM loss for EqM's `c(γ)`-weighted target. Keep everything else identical (UNet, optimizer, data pipeline, eval).
4. Run vanilla EqM on CIFAR-10 for the same number of steps as the paper's baseline (which they don't specify — match FM's default, likely ~500k–1M steps).
5. **Exit criterion**: vanilla EqM FID within ~0.3 of 3.36 on 50K samples, paper's sampler settings.

### Option B: confirm the architecture hypothesis with a cheap smoke test first
Before committing to Option A's port, run one cheap test to confirm it's really the architecture:
1. Take an off-the-shelf DDPM U-Net implementation (e.g., Ho et al. 2020 reference, ~50M params).
2. Train it on CIFAR-10 with plain FM loss for, say, 200 epochs.
3. If you get FID in the 5–10 range, the architecture is confirmed as the dominant factor and Option A is justified. If not, something else is wrong (data, eval).

Option B is maybe 1–2 days; Option A is ~1 week. B is low-regret insurance.

### Non-options (do NOT do)
- **Keep training the current transformer for more epochs**: the FID floor at 497 suggests a model that cannot converge to a reasonable generative distribution on 32×32 pixel space at this size. More compute will not help.
- **Report CIFAR results using the broken config**: would be scientifically invalid.

## Implications for the publishability plan
- **Stage A.5 is now concretely scoped**: port FM's CIFAR U-Net + EqM loss, match paper's 3.36 with vanilla, then Stage C secondary result runs DG-ANM on that stack.
- **Stage B (ImageNet-256 B/2)**: our existing `eqm-upstream/` *is* the right codebase for ImageNet. The transformer models there are what the paper used for ImageNet. Our IN-100 80ep result (vanilla 121.24, DG-ANM 112.58) used this stack, so Stage B infrastructure is fundamentally closer to correct than our CIFAR infrastructure. Good news: Stage B is not blocked by the CIFAR bug.
- **Revised ordering**: since Stage B uses a different (working) stack, the Stage A.5 CIFAR fix and the Stage B ImageNet-256 baseline can proceed in parallel. The CIFAR fix is only blocking the Stage C secondary result.

## Recommended immediate actions
1. Stop all CIFAR experiments in their current form.
2. Decide: Option A (port FM UNet) vs. Option B (DDPM UNet smoke test) vs. defer CIFAR entirely and go all-in on ImageNet-256 for Stage B.
3. Update queue.md and pipeline.json's `next_action` with the chosen path.
