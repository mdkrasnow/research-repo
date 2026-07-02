# CAFM → EqM Port Design Doc

**Date**: 2026-05-19
**Status**: design locked; Phase 1a implementation pending
**Owner**: claude

## Goal
Adapt Lin et al. CAFM (arxiv 2604.11521) to post-train our vanilla EqM-B/2 80ep checkpoint (FID 31.41). Reuse Lin's MIT-licensed code where possible; adapt for EqM's no-time-conditioning + c(γ) target geometry.

## What CAFM does on SiT (recap)

From `external/Adversarial-Flow-Models/train_continuous_adversarial_flow_imagenet.py`:
- **Generator (gen)** = pretrained flow matching model (SiT-XL/2, 1400ep ckpt).
- **Discriminator (dis)** = DiscriminatorJVP at `models/cafm/jvp/discriminator.py`. JVP-based, takes (x_t, t, ẋ_t, ṫ).
- **EMA** copy of generator.
- **VAE** for latent-space training.
- Losses (per arxiv extraction):
  ```
  L_G = E[f(D_jvp(x_t, t, G(x_t,t), T)) − f(D_jvp(x_t, t, v̄_t, T))] + λ_ot · ||G(x_t,t)||²
  L_D = E[f(D_jvp(x_t, t, v̄_t, T)) − f(D_jvp(x_t, t, G(x_t,t), T))] + λ_cp · D(x_t,t)²
  f(a,b) = (a−1)² + (b+1)²   # least-squares GAN
  ```
- N=16 discriminator updates per generator update.
- 10 epochs post-training.
- Optimizer: Adam β=(0, 0.95), LR=1e-5.
- λ_cp=0.001, λ_ot=0 (post-training).

## Compatibility analysis EqM-B/2 vs SiT

| CAFM component | SiT behavior | EqM-B/2 behavior | Action |
|---|---|---|---|
| Generator backbone | SiT-XL/2 DiT with time conditioning t∈[0,1] | EqM-B/2 SiT backbone with **t=0 fixed** | Generator is identical architecture; we just stop passing t |
| Pretrained ckpt | SiT-XL/2 1400ep public ckpt | Our vanilla EqM-B/2 80ep ckpt (FID 31.41) | Use ours as drop-in pretrained gen |
| Velocity field target | u_t(x) — flow matching velocity | `(ε − x) · c(γ)` — EqM gradient field | **Different target structure**. CAFM loss compares discriminator on `G(x_t,t)` vs `v̄_t`. For EqM: `v̄_γ = (ε − x) · c(γ)` |
| Discriminator input | (x_t, t, ẋ_t, ṫ) JVP form | EqM has no t; ẋ_t/ṫ undefined in the same way | **ADAPT**: Use γ instead of t. Provide γ to discriminator. Use JVP over γ direction. |
| γ-sampling | t ~ U(0,1) | γ ~ U(0,1) | Identical structure |
| OT regularization λ_ot · ||G||² | Penalize generator magnitude | EqM target has c(γ)-decay built in (target → 0 at γ=1) | λ_ot=0 for post-training (matches CAFM default); EqM's c(γ) handles magnitude naturally |
| Centering λ_cp · D² | Numerical stability | Same | Keep |
| N=16 disc updates/gen | Discriminator updates per gen step | Same compute pattern | Keep |
| EMA | Standard EMA copy | Same | Keep |
| VAE latent space | 32×32×4 latents | EqM also operates in latent (check) | **VERIFY** — is our vanilla EqM-B/2 latent? If pixel-space, need VAE pipeline |
| Sampler at eval | Euler ODE with NFE | EqM uses NAG-GD or GD on landscape | **DIFFERENT** — keep EqM's native sampler for FID eval |

## Required new code

### File: `projects/diff-EqM/experiments/cafm_eqm/model_wrapper.py`
Wraps our EqM-B/2 model to the interface CAFM trainer expects:
```python
class EqMGeneratorWrapper(nn.Module):
    def __init__(self, eqm_model):
        self.eqm = eqm_model  # SiT backbone, t=0 fixed
    def forward(self, x_gamma, gamma):
        # CAFM expects G(x_t, t) -> velocity prediction
        # EqM: f(x_gamma) returns gradient field; t arg ignored
        # We pass gamma=0 (EqM convention) regardless
        return self.eqm(x_gamma, torch.zeros_like(gamma))
```

### File: `projects/diff-EqM/experiments/cafm_eqm/discriminator_gamma.py`
γ-conditional discriminator (adapted from Lin's `DiscriminatorJVP`):
```python
# Same DiT-like architecture, but condition on gamma (not t)
# JVP direction: tangent = velocity vector (G output vs target v̄)
class DiscriminatorGammaJVP(nn.Module):
    def __init__(self, dit_config):
        ...
    def forward(self, x_gamma, gamma, v_primal, v_tangent):
        # JVP through DiT with primal (x_gamma, gamma) and tangent (v_primal, 0)
        ...
```

### File: `projects/diff-EqM/experiments/cafm_eqm/cafm_eqm_train.py`
Top-level CAFM post-training entrypoint for EqM. Adapts Lin's `train_continuous_adversarial_flow_imagenet.py`:
- Replace `models.cafm.jvp.discriminator.DiscriminatorJVP` import with our `discriminator_gamma.DiscriminatorGammaJVP`.
- Replace generator with `EqMGeneratorWrapper`.
- Replace t-sampling with γ-sampling (already identical structure).
- Replace flow-matching velocity target `v̄_t` with EqM gradient target `(ε − x) · c(γ)`.

### File: `projects/diff-EqM/configs/cafm/eqm_b2_in256_cafm.yaml`
OmegaConf config based on `external/Adversarial-Flow-Models/configs/train/cafm/train_cafm_sit.yaml`:
- gen: load `stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt`
- dis: DiscriminatorGammaJVP with B/2-matched DiT config
- λ_cp=0.001, λ_ot=0
- N=16
- 10 epochs

### File: `projects/diff-EqM/slurm/jobs/cafm_eqm_b2_in256.sbatch`
Clone `stage_b_vanilla_in1k_80ep.sbatch`. Modify:
- Job name: `cafm-eqm-b2-in256-10ep`
- Time: 48h (10ep CAFM × N=16 ≈ ~30h estimated on 4×A100)
- Entry: `python projects/diff-EqM/experiments/cafm_eqm/cafm_eqm_train.py --config projects/diff-EqM/configs/cafm/eqm_b2_in256_cafm.yaml`
- Partition: `seas_gpu` (80G A100s required for JVP memory)

## Diagnostics to log every N=200 gen steps

- `loss_G_adv` — adversarial term for generator
- `loss_G_ot` — OT regularization (should be ~0 post-training)
- `loss_D_adv` — discriminator adversarial loss
- `loss_D_cp` — centering penalty
- `disc_grad_norm` — discriminator gradient norm (instability indicator)
- `gen_grad_norm` — generator gradient norm
- `v_norm_real` — mean ||v̄|| (EqM target magnitude)
- `v_norm_gen` — mean ||G(x_γ)|| (generator output)
- `gamma_distribution_hist` — sanity check γ-sampling
- `FID@5K` every 1 epoch

## Open questions

1. ~~Is our vanilla EqM training in pixel space or latent space?~~ **RESOLVED 2026-05-19**: EqM IS latent-space, uses `stabilityai/sd-vae-ft-{ema|mse}`, 4×32×32 latents at 256-resolution. CAFM port directly compatible (CAFM also uses same SD VAE). No additional VAE pipeline needed.
2. **JVP through `t=0`-fixed DiT**: when t is constant, ṫ = 0 in JVP. Does this degenerate the JVP computation? May need to drop ṫ direction entirely and condition on γ only.
3. **Does γ-conditioning the discriminator capture enough information?** Alternative: γ-unconditional discriminator (simpler, may be enough since CAFM's signal comes from JVP through input space).
4. **CAFM uses VAE-encoded latents.** Our EqM training (per sbatch) loads from `IN1K_TRAIN` and seems to do pixel-space (need to verify). If pixel, need to encode through VAE first, OR retrain EqM-B/2 in latent space (~24h cost).
5. **Pretrained checkpoint compatibility**: our ckpt is at step 380000 (epoch 76). Acceptable; CAFM post-train resumes from there.

## Phase 1a plan refinement

1. **Day 1-2** (Jun 9-10): verify EqM pixel-vs-latent; if pixel, set up VAE encoding pipeline.
2. **Day 3-4**: implement `EqMGeneratorWrapper` + `DiscriminatorGammaJVP`.
3. **Day 5-6**: implement CAFM-EqM training loop. 100-step smoke locally (CPU) — check loss flow + shapes.
4. **Day 7-10**: 1-epoch CAFM-EqM smoke on cluster. Verify losses finite, FID rough estimate.
5. **Day 11-14**: full 10-epoch post-training seed 0 → Phase 1b deliverable.

## Phase 1a exit gate (PASS = ALL)

- Training runs to completion 1 epoch.
- All four losses finite (gen adv, gen OT, disc adv, disc cp).
- Discriminator loss not collapsing to 0 (gen domination indicator).
- Generator loss not blowing up.
- Rough FID estimate < vanilla 31.41 (post-training should help even at 1 epoch).
- No NaN.

## Risks specific to port

- VAE pipeline mismatch (if EqM pixel-space): adds 1 week.
- γ-conditioning insufficient: fall back to γ-unconditional discriminator.
- JVP degenerate at t=0: drop ṫ direction; use γ-tangent only.
- N=16 discriminator updates expensive on B/2: try N=8, document degradation if any.
