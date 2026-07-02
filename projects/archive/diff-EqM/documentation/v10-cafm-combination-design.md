# v10 + CAFM Combination Design Doc

**Date**: 2026-05-19
**Status**: design locked; Phase 2 implementation pending
**Depends on**: `cafm-eqm-port-design.md` (Phase 1a) complete.

## Combined loss

```
L_G_total = L_CAFM_G(G, D, x_γ, γ) + λ_v10 · L_base(x_γ + δ*, target)
δ* = argmax_{||δ||≤ε} ||G(x_γ + δ, γ) − target||²
target = (ε − x) · c(γ)  # EqM gradient target
```

Generator update: gradient of L_G_total. Discriminator update: standard CAFM L_D (unchanged).

## Variant proposal template (per CLAUDE.md)

**Variant name**: v10 + CAFM combination

**Hypothesis**: PGD-mined hard examples on the regression target catch local failures of f that the discriminator does not, because the discriminator measures global distributional fit while PGD measures per-sample regression error. The two signals are complementary and additive.

**Failure mode addressed**: 
- CAFM-only: discriminator may converge before all per-sample regression errors are minimized. Local "blind spots" remain.
- v10-only: misses global mode-coverage / distributional issues that PGD-on-pointwise-loss can't surface.

**EqM compatibility argument**: 
- v10 base loss = ||f(x_γ + δ) − target||² is the EqM training loss evaluated at a perturbed input.
- Target is the **same** target as clean (NOT the perturbed-input target). v10 is invariance-flavored.
- Briglia 2025 argues invariance-flavored AT for diffusion can collapse. Mitigation: small λ_v10 (≤0.1) initial; equivariant fallback ready if collapse.

**Loss definition**: see top of doc.

**Hyperparameters** (locked defaults; sweep in Phase 2 retune if needed):
- λ_v10 = 0.1 (small per Briglia)
- K = 1 (FGSM-style; cheaper; try K=3 if K=1 inert per CIFAR sanity)
- ε_radius = 0.3 (CIFAR-validated from v02)
- mining_lr = 0.05
- mine_every = 1 gen update (every gen step gets v10 contribution)

**Expected diagnostics if working**:
- L_hard > L_base early in training (PGA finds nonzero hard examples).
- L_hard descends over training.
- L_base does NOT regress vs CAFM-only baseline at matched step.
- ||δ|| at L2 boundary (mining constraint active).
- FID @ 10 ep < CAFM-only FID by ≥0.3.

**Expected diagnostics if failing**:
- L_hard ≈ L_base always → PGA inert; raise ε or K.
- L_base regresses > 10% → λ_v10 too aggressive; lower to 0.03.
- ||δ|| → 0 → mining gradient vanishing.
- FID worse than CAFM-only → combination doesn't compose; fall back to either CAFM-only paper or activate equivariant v10.

**Minimal test**:
1. CIFAR sanity 150ep with v10+CAFM combined (CIFAR variant harness).
2. EqM-B/2 IN-256 80ep + 10ep CAFM + v10, seed 0.

**Promotion rule**: combination FID < CAFM-alone FID by ≥0.3 at seed 0 → promote to 3 seeds.

**Kill rule** (per CLAUDE.md 1-retune limit):
1 retune of λ_v10 ∈ {0.03, 0.3} → if all worse than CAFM-only → kill combination, revert to v10-only (CIFAR/CIFAR scaling) workshop paper.

## Where v10 fires in the CAFM training loop

```python
# CAFM loop iteration (gen update)
x_clean = data_batch
gamma = sample_gamma()  # ~U(0,1)
eps = randn_like(x_clean)
x_gamma = (1 − gamma) * eps + gamma * x_clean  # EqM mixing
target = (eps − x_clean) * c_gamma(gamma)  # EqM target

# CAFM gen loss (uses gen + discriminator + EMA)
G_out_clean = gen(x_gamma, gamma)
L_CAFM_G = lsgan_loss(D, G_out_clean, target, ...)

# v10 mining
delta = mine_hard_example(gen, x_gamma, gamma, target, K=1, eps=0.3, lr=0.05)
G_out_hard = gen(x_gamma + delta, gamma)
L_v10 = mse(G_out_hard, target)

# Combined gen loss
L_G_total = L_CAFM_G + λ_v10 * L_v10
L_G_total.backward()
optimizer_G.step()

# Discriminator update (N=16 per gen update; standard CAFM, unaffected by v10)
for _ in range(N):
    L_D = lsgan_disc_loss(D, ...)
    L_D.backward()
    optimizer_D.step()
```

**Compute cost**: v10 adds K=1 forward + 1 backward per gen step ≈ 2× forward passes for gen branch. CAFM gen branch is small fraction of total (discriminator dominates with N=16). Net wall time impact: ~5-10% overhead vs CAFM-only.

## γ-conditional λ (optional ablation per SYNTHESIS §3.4)

Try: λ_v10(γ) = λ_0 / max(c(γ), 0.1) — upweight aux loss where target magnitude is small (near data manifold).

## Diagnostics (full list)

Logged every 200 gen steps:
- L_CAFM_G (gen adversarial term)
- L_CAFM_ot (always ~0 in post-training)
- L_D (disc loss)
- L_D_cp (centering penalty)
- L_base (clean EqM regression loss)
- L_hard (v10 PGD-mined regression loss)
- ratio = L_hard / L_base
- ||δ|| mean + std
- ||G(x_γ)|| mean (generator output magnitude)
- ||target|| mean
- disc_grad_norm
- gen_grad_norm
- per-step wall time
- FID @ 5K every 1 epoch

## Files to create

- `projects/diff-EqM/experiments/cafm_eqm/v10_cafm_train.py` — modified CAFM trainer with v10 mining step
- `projects/diff-EqM/configs/cafm/v10_cafm_eqm_b2_in256.yaml` — config with v10 hyperparameters
- `projects/diff-EqM/slurm/jobs/v10_cafm_eqm_b2_in256.sbatch` — sbatch

## Open questions

1. Apply PGD before or after the gen forward pass in the CAFM loop? Order may matter for stability.
2. Backprop through the PGD inner loop, or detach δ? **Detach δ** (standard adversarial training practice — δ is a found point, model gradients don't flow back through PGD).
3. Mining at the EMA model or current model? **Current model** (cheaper; standard).
4. Sign-of-PGA: maximize L_base, so δ ← δ + lr · grad.sign() (ascent). Confirm direction in code review.
