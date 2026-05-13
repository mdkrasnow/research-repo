# v10 Hard-Example EqM — Variant Proposal

Per CLAUDE.md "Research Process Rules for EqM / ANM Experiments". Mechanism check filled before code.

## Variant name
`v10_hard_example_eqm`

## Hypothesis
At finite training, EqM-B/2 still has nearby points x_γ + δ where the base EqM regression target is poorly satisfied (high L_base). Training on adversarially mined high-loss perturbations — using the **same** EqM regression objective — should improve robustness of the velocity field without contradicting EqM's target geometry. This is adversarial / hard-example training (Madry 2017) applied to regression, scoped to the EqM loss.

## Failure mode addressed
Prior DG-ANM variants (v01 hinge, v02 cosine, v06 reverse-recovery, v09 Jacobian) optimized auxiliary geometric properties unrelated to the EqM target. Empirically:
- v01: hinge `ReLU(margin - ||v||)` pushes velocity norm large near data → fights c(γ) decay → kills near-manifold target.
- v02: cosine-contrastive saturated on EqM-B/2 (|v|≈220 >> |J·δ|, cos≈1 for any small δ, PGA gradient vanishes).
- v06: reverse-ODE recovery target unrelated to (ε−x)·c(γ); empirically no gain over v02.
- v09: Jacobian-Frobenius penalty flattens the velocity field — directly fights local sensitivity that transport needs; collapsed at λ=0.1, neutral at λ=0.001.

Pattern: each prior variant either contradicts EqM's target (v01, v09) or is geometry-blind (v02, v06). v10 fixes this by using the EqM loss itself as the adversarial objective.

## EqM compatibility argument
EqM trains f(x_γ) → target(x, ε, γ) where target = (ε − x) · c(γ), with c(γ) hardcoded as truncated linear, interp=0.8, multiplied by 4 (transport.py:122–126). c(γ) → 0 as γ → 1 (at data manifold). Statement 1 of the EqM paper: ||f(x_data)|| ≈ 0 asymptotically. Statement 2: spurious minima ≈ data.

v10 auxiliary target = the **same** target on a perturbed input:

```
L_total = L_base(x_γ) + λ · L_base(x_γ + δ*)
δ* = argmax_δ ||f(x_γ + δ) − target(x, ε, γ)||² , s.t. ||δ|| ≤ ε_radius
```

The auxiliary loss is the EqM regression loss applied at hard nearby points. It does **not** push velocity norm, does **not** flatten the Jacobian, does **not** introduce a cosine/hinge geometry. It asks the same question harder. Therefore it respects c(γ) decay (target shrinks near data; hard points near data have small target, large-error hard points far from data still match the data target).

This is also a faithful interpretation of Prof. Du's "try ANM" suggestion: adversarial **negative** mining on the **base** regression objective, not on imported auxiliary objectives.

## Loss definition
```python
# x_γ: latent input at flow time γ, shape [B, C, H, W]
# γ:    flow time scalar per sample
# eps:  L2 perturbation radius
# K:    PGA steps (e.g. 3)
# lr:   PGA step size (e.g. eps / K * scale)

target = (eps_noise - x) * c_gamma(γ)        # EqM regression target

# Mine δ to maximize base loss
δ = torch.zeros_like(x_γ).normal_(0, eps/2)
for _ in range(K):
    δ = δ.detach().requires_grad_()
    pred = model(x_γ + δ, γ_token)
    L_adv = ((pred - target) ** 2).mean()
    g = torch.autograd.grad(L_adv, δ)[0]
    δ = δ + lr * g.sign()                    # PGD step
    δ = project_l2_ball(δ, eps)               # back into ball

# Final losses (both forward passes participate in autograd)
pred_clean = model(x_γ, γ_token)
pred_hard  = model(x_γ + δ.detach(), γ_token)
L_clean = ((pred_clean - target) ** 2).mean()
L_hard  = ((pred_hard  - target) ** 2).mean()
L_total = L_clean + λ * L_hard
```

Hyperparameters to fix from prior work / sweep small:
- `eps_radius`: 0.3 (CIFAR-validated v02 winner)
- `K`: 3 PGA steps
- `lr`: 0.05 (CIFAR-validated)
- `λ`: 0.1, 0.3, 1.0 (small sweep on CIFAR first)
- `mine_every`: 1 (every step; per user feedback "mine every 10 is NOT right")
- warmup: 0 (no warmup; objective is EqM-aligned from step 0, no init-saturation issue like v02)

## Expected diagnostics if working
- `L_hard > L_clean` early in training (hard mining finds nonzero gap)
- `L_hard` decreases over training (model learns to satisfy target on hard points)
- `L_clean` does NOT regress materially vs vanilla (within ~5%)
- `||δ||` close to ε at the L2 boundary (active constraint, PGA not vanishing)
- velocity-field norm `||f(x_γ)||` stays in vanilla order of magnitude (no collapse, no blow-up)
- IN-1K FID ≤ 31.41 (improves on baseline) or ≤ 32 (matches within noise)

## Expected diagnostics if failing
- `L_clean` worsens materially (>10% over vanilla at matched step) → λ too aggressive
- `L_hard ≈ L_clean` always → PGA inert, ε too small or mining broken
- `L_hard` grows over training → adversarial training fails (regression collapsing)
- `||δ|| → 0` → mining gradient vanishing (target-direction degenerate)
- field norm blows up or collapses → loss imbalance with c(γ) decay
- FID degrades vs 31.41

## Minimal test
1. **CIFAR sanity (stability check only, per CIFAR rule):** 1 seed, 150ep, default v02 harness wiring. Confirm: no collapse, diagnostics finite, L_hard > L_clean early, FID within ±2 of vanilla 12.96±0.70 (not a promotion gate — just "did it break").
2. **IN-1K-256 80ep seed 0** (full paper-comparable run, gated only on CIFAR sanity not regressing): seas_gpu 4×A100, ~24h. Compare FID to 31.41 baseline.

## Promotion rule
Promote to multi-seed IN-1K run if ALL hold:
- CIFAR run completes without collapse and L_hard > L_clean for >= first 50% of steps
- IN-1K FID ≤ 31.41 + 0.5 (matches or beats baseline within noise)
- `L_clean` final value within 5% of vanilla EqM final loss at matched step
- All "Expected diagnostics if working" present in training logs

## Kill rule
Kill direction (do NOT retune indefinitely; max 1 hyperparameter retune per CLAUDE.md) if ANY hold:
- L_clean regresses >10% vs vanilla → fights EqM objective
- ||δ|| → 0 OR L_hard ≈ L_clean across the run → mining inert
- Field norm collapses (mean ||f|| < 1) or blows up (mean ||f|| > 1000)
- IN-1K FID > 35 (worse than baseline + 4) after full 80ep

## Risks / open questions
1. **PGA cost on EqM-B/2**: 3 PGA steps × per-train-step = ~4× forward passes per train iter. At 4.57 sps vanilla, mine_every=1 takes throughput to ~1.1 sps → 80ep would be ~100h. Mitigation: mine_every=4 (4× cost amortized → ~1.3× wall), accept slightly less aggressive mining. CIFAR run will measure actual throughput hit.
2. **PGA might still find ε boundary trivially** if EqM regression target is locally near-constant in δ direction near data manifold (where c(γ) is small). If so, L_hard ≈ L_clean → kill. This is the central risk; the experiment falsifies the hypothesis cleanly.
3. **EMA model**: EqM trains an EMA copy. Mine with online model (cheaper, standard adversarial training).
4. **Sign of PGA**: maximize, not minimize. Confirm direction in code review before launch.

## Diagnostics to log every N=200 steps
- L_clean, L_hard, ratio L_hard/L_clean
- mean ||δ||, std ||δ||
- mean ||f(x_γ)||, mean ||f(x_γ + δ)||
- per-step wall time vs vanilla baseline
- target norm ||target||

## Files to add/modify
- `projects/diff-EqM/experiments/dganm_variants/v10_hard_example.py` (CIFAR harness step_fn, ~50 lines)
- `projects/diff-EqM/experiments/train_imagenet.py` — add `_v10_hard_example_step()` + `--mining-flavor v10` arg
- `projects/diff-EqM/configs/variants/v10_hard_example_150ep.json` (CIFAR config)
- `projects/diff-EqM/slurm/jobs/stage_b_v10_in1k_80ep.sbatch` (IN-1K sbatch, model from `stage_b_vanilla_in1k_80ep.sbatch`)

## Postmortem trigger
Write postmortem in `documentation/v10_postmortem.md` regardless of outcome (promotion or kill), per CLAUDE.md stop-condition rule.
