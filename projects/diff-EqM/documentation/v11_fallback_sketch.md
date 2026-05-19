# v11 Fallback Sketches

**Date**: 2026-05-19
**Status**: design locked. Activate if Phase 0 CIFAR sanity OR Phase 2 IN-1K combined gate fails.

Priority order set by SYNTHESIS.md §3.1: v10-equivariant is primary fallback because it directly addresses Briglia 2025's mechanism threat (invariance fails for diffusion AT). Other variants secondary.

---

## v11 — v10-equivariant (PRIMARY FALLBACK)

**Hypothesis**: v10's invariance-flavored loss (`||f(x+δ) − target||²` with fixed target) may collapse at non-trivial λ per Briglia 2025. An equivariance-flavored alternative asks the output to track the input perturbation linearly, which Briglia shows is the SAFE form for diffusion AT.

**Loss**:
```
L_v11_aux = ||f(x_γ + δ) − [f(x_γ).detach() + J_target · δ]||²
```
where `J_target = ∂target/∂x_γ`. Since target = (ε − x) · c(γ), and (x, ε) are fixed for the perturbation, `∂target/∂x_γ = 0` (target doesn't depend on x_γ for a given (x, ε, γ)).

This **simplifies to**:
```
L_v11_aux = ||f(x_γ + δ) − f(x_γ).detach()||²
```

i.e., enforce smoothness: nearby points produce similar outputs.

**EqM compatibility argument**:
- Briglia 2025 prescribes this form for diffusion AT.
- v11 enforces local Lipschitz behavior of f (matches Wang+Du's L-smooth energy assumption for convergence guarantee).
- Does NOT contradict c(γ) decay: model can still have c(γ)-modulated magnitude, just smooth in δ direction.

**Risk**:
- Pure Lipschitz regularization (no target anchor) may hurt generation quality (we tried v09 Jacobian penalty which failed).
- Mitigation: combine with v10 NOT as replacement but as augmentation: `L = L_base + λ_v10 * L_v10_aux + λ_v11 * L_v11_aux`. Two-sided regularization.

**Promotion / kill rules**: same Phase 0 CIFAR sanity gates as v10.

---

## v12 — γ-weighted hard mining

**Hypothesis**: at small γ (near data, c(γ) small) target is small → mining produces weak signal; at large γ (near noise) target is large → mining dominates. Reweight aux loss inversely with c(γ).

**Loss**:
```
L_v12 = L_base(x_γ) + λ · w(γ) · L_base(x_γ + δ*)
w(γ) = 1 / max(c(γ), 0.1)
```

**Rationale**: upweight aux loss where the regression task is informative. Briglia's `λ_t = c · σ(t)^{-1}` is the same idea applied to diffusion noise scheduler.

**Risk**: at γ near data, the model is supposed to predict ~0; aux loss there may not have much signal to find. Empirically uncertain.

---

## v13 — stale-EMA PGA (two-time-scale AT)

**Hypothesis**: mining against the EMA model (slower-moving target) produces more stable adversarial signal than mining against the current model. Standard two-time-scale AT (TTUR-style for the adversarial component).

**Loss**:
```
δ* = argmax_{||δ||≤ε} ||f_ema(x_γ + δ) − target||²
L_v13 = L_base(x_γ) + λ · ||f(x_γ + δ*) − target||²
```

Mining happens in EMA-model parameter space; training updates the current model.

**Risk**: EMA is conservative; mining may find easier examples than current-model mining → less useful signal.

---

## v14 — velocity-correlation hard mining (direction-sensitive)

**Hypothesis**: replace L2-regression objective in PGA with a direction-sensitive metric. Direction errors matter for sample quality more than magnitude errors.

**Loss**:
```
δ* = argmin_{||δ||≤ε} cos(f(x_γ + δ), target)
L_v14 = L_base(x_γ) + λ · max(0, m − cos(f(x_γ + δ*), target))
```

**Risk**: cosine objectives saturate (v01, v02 history). May not produce useful gradient on EqM-B/2.

---

## Decision tree for activation

| Trigger | Activate |
|---|---|
| v10 CIFAR sanity collapses (NaN, FID >> baseline) | v11 (equivariant) |
| v10 mining inert (L_hard ≈ L_base, ||δ|| → 0) | v12 (γ-weighted) |
| v10+CAFM combination doesn't compound (FID ≈ CAFM-only) | v13 (stale-EMA) |
| v10 magnitude-only signal (works on norm but not direction) | v14 (cosine) |

Maximum 1 fallback activation per Phase. Per CLAUDE.md: 1 retune of any failing direction before kill.
