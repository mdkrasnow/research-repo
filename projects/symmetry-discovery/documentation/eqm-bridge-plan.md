# EqM Bridge Plan — lifting the symmetry-discovery recipe onto real diff-EqM

Status: DRAFT (2026-06-04), written during rung-16 run. Finalize after rung 16 result + before any
cluster run. This is the rung-17 deliverable: the smallest real-EqM test of the toy recipe.

## The validated toy recipe (rungs 9–14, ≥15 caveated)
Near-oracle unsupervised latent symmetry discovery =
1. **frozen data anchor** (energy-distance/MMD to unlabeled real samples) — NOT live-field closure
2. **coherent operator** in a latent (single map, not free residual)
3. **group generator** `M = matrix_exp(A)`
4. **stability-only reg** (det≈1 [+ well-conditioned cond→1 for isometric symmetries])
Operator discovered BEFORE/separately from EqM, FROZEN, then used as EqM augmentation.
Caveat (rung 15): coherent discovery needs structure-ALIGNED latent coordinates; a recon-only learned
latent gives coverage-recall but incoherent operator. So the first real test should use a fixed,
reasonable feature space, not a jointly-learned one.

## Integration points found in `projects/diff-EqM/experiments/dganm_variants/`
- **Variant framework** = the plug-in. Each variant = a `step_fn(model, x, step, device, args) -> (loss, diag)`
  handed to `train_loop`. Shared infra in `_common.py`: `eqm_loss`, `eqm_ct` (the c(γ) schedule),
  `build_unet`, `sample_euler`, `compute_model_fid`, `build_cifar_loader`, `TrainArgs`, `train_loop`.
- **Feature space available:** `UNetWrapper` registers a forward hook on `unet.middle_block` and exposes
  `self._features` — a ready latent/feature space for a frozen anchor + operator (no new encoder needed).
- **Precedent:** `v00_vanilla` (base), `v10_hard_example` (mining — our negative baseline),
  `v11_hard_example_equivariant` (already does equivariant augmentation `||f(x+δ)-f(x).detach()||²`).
- **Run path:** `run_variant.py` harness; CIFAR-scale; FID via `evaluate_fid.py`; reference stats already
  used by prior runs.

## Proposed minimal real-EqM test: `v12_stable_generator_aug`
Smallest credible test = CIFAR-10 (the variant harness's native scale), data-space operator augmentation,
operator discovered against a FROZEN feature anchor.

### Stage 1 — discover operator (offline, before/around EqM training; NO live-field closure)
- Anchor: a fixed batch sampler of REAL CIFAR images → frozen feature embeddings via a FROZEN encoder
  (options, cheapest first): (a) a frozen pretrained Inception/feature net (already used for FID), or
  (b) features from a short-pretrained vanilla EqM UNet middle_block, FROZEN. Energy-distance/MMD anchor
  in that feature space.
- Operator `T_θ` candidates (start simplest):
  - data-space affine/group action is huge for images → instead learn the operator in the FROZEN feature
    space `g`: `T̂ = decode_g(exp(A)·g(x))` is not available (no feature decoder). So for v12-min, restrict
    the operator to a SMALL, structured data-space transform family whose generator is low-dim:
    e.g. learned global photometric/affine channel mixing, or a learned small spatial warp field —
    parametrized as `exp(A)` on a low-dim control, discovered to (i) preserve the frozen-feature
    distribution (anchor) and (ii) move (bounded), with stability reg. This keeps "frozen anchor +
    stable group generator" honest at image scale without a feature decoder.
  - This is the main design risk: choosing an operator family expressive enough to contain a useful
    symmetry yet low-dim enough for the `exp(A)` stability recipe. Document the choice explicitly.
- Validate operator on frozen anchor BEFORE EqM: on-manifold rate (feature-MMD low), non-identity, det≈1,
  coherence. Freeze.

### Stage 2 — EqM training with frozen-operator augmentation (new variant step_fn)
```
def step_fn(model, x, step, device, args):
    base = eqm_loss(model, x, device, ...)
    aug  = eqm_loss(model, T_frozen(x).detach(), device, ...)   # frozen discovered operator
    return base + args.alpha_aug * aug, {"base": base.item(), "aug": aug.item()}
```
No held-out targets; T is frozen; no live-field closure.

### Comparison arms (one table, controlled)
- `v00_vanilla` — BASE/floor.
- `v10_hard_example` — mining, NEGATIVE baseline (expected: no manifold-structure gain).
- known-augmentation (flips/crops as EqM aug) — KNOWN-symmetry positive reference.
- `v12_stable_generator_aug` — discovered stable-generator augmentation (treatment).
Metric: FID (lower better) + the diagnostic that the discovered operator is on-manifold/coherent.

## What needs the human / cluster (STOP conditions)
- Any real CIFAR EqM training run is cluster work (GPU). Current session **cannot** submit: SSH session
  needs interactive **2FA re-auth** — run `! scripts/cluster/ssh_bootstrap.sh` (user action).
- Decision needed (research-prior legitimacy): the operator family choice for images (photometric vs
  spatial-warp vs feature-space) is a design prior; pick deliberately, not silently.

## Recommended sequence
1. Finish toy rungs (15 done, 16 running; optionally a symmetry-aligned-latent rung later).
2. Land this plan (rung 17).
3. On cluster re-auth: implement `v12` data-space operator (photometric-affine first — lowest-dim, safest),
   discover vs frozen Inception-feature anchor, run CIFAR v00/v10/known/v12, compare FID.
4. Only if v12 beats v00 and ≥ known-aug at CIFAR → consider IN-1K.

## Honest risk
The toy used a clean low-dim latent where a linear `exp(A)` is the natural symmetry. Real image symmetries
are high-dim and may not be a low-dim linear group in any cheap feature space. The bridge's success hinges
on the operator-family choice (Stage 1). If no low-dim stable generator captures a useful image symmetry,
the honest fallback is KNOWN-symmetry augmentation (flips/crops), which the toys already say is the safe
lever. v12 is a research bet, not a sure thing — frame it as such.
