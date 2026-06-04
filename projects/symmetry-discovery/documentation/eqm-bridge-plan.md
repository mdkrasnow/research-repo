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

---
# IMPLEMENTATION (2026-06-04) — built + locally smoked

## File/function map (discovered)
- EqM field training: `dganm_variants/_common.py::train_loop` (loop) + `eqm_loss` (pixel-space
  field-matching loss) + `eqm_ct` (c(γ) schedule). Backbone `build_unet`/`UNetWrapper` (middle_block
  feature hook). FID via `evaluate_fid.py`. CIFAR loader (already does RandomHorizontalFlip).
- v10 mining plugs in: `dganm_variants/v10_hard_example.py::step_fn` (PGA δ on base loss).
- Known aug plugs in: NEW `dganm_variants/vK_known_aug.py::step_fn` (`L += λ·eqm_loss(model, T_known(x))`).
- Stable-generator aug plugs in: NEW `dganm_variants/v12_stable_generator_aug.py` — `train()` discovers
  a frozen operator BEFORE `train_loop`, `step_fn` augments `L += λ·eqm_loss(model, affine_warp(x,M).detach())`.
- Operator discovery: NEW `dganm_variants/_stable_operator.py::discover_stable_affine` — frozen
  RandomConvAnchor (non-co-adapting features) + energy-distance + move + stability(det~1,cond→1); operator
  = `M=exp(A)` (2x2) applied as a spatial affine warp. NOT live-EqM closure.

## Representation space chosen
Operator acts as a low-dim IMAGE group action (spatial affine via affine_grid/grid_sample), because
`eqm_loss` needs an image and there is no feature→image decoder. Frozen anchor lives in a frozen
random-conv FEATURE space (cheap, no external weights; Inception reserved for FID). This is the
"lowest-dim, safest" photometric/affine option from the original plan.

## Local CPU smoke (real CIFAR; `experiments/smoke_v12_bridge.py`) — PASS
- Operator discovery on real CIFAR (250-step smoke): angle −9.8°, cond 1.02 (≈isometric), off_identity
  0.30, anchor loss drops (T(real) stays near data-manifold features). det 0.74 (not yet ~1 — 250-step
  smoke; tightens at full 600 steps — verify on cluster).
- affine_warp sane; v12 + vK step_fn compute base+aug and backprop on a tiny stand-in model.
- NOTE: `feature_shift_consistency` is NOT a meaningful coherence metric for a GLOBAL affine (coherence
  is guaranteed by the single shared M; the metric measures content-dependent feature directions). Trust
  det/cond/off_identity/anchor-loss + the v12 vs v12_random contrast instead.

## Arms + exact run (cluster; one sbatch per config via existing variant_pilot.sbatch)
configs in `projects/diff-EqM/configs/variants/bridge/`:
  bridge_v00_base.json (BASE) · bridge_v10_hardneg.json (HARDNEG, neg) ·
  bridge_vK_known.json (KNOWN_AUG, positive control) · bridge_v12_discovered.json (TREATMENT) ·
  bridge_v12_random.json (negative control: undiscovered random operator)
All 40-epoch, final FID 5000 samples (cheap first pass), same backbone/budget (controlled).

Submit (after 2FA, from repo root), e.g.:
  root="$(cd "$(dirname scripts/cluster/remote_submit.sh)/../.." && pwd)"
  CONFIG_PATH=projects/diff-EqM/configs/variants/bridge/bridge_v12_discovered.json \
    bash scripts/cluster/remote_submit.sh "$root/projects/diff-EqM/slurm/jobs/variant_pilot.sbatch" diff-EqM
  # repeat for the other 4 configs (CONFIG_PATH env per submit). gpu partition (mining/aug fit; >=40G safe).

## Interpretation rule for the result
- Positive control gate: KNOWN_AUG must move FID vs BASE (else harness can't use aug → stop).
- Treatment: v12_discovered FID < v10_hardneg AND ≤ KNOWN_AUG → recipe beats mining, approaches known.
- v12_discovered vs v12_random: discovered must beat random (else the operator isn't doing real work).
- Trust operator_diag.json (det/cond/off_identity/anchor) — not FID alone (toy coverage/coherence lesson).

## STOP — needs human: cluster 2FA
Local session cannot submit (SSH control master expired; interactive 2FA). Required human action:
  ! scripts/cluster/ssh_bootstrap.sh
Then either you submit the 5 configs above, or I submit once the session is live. Also confirm the
operator-family prior (spatial-affine) is acceptable as the first image instantiation (research-prior
decision); alternatives = photometric (RGB 3x3) or feature-space operator with a learned decoder.
