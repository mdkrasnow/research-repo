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

---
# FAST FEATURE-SPACE PROXY (2026-06-04) — result: NEGATIVE (do not extend CIFAR/FID for v12 yet)

Why: the 40ep CIFAR-FID bridge was too slow/noisy (BASE FID 228, KNOWN_AUG moved only ~10, v10 lowest,
v12_random≈KNOWN → noise-limited single-seed). Built a seconds-scale feature-space proxy instead.

Proxy: `projects/symmetry-discovery/experiments/feature_gap_proxy_eqm_bridge.py`. Real CIFAR → FROZEN
random-conv encoder → PCA k=32. Known transform = rotation; visible angles exclude a held-out band
{+15,+20,+25}°; anchor = features over ALL angles (unlabeled). Operator M=exp(A) in PCA space, learned
vs frozen anchor (energy distance) + stability(det~1,cond→1) + move. Metric = energy-distance(arm
features, held-out features); operator diagnostics mandatory.

| arm | gap_to_heldout | det | cond | move | read |
|---|---|---|---|---|---|
| BASE_FEATURE | 1.065 | — | — | — | baseline gap |
| KNOWN_AUG_FEATURE | 0.182 | — | — | — | positive control ✓ (gate passes) |
| RANDOM_STABLE_FEATURE | 4.826 | 0.24 | 5.05 | 86.5 | negative control ✓ (off-manifold, fails) |
| DISCOVERED_STABLE_FEATURE | 1.187 | 1.01 | 1.14 | 14.4 | ✗ ≈ BASE; beats random; STABLE diagnostics |

**Verdict: harness VALID (KNOWN 0.18 ≪ BASE 1.07; RANDOM fails), but the DISCOVERED stable-generator
does NOT close the held-out gap (1.19 ≈ BASE 1.07 ≪ KNOWN 0.18).** It produces a valid stable
on-distribution operator (det≈1, cond≈1, non-identity, gap-to-anchor 0.41, beats random) — but
anchor-matching the FULL feature distribution does not TARGET the held-out band. This is the rung-9/15
confound in CIFAR feature space: a single linear M in a frozen-random-conv+PCA space cannot capture the
rotation's nonlinear feature shift; matching the overall distribution ≠ reaching held-out angles.
(Robust across quick/full: discovered −base = +0.07 / −0.12 → ≈0 within noise.)

**Decision (per goal):** mechanism FAILS in real CIFAR feature space as-is → DO NOT launch longer
CIFAR/FID for the discovered operator. The feature-space recipe needs a fix first: either (a) a
SYMMETRY-ALIGNED feature space (frozen random conv + PCA is not aligned to rotation — same lesson as
rung 15), or (b) an operator/anchor that targets the held-out-implied symmetry rather than the full
distribution. Known-symmetry augmentation remains the safe lever (KNOWN closes the gap; loader already
does flips).

Running CIFAR/FID jobs (v00 228.75, v10 205.50, vK 218.84, v12_random 217.91, v12_discovered pending):
let finish + record for completeness, but they are NOT the inner loop and the proxy supersedes them for
the mechanism question.

---
# ARCHITECTURE-CORRECT PROXY (2026-06-04) — spatial Lie-generator. Architecture fixed; TARGETING is the open problem.

`projects/symmetry-discovery/experiments/feature_gap_proxy_spatial_operator.py`. Operator = spatial
affine M=exp(A) (2x2) via grid_sample on IMAGES (= v12's actual operator), discovered vs frozen feature
anchor, evaluated in a COMMON PCA32(frozen-conv) space. Flat-PCA dense-M kept as negative-arch baseline.

| arm | gap_heldout | gap_anchor | angle | det | cond | read |
|---|---|---|---|---|---|---|
| BASE | 1.065 | — | — | — | — | floor |
| KNOWN_AUG | 0.182 | — | — | — | — | positive control ✓ (gate) |
| PCA_LINEAR_DISCOVERED (old arch) | 1.119 | 0.40 | — | 0.95 | 1.16 | ≈base (wrong arch, as expected) |
| SPATIAL_RANDOM | 1.120 | 0.56 | 1.1 | 1.47 | 1.10 | negative control |
| SPATIAL_DISCOVERED | 2.893 | 1.42 | -23.7 | 1.00 | 1.00 | CLEAN rotation but WRONG direction |
| SPATIAL_DISCOVERED_RESIDUAL | 3.307 | 1.88 | -23.3 | 1.00 | 1.00 | residual-anchor didn't fix targeting |

**Conclusion (two separated findings):**
1. ARCHITECTURE CRITIQUE VALIDATED: the spatial operator learns a COHERENT clean rotation (det 1.00,
   cond 1.00) — the flat-PCA dense-M (≈base, incoherent) was indeed the wrong architecture. Right
   operator family works.
2. TARGETING IS THE OPEN PROBLEM (not architecture): SPATIAL_DISCOVERED learned the WRONG-DIRECTION
   rotation (-23.7°) vs the one-sided held band (+15..25°), so gap WORSENED. Anchor-distribution-matching
   is DIRECTION-AGNOSTIC: a roughly symmetric full-anchor + move + stability is satisfied by ±rotation,
   so the objective cannot aim at an asymmetric held-out region. Residual-anchor (far-from-visible
   weighting) also failed.

**Deep insight (reconciles with the toy ladder):** rungs 12-14 succeeded partly because the held-out
modes were SYMMETRIC under the group (either rotation direction fills them), so anchor-matching sufficed.
Here the held-out band is ONE-SIDED -> a specific direction/magnitude is required, which a
distribution-matching objective with NO held-out signal cannot determine. Discovery via frozen-anchor
matching works only when the held-out support is reachable symmetrically; it is not a general
held-out-targeting mechanism.

**Decision:** still DO NOT scale to CIFAR/FID. Architecture is solved; the remaining gap is the
OBJECTIVE/targeting. Options: (a) a targeting signal that breaks direction symmetry without held-out
labels (hard / may require weak supervision); (b) accept KNOWN-symmetry augmentation (works, gate passes,
loader already flips) as the lever for the paper; (c) restrict claims to symmetric-held-out settings.
KNOWN-aug remains the safe, validated path.

---
# CORRECTION (2026-06-04, same day) — the "TARGETING is open" conclusion above was a BUG ARTIFACT.

Two load-bearing bugs in `feature_gap_proxy_spatial_operator.py` (and in the real
`diff-EqM/.../_stable_operator.py`) invalidated the spatial-proxy interpretation directly above:

1. **No-grad encoder blocked discovery.** `FrozenConv.forward` / `RandomConvAnchor.features` were
   `@torch.no_grad`. For the SPATIAL arms the operator is applied in IMAGE space, then features are read
   through that no-grad encoder -> the anchor/energy-distance term contributed **ZERO gradient to A**.
   The operator was driven ONLY by the move + stability terms. Verified by autograd: the anchor term
   `does not require grad`. So "discovery" never happened; the learned rotation's MAGNITUDE came from the
   move target and its DIRECTION (±23.7°) was the random init seed. "Wrong direction" and
   "direction-agnostic" were symptoms of this bug, not a property of anchor-matching.

2. **Move-target leaked the held-band magnitude.** `move_target` = pixel-distance of a 20° rotation =
   the CENTER of the held band [15,20,25]. That handed the operator its magnitude (CLAUDE.md: "do not
   force the move magnitude to the held-out angle").

**Fixes:** (a) grad-flowing frozen encoder (`feat_grad`/`features_grad`: params frozen, no `no_grad`, so
gradient flows to the input); (b) generic non-leaking move HINGE [5°,45°] (non-collapse + non-blowup
only, NOT a target at the held center); (c) **MIXTURE-anchor objective**: match
`(1-π)·P_real + π·T(P_real)` to fresh real features (mass-correct), not `T(P_real)` alone.

**Corrected proxy result (grad-flowing, de-leaked; quick & full agree):**

| arm | gap_HO | angle | det | cond | read |
|---|---|---|---|---|---|
| BASE | 1.065 | — | — | — | floor |
| KNOWN_AUG | 0.182 | — | — | — | positive control ✓ |
| SPATIAL_RANDOM / _BIDIR | 1.12 / 1.41 | — | — | — | negative controls ≈ base ✓ |
| PCA_LINEAR_DISCOVERED | 1.119 | — | 0.95 | 1.16 | wrong arch ≈ base ✓ |
| SPATIAL_DISCOVERED_FORWARD (T-only) | 0.897 | +4.9 | 1.01 | 1.07 | correct DIRECTION now; **undershoots** magnitude (sits at move floor) |
| SPATIAL_DISCOVERED_INVERSE | 1.444 | -4.7 | — | — | worse — direction is now DETERMINED (gauge ambiguity gone) |
| **MIXTURE_SINGLE** | **0.305** | **+29.5** | 0.99 | 1.13 | **best**; correct direction AND magnitude; near KNOWN; beats base 3.5× & random |

**Corrected conclusions:**
1. ARCHITECTURE: validated (unchanged) — spatial affine exp(A) represents the rotation cleanly.
2. The earlier "gauge flip fixes it" reading was ALSO an artifact: once the anchor gradient actually
   trains A, the **direction is determined** (INVERSE is genuinely worse), so gauge/orbit arms are moot
   for a one-sided gap.
3. **The real positive result is the MIXTURE-anchor objective.** T-only `ed(T(vis),anchor_full)` is
   minimized by staying near identity (visible mass dominates the anchor -> a small rotation already
   matches the bulk -> undershoots, +4.9°). The mixture lets `(1-π)·vis` cover the bulk, freeing
   `T(vis)` to GO FILL the missing held region -> +29.5°, gap 0.305. **This DISCOVERS the gap-filling
   direction+magnitude unsupervised** (no held-out labels; the held band enters only as unlabeled mass
   in the frozen anchor). Anchor-matching is NOT inherently direction-agnostic — the mass-mismatched
   T-only objective was.
4. Toy rungs 12-14 reconciliation: those used T-only and still worked because their held-out was
   SYMMETRIC under the group; the mixture objective is what generalizes T-only to ONE-SIDED gaps.

**Real v12 updated to match the validated recipe:** `_stable_operator.py` now uses grad-flowing
features + mixture-anchor + non-leaking move hinge and returns the generator `A`; `v12_stable_generator_aug.py`
augments EqM with the GROUP exp(t·A), t~U(-1,1) (knob `aug_mode` = orbit|single|bidir). CPU smoke passes
(discovery anchor 9.85→3.70; orbit aug fires, ratio ~0.88).

**Decision:** mechanism now PASSES the fast proxy honestly (best discovered 0.305 < base 1.065, < random
1.41, controls clean). Per the success path this WARRANTS one small EqM/CIFAR run with the corrected v12
(orbit aug) — but the prior 40ep FID was noise-limited, so any FID run must be long enough to resolve the
KNOWN_AUG≈10-FID lever (≥ the epochs where vanilla FID stabilizes). NOTE: feature_shift_consistency is low
(~0.01) in the real-CIFAR smoke — expected (a spatial rotation is NOT a global feature-space translation),
but watch operator-quality diagnostics over recall per the rung-15 coverage confound.

---
# CIFAR-FID BRIDGE RESULT (2026-06-05) — NEGATIVE. Proxy did NOT predict FID; rotation isn't a useful CIFAR symmetry.

150ep CIFAR, 1 seed, FID(5000), corrected v12 (mixture-anchor discovery + grad-flow + orbit aug):

| arm | FID | vs base | role |
|---|---|---|---|
| v00 BASE | 14.31 | — | floor (matches prior harness v00 14.17 -> harness OK) |
| v12_random | 13.63 | -0.68 | negative control — BEAT everything |
| v12_discovered_orbit | 14.41 | +0.10 | TREATMENT — ~=base, NO benefit |
| vK_known (rotate15) | 14.82 | +0.51 | positive control — WORSE than base, FAILED |

**Verdict: bridge NEGATIVE.** Ordering random > base > discovered > known is wrong for the mechanism.
Two decisive reads:
1. **Positive control FAILED**: known rotation aug did not beat base (14.82 > 14.31). By the project's
   own control rule, a failed positive control means the treatment is uninterpretable as a mechanism win
   — BUT here it is interpretable: rotation is NOT a useful symmetry of CIFAR (objects have canonical
   orientation), so augmenting with rotations (known OR discovered) trains on unnatural off-distribution
   images and does not help FID. v12_random (small random affine) won only as a weak generic regularizer.
2. **Proxy/FID DISAGREE on the same lever.** The fast proxy's positive control (KNOWN rotation) strongly
   beat base (0.18 vs 1.07) because the proxy INJECTED rotation as ground-truth and measured held-out
   rotation recovery. FID measures full-distribution generation, where rotation aug is useless/harmful.
   So the proxy validated the discovery MECHANISM (recovering an injected symmetry) but the injected
   symmetry itself is not FID-relevant for CIFAR. The proxy is NOT a valid predictor of FID benefit when
   the target symmetry is not actually exploitable by the generative task.

**Decision (per CLAUDE.md stop-conditions + prior memory):** kill the v12 discovery->CIFAR-FID bridge.
Do NOT run more seeds — the positive-control failure shows the lever (rotation aug) itself doesn't help
CIFAR EqM, so a multi-seed treatment can't rescue it. The discovery method remains validated in the toy
ladder + feature proxy, but it needs a setting where the discovered symmetry is genuinely useful to the
generative target (CIFAR rotation is not). Matches earlier note: "v12 mechanism fails on real CIFAR;
CIFAR/FID noise-limited; do not extend CIFAR/FID for v12." Known-symmetry aug is NOT a working lever here.

---
# v13: ARCHITECTURE-CORRECT BRIDGE (2026-06-05) — v12 killed the rotation arch, NOT the bridge.

**v12 rotation-like bridge result (context, NOT a bridge kill):**
150ep CIFAR FID: v00_base 14.31, v12_random 13.63, v12_discovered 14.41, vK_known_ROTATION 14.82.
- v12 rotation-like discovered operator: no benefit.
- random affine: slight win (generic regularization).
- known ROTATION: harmful.
- **Reason: rotation is not a useful CIFAR augmentation.** The operator family (single global 2x2
  rotation/scale/shear, NO translation/crop) AND the positive control (known rotation) were both
  MISMATCHED to CIFAR. CIFAR benefits from crop/translation/flip/color nuisance transforms.
- **Therefore: do NOT kill the bridge.** Kill only the v12 rotation architecture. vK_known_ROTATION
  failing does NOT mean known augmentation failed — rotation was the wrong known control.

**v13 plan:** SE(2)-style homogeneous affine generator (3x3, includes TRANSLATION/crop/scale).
A = 3x3 (top 2 rows learned, bottom row 0), M = matrix_exp(A) acts on homogeneous image coords ->
can express translation + small crop/zoom/scale/shear/rotation. Same corrected discovery recipe:
frozen GRAD-FLOWING anchor, MIXTURE objective ((1-pi)P_real + pi*T(P_real) ~= P_anchor), broad
NON-LEAKING move hinge, stability reg (det~1, cond ctrl, translation bounded), no live-EqM closure.
Positive control fixed: KNOWN_TRANSLATE_CROP (random crop pad4 / translate +-2..4px) must beat BASE
(loader already does hflip -> KNOWN_FLIP saturated). KNOWN_ROTATION kept as a wrong-transform control.
Gate order: fast SE2 proxy (inject translation gap) -> v13 EqM smoke -> full FID only if both pass.

---
# v13 SE2 BRIDGE — PRE-REGISTERED DECISION MATRIX (written 2026-06-05, BEFORE FIDs land)

Arms (all 150ep CIFAR, same harness/seed0):
  v00_base 14.31 (reused floor) | vK_rotation 14.82 (reused wrong-transform ctrl)
  v10_hardneg 19405301 (mining competitor) | vK_translate_crop 19404840 (POSITIVE control)
  v13_discovered_se2 19404844 (TREATMENT) | v13_random_se2 19404851 (negative control)

Main question: does v13_discovered_se2 beat base, random SE2, AND v10 mining, while
vK_translate_crop validates the harness (beats base)?

Read controls FIRST, then treatment, in this order:

A. HARNESS FAIL — vK_translate_crop NOT < v00_base.
   -> Do NOT interpret v13. The CIFAR aug lever failed (or 150ep/1seed noise too large). Stop, do not
      scale. (Mirror of the v12 rotation positive-control failure, now for translation.)

B. GENERIC REGULARIZATION — v13_random_se2 < v13_discovered_se2.
   -> Gain is generic affine/crop regularization, NOT discovery. Discovery direction not the source.
      Discovery mechanism not supported even if v13_disc < base.

C. DISCOVERY WORKS, NOT > v10 — v13_discovered < base AND < random, but v10_hardneg < v13_discovered.
   -> Discovery transfers to EqM, but v10 mining remains the stronger practical lever. Report as
      competitive-not-superior; mining stays the recommended mechanism.

D. STRONG BRIDGE WIN — v13_discovered < base AND < random AND < v10, with sane operator diagnostics
   (translation-leaning, det~1, cond bounded, anchor improved, non-identity).
   -> Frozen-anchor stable-generator discovery is a REAL EqM mechanism that beats mining. Worth
      capability probes (bridge_capability_eval.py) + multi-seed replication + (only then) IN-1K.

E. KNOWN AUG WINS, v13 DOESN'T — vK_translate_crop < base but v13_discovered NOT < base.
   -> Operator FAMILY (translation/crop) is useful, but discovery is not learning the useful augmentation
      well enough. Debug discovery objective/operator before any scale; known-aug is the lever meanwhile.

F. WITHIN NOISE — controls do not separate (e.g. translate_crop ~= base ~= random ~= disc within ~0.5 FID).
   -> Inconclusive. Need multi-seed before any claim. Do not over-read a single-seed ordering.

Interpretation discipline: read A (harness) before anything; then B (generic vs discovered); then C/D
(vs v10); E/F as fallbacks. Trust operator_diag.json (det/cond/tx/anchor) over FID alone per the rung-15
coverage/coherence confound. collect_bridge_results.py auto-emits this read once FIDs are present.
