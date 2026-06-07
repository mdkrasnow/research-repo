# Results Summary — Symmetry Discovery (rungs 1–16)

## Research question
What lets EqM generalize onto held-out manifold regions? Compare hard-negative mining vs known-symmetry
constraints vs unsupervised symmetry discovery (frozen anchor + learned operator).

## Headline findings
1. **Hard-negative / v10 mining installs NO manifold structure** — dead in every rung.
2. **Known-symmetry constraints/augmentation generalize cleanly** (ORACLE).
3. **Unsupervised discovery works near-oracle in a well-posed regime:** frozen data anchor + coherent
   group generator `M=exp(A)` + stability reg (det≈1, cond→1) recovers ROTATION/ISOMETRIC symmetries in a
   clean (supervised/aligned) latent — including higher-dim K=3 with correct active-subspace selection and
   passive-dim preservation (rung 14), WITHOUT naming the symmetry class.
4. **Recall is COVERAGE-confounded** — the key methodological caution. A stable on-manifold reshuffle can
   fill held-out regions (high recall) without being the true symmetry. Trust operator-quality metrics
   (on-manifold rate, shift coherence, structure preservation, recovered generator), not recall alone.
5. **Outside the clean-rotation regime, discovery degrades to coverage-recall without coherent symmetry:**
   learned/unaligned latent (rung 15) → incoherent operator; non-rotation screw (rung 16) → inconclusive
   (toy ill-posed, 2 attempts).

## Rung-by-rung (recall as %ORACLE where applicable; operator quality in findings.md)
- 1 arithmetic: equivariance perfect; mining dead.
- 2 ring (linear-in-obs): known + discovered rotation work.
- 3 nonlinear-hidden free op: fail (off-manifold).
- 4 enc-linear-dec recon-frozen: fail (recon latent ≠ linearizing).
- 5 ORACLE_LATENT control: recipe-bug exposed (M→identity even with clean latent).
- 6 hinge anti-identity: insufficient.
- 7 continuous manifold: cyclic pulls to identity.
- 8 identity-exclusion: M drifts to non-symmetry; closure can't anchor.
- 9 FROZEN ANCHOR: fixes co-adaptation (T on-manifold) but distributional anchor → incoherent shuffle.
- 10 single latent matrix: coherence fixed; ~37% ORACLE (precision gap).
- 11 orbit aug: coverage rejected; gap is precision.
- 12 group generator exp(A): SKEW=ORACLE, FREE~69% — precision fixed.
- 13 prior-budget (2D): GEN_FREE_STABLE (det+cond, no skew) = 99% ORACLE; rotation emerges from stability.
- 14 higher-dim K=3 (stable≠rotation): GEN_FREE_STABLE = oracle, discovers active plane + preserves
  passive dim. STRONGEST clean result.
- 15 learned latent: recall 89–97% but via COVERAGE (incoherent operator, passive leak); coherent
  discovery needs aligned coords. Partial.
- 16 screw/helix (non-rotation): INCONCLUSIVE ×2 (confounded then ill-posed). Unresolved.

## Status & next
Toy ladder PAUSED at a natural stop. EqM bridge plan written (`documentation/eqm-bridge-plan.md`):
new variant `v12_stable_generator_aug` discovering a frozen stable operator vs a frozen feature anchor,
compared against v00/v10/known-aug on CIFAR FID.
**Next human action:** `! scripts/cluster/ssh_bootstrap.sh` (2FA) to enable the EqM bridge, OR choose a
toy sub-problem (symmetry-aligned unsupervised latent / better non-rotation testbed).

## v14 CPU ladder (2026-06-05) — VERDICT: v14 NOT authorized for CIFAR/FID
Tested whether discovering an augmentation DISTRIBUTION (vs v13's single frozen operator) adds value.
- Rung A (anchor-grad gate): PASS — anchor grad to operator 924.7, encoder frozen, no-grad path=0.
- Rung B (move/leakage): PASS — broad hinge anchor-driven, no leak; single-op UNDERDETERMINED on
  translation-spread anchor (motivates distribution).
- Rung C (single vs distribution, translation-space coverage): PASS — single op CANNOT cover a 2D crop
  region (cover 5.19 vs base 1.40); discovered DISTRIBUTION covers it (0.32 vs oracle 0.017), high-rank
  2D support (eig_ratio 0.39) emerges without entropy floor.
- Rung D (aug-training, small CNN, 3 seeds): DECISIVE NEGATIVE — translated-val acc: base 0.391,
  known_crop 0.431, single 0.377 (v13 single HURTS), random_dist 0.421, disc_dist 0.420. Discovered
  distribution ~= random distribution; closes 72% of gap, crushes single, but adds NOTHING over random.
- Rung E (EqM-lite): INCONCLUSIVE (no signal; translated-field gap 0.0014 within noise).
CONCLUSION: the distribution mechanism fixes v13's under-diversity (A-C) but the DISCOVERED distribution
equals a RANDOM one in value (D). For known/generic CIFAR nuisance symmetries (translation/crop) there is
nothing to discover — the useful aug is just random crop. Do NOT build v14 production or run FID.
Files: v14_ladder_anchor_grad_test.py, v14_ladder_move_leakage_test.py, feature_gap_proxy_cifar_se2_distribution.py,
aug_training_proxy_cifar.py, eqm_lite_aug_proxy.py, _se2_discovery.py.

## v14 (beat-crop) policy ladder (2026-06-05) — VERDICT: NOT authorized for FID
A (policy safety) PASS; B (policy support) PASS — anchor+entropy(2D-cov floor) high-rank 2D, beats
single/random, bounded; ablation shows the 2D floor is needed. C (utility vs crop, 3 seeds) — discovered
anchor+entropy+utility policy TIES crop (transl±6 0.382 vs 0.377, within crop noise 0.018) + beats random,
but does NOT beat crop; utility = faint edge over entropy/random, none over crop. D (EqM-lite) INCONCLUSIVE
(no signal). Blocked by: no headroom over near-optimal crop for a known/generic symmetry. Mechanism works;
value-over-crop does not materialize. Files: v14_policy_safety_tests.py, feature_gap_proxy_cifar_policy.py,
aug_policy_training_proxy_cifar.py, eqm_lite_policy_proxy.py, discover_policy in _se2_discovery.py.

## v15 (model-aware SAFE-ADVERSARIAL aug) ladder (2026-06-05) — VERDICT: NOT authorized for FID; STRONGER negative
Reframing (user): stop imitating crop; learn a SAFE-HARD policy that targets the current model's weaknesses
(anchor=safety, scorer-loss=hardness/utility, conditionality=per-image adaptivity). The one lever v14 never
tested: adversarial utility + qθ(T|φ(x)).
- Rung 1 (KNOWN ceiling, 3 seeds): crop_pad4 = 0.409 (tstd 0.001) is the clear ceiling; crop_pad6/transl_scale/
  transl_flip all ≤ it. "Beat crop_pad4" is the VALID target (moderate aug matches the ±6 test nuisance best).
- Rung 2 (frozen-scorer hardness gate): PASS — safe-hard makes augs HARDER on a frozen scorer (l3.0 CE 1.82 vs
  crop 1.70) while staying MORE on-manifold than crop (anchor ED 2.95 < 3.26) and 2D. **But this PASS was a
  false positive for downstream value (see Rung 3).**
- Rung 3 (TRAIN with global safe-hard, 3 seeds): NO BEAT at every lam — l1.5 0.397, l3.0 0.385, both < crop
  0.409; MONOTONE: more adversarial pressure → worse (l3.0 even < random 0.393). Frozen-scorer hardness ≠
  training value; the policy over-augments off the useful regime.
- Rung 4 (CONDITIONAL qθ(T|φ(x)), 3 seeds): NO BEAT, worse — conditional 0.362–0.363 ≪ crop 0.409 (≈ base
  transl±6) and crushes CLEAN acc (0.38 vs base 0.47) at lam {0.5,1.5}. Per-image mu_std real (3.4–8.1 px in x)
  so the policy IS image-dependent — but the adaptivity ACTIVELY HURTS (adversarial-degenerate shifts).
CONCLUSION: model-aware safe-adversarial augmentation does not just fail to beat crop — being clever (hard /
conditional) is WORSE than uniform crop. Mechanism reason: adversarial-vs-frozen-scorer finds the scorer's
blind spots, not generalizing transforms; the anchor keeps images realistic but the chosen SHIFTS are
adversarial-degenerate → train/test mismatch. CIFAR translation is genuinely uniform/isotropic ⇒ crop ≈ Bayes
-optimal ⇒ nothing to discover and cleverness backfires. This is the strongest confirmation of the whole-arc
thesis: discovery/adaptivity earns value ONLY for UNKNOWN, non-generic structure (toy ladder rungs 12–14), not
for a known generic nuisance. Do NOT run FID. Files: v15_rung1_known_ceiling.py, v15_rung2_safe_hard.py,
v15_rung3_train_safe_hard.py, v15_rung4_conditional.py (discover_conditional/CondPolicy), results_v15_rung*.json.

## v16 (RESIDUAL policy over best-known, VALIDATION-utility objective) ladder (2026-06-06) — VERDICT: NOT authorized for FID
New hypothesis (user): drop frozen-scorer hardness (v15's failure). Crop is a strong BASE; a bounded RESIDUAL
distribution stacked on best-known crop may beat it IF optimized for actual short-run VALIDATION utility
(bilevel-lite ES), with anchor/entropy as constraints. Genuinely different objective from v14 (anchor-match)
and v15 (hardness).
- Exp 1 (KNOWN ceiling, 3 seeds, mixed-corruption robust val): BEST_KNOWN = crop_pad4_color (robust 0.429 >
  crop_pad4 0.420, margin 0.009 > noise 0.006); color helps (val corruption includes brightness).
- Exp 2 (residual policy, ES on validation utility, 3 seeds, separate test split): NO BEAT vs random.
  base 0.391, best_known 0.4008, rand_residual 0.41458, LEARNED_residual 0.41483, no_anchor 0.416, no_entropy
  0.413. Learned vs random Δ=0.00025 ≪ noise 0.005 → learning the residual SHAPE adds NOTHING over a random
  mild residual. The residual CATEGORY marginally beats best_known (+0.014 ≈ noise) purely from added diversity.
- Exp 3 (stage CURRICULUM, ES on schedule φ=(a,b), 3 seeds): NO BEAT. best_known 0.4008, static_residual 0.4148,
  curriculum_learned 0.4132, random_curriculum 0.4148. Curriculum ≤ static = random schedule; ES drifted to a
  mild decay (s 0.40→0.30) with zero benefit. TIMING adds nothing either.
- Exp 4 (EqM-lite): NOT RUN — gated on Exp 2/3 passing; both no-beat → stop-rule (do not run FID).
CONCLUSION: over a known generic base (crop+color), neither the residual's SHAPE (E2) nor its TIMING (E3)
can be learned to beat a fixed/random mild residual; the only gain anywhere is "add a mild residual at all"
(≈ noise), needing no learning. FOURTH consecutive negative (v14 dist==random, v15 hardness-backfires,
v16-shape==random, v16-timing==random): a validation-utility bilevel objective also converges to random.
Confirms the whole-arc thesis decisively — augmentation discovery has no headroom for a known generic CIFAR
nuisance. Do NOT run FID. Files: v16_known_aug_ceiling.py, v16_residual_policy_proxy.py,
v16_residual_curriculum_proxy.py; results_v16_exp{1,2,3}.json.

## v17 MorphismGym — UNKNOWN image-morphism discovery (2026-06-06) — VERDICT: FIRST POSITIVE, EqM-lite PASS

PIVOT (per pipeline hint (a)): the v13-16 negatives were all on CIFAR + a KNOWN GENERIC nuisance (crop/
translation) where crop is ~Bayes-optimal so there is nothing to discover. v17 changes the TESTBED: a
controlled rendered-shape gym (circle/square/triangle/star/ring/cross x latent factors cx,cy,rot,scale,
hue,bright,stroke,bg) where valid morphisms are REAL but HIDDEN (visible=narrow band, anchor=broad valid
range unlabeled, heldout=disjoint valid tail eval-only). Crucially valid morphisms keep images ON the
rendered-shape manifold while decoys (crop-erase/big-shear/bg-shortcut/shape-warp/occlude/color-collapse)
LEAVE it -> a LABEL-FREE anchor (PCA-whitened random-conv energy-distance + structure/chroma stats) can
separate valid from invalid (impossible for generic CIFAR crop). NO pretrained semantics; GT latents
eval-only. Policy qtheta(family,magnitude,composition) learns magnitudes by reparam + family weights by an
EMA-reward bandit (grouped application).

- Phase 0 (calibration) PASS: 0A gym structured (separability AUC 1.0); 0B oracle payoff -- the shape-ID
  CLASSIFIER is insensitive (oracle~=random, shape ID robust to these morphisms) so gate switched to
  EqM-lite which brackets cleanly x2 seeds (BASE 0.43 > RANDOM_DECOY 0.14 > RANDOM_VALID 0.11 >
  ORACLE_MULTI 0.077); 0C all 3 anchors AUC 1.0 (ae widest, randomconv used); 0D no hallucination.
- Phase 1 (discovery) PASS all 7 tasks: recovers true families (recall .78-1.0), avoids decoys
  (decoy_use ~0.00), stays valid (.82-.92 vs random_decoy .50-.64), multi>single per-factor coverage,
  NO_ANCHOR ablation degrades VALIDITY (anchor's role = on-manifold/decoy-avoid, not coverage),
  impossible_control no hallucination.
- Phase 2 (payoff) EqM-lite PASS: multi_independent (3 seeds) DISCOVERED_MULTI gap 0.062 BEATS random_valid
  0.107 and BEATS oracle 0.078 vs base 0.466; NO_ANCHOR 0.30 (anchor essential). This is the FIRST time
  discovery BEATS random in the whole v13-v17 arc. single_rotation = no payoff headroom (base gap 0.004,
  all tie). Classifier diagnostic factor-dependent: PASS single_rotation (rotation breaks shape-ID), insens
  multi_independent. discovered_multi TIES single on EqM-lite (one morphism saturates field-robustness).

WHAT IT MEANS: targeting WITHOUT labels (the bottleneck recorded after v12-16) IS solvable when the data
has real hidden structure + a manifold anchor exists. WHAT IT DOES NOT MEAN: synthetic gym + EqM-LITE
proxy, NOT real EqM/FID. FID remains NEVER auto-authorized -> RECOMMEND integration pending EXPLICIT human
approval. NEXT: Phase 3 natural-ish transfer (dSprites/rotated-symbols) before any real-EqM bridge.
Files: v17_morphism_gym.py, v17_policy.py, v17_eval_metrics.py, v17_common.py, v17_run_{calibration,
discovery,payoff}.py, v17_collect_results.py; results/v17_*.json, v17_report.md, v17_verdicts.json.

## v17 Phase 3 — natural-ish transfer (MNIST), 2026-06-07 — PASS (3 seeds)

Objects = real MNIST handwriting (colorized on 32x32); SAME morphism families, decoys, label-free anchor,
policy, metrics applied on top. multi task (hidden rot+scale+hue). EqM-lite payoff (lower=better), 3 seeds:
- DISCOVERED_MULTI gap 0.022-0.023 BEATS random_valid 0.080, BEATS oracle 0.049, vs base 0.117;
  NO_ANCHOR 0.108-0.117 (~=base -> anchor essential); decoy_use 0.00 all seeds.
The discovery pattern (discovery > random, approaches/beats oracle, anchor load-bearing) HOLDS on real
images, not just the synthetic shape generator. Phase 0->3 ladder complete and passing.

VERDICT (whole v17): unsupervised morphism discovery from data structure alone WORKS and transfers --
FIRST positive of the v13-v17 arc. Mechanism: real hidden structure + a label-free manifold anchor that
separates valid (on-manifold) from invalid (off-manifold) morphisms, which generic CIFAR/crop lacked.
Bottleneck "targeting without labels" is SOLVABLE in this regime. STILL synthetic/MNIST + EqM-LITE proxy:
FID NEVER auto-authorized -> RECOMMEND real-EqM integration pending explicit human approval + (ideally) a
non-MNIST natural dataset (dSprites/Shapes3D) confirmation. Files: v17_transfer.py; v17_transfer_multi_seed{0,1,2}.json.

## v17 Phase 3b — dSprites confirmation (2026-06-07) — PASS (3 seeds)

Second natural-ish generator (binary geometric sprites, GT shape/scale/orientation latents; distinct from
MNIST handwriting and the SDF shape gym). multi task, EqM-lite (lower=better), 3 seeds:
- DISCOVERED_MULTI 0.013-0.017 BEATS random 0.038-0.041, BEATS oracle 0.030-0.034, vs base 0.048-0.050;
  NO_ANCHOR ~=base (anchor essential); decoy_use ~0.
The discovery > random > and approaches/beats oracle pattern now holds across THREE distinct image
distributions (synthetic SDF shapes, MNIST, dSprites), 3 seeds each. Robust positive. Files:
v17_transfer_dsprites_multi_seed{0,1,2}.json; v17_transfer.py (--dataset mnist|dsprites).
