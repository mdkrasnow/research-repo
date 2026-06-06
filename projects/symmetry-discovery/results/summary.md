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
