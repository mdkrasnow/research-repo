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
