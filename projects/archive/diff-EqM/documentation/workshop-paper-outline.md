# Workshop Paper Outline — NeurIPS 2026 (Aug 29 deadline)

**Status**: draft v1, locked 2026-05-19 post-Round-2 lit review.
**Target venue**: NeurIPS 2026 workshop on generative models / score-based / flow / energy methods (specific workshop list announced ~July).
**Page budget**: typical workshop 4–8 pages + appendix.
**Working title**: "Adaptive Hard-Negative Mining for Regression-Target Generative Models"

## TL;DR (the one sentence we need reviewers to take away)

> PGD-mined hard examples on the EqM regression target compose with CAFM-style discriminator post-training (Lin et al. 2026) to compound FID gains, addressing VeCoR's explicit open call for "adaptive hard-negative mining" in flow models.

## Sections

### 1. Introduction (~1 page)
- Hook: regression-target generative models (flow matching, Equilibrium Matching) achieve SOTA but training is plain MSE — no adversarial signal.
- Lin et al. 2026 CAFM introduces discriminator-based AT for flow matching → SOTA gains.
- VeCoR 2025 introduces velocity contrastive regularization → §7 calls out "adaptive hard-negative mining" as open future work.
- **Our contribution**: PGD-mined hard examples on the regression target itself (v10), composed with CAFM-style discriminator AT, compounds gains beyond either alone on EqM.
- Brief result preview: vanilla EqM-B/2 80ep 31.41 → CAFM-only X → v10+CAFM Y. CIFAR-10 multi-seed validates mechanism.

### 2. Background (~1 page)
- §2.1 Equilibrium Matching (Wang & Du 2025): regression target `(ε − x) · c(γ)`, implicit gradient field of an energy landscape.
- §2.2 Flow matching → SiT → CAFM: training paradigm + adversarial post-training.
- §2.3 Adversarial training of generative models: brief tour (Madry, Briglia equivariance, Wu DAT, Geng AEBM-Diff, Kong ACT, VeCoR, Rob-GAN).
- §2.4 Open gap: no prior work combines PGD-input-mining with regression-target AT.

### 3. Method (~1.5 pages)
- §3.1 v10: PGD hard-example mining on EqM target.
  - L_v10 = ||f(x_γ + δ*) − target||²
  - Mining: L2-bounded, K-step PGA (default K=1, FGSM).
  - Mechanism: discriminator catches global mismatch; PGD catches local regression failure.
- §3.2 v10 + CAFM combined:
  - L_total = L_CAFM_gen + λ · L_v10.
  - Schedule: N=16 discriminator updates per generator update (Lin's recipe).
  - γ-conditional discriminator (adapted from Lin's SiT setup; EqM's t=0 maps to γ).
- §3.3 Equivariance vs invariance discussion (Briglia 2025).
  - v10 is invariance-flavored but ground-truth-anchored.
  - Small λ + equivariant fallback (v11) ready for ablation.

### 4. Experiments (~2 pages)

#### CIFAR-10 sanity (§4.1)
- Variant harness 150 epochs, 1 seed.
- Diagnostic-driven gate: L_hard > L_clean, ||δ|| at boundary, no collapse.
- Result: v10 stable; v10+CAFM-CIFAR-port stable.

#### EqM-B/2 IN-256 ablation (§4.2 — HEADLINE)
- 4 conditions × 3 seeds: {vanilla, v10-only, CAFM-only, v10+CAFM}.
- All 4 are 80-epoch baseline + 10-epoch post-training (vanilla = no post-training).
- Table: FID-50K (guidance-free + CFG=1.5).
- Welch t-test: v10+CAFM beats CAFM-only and v10-only.

#### Scaling on EqM-S/2 IN-100 (§4.3)
- 3 seeds × 4 conditions on a smaller scale.
- Confirms result is not B/2-specific.

#### Diagnostics + ablations (§4.4)
- λ sweep {0.03, 0.1, 0.3, 1.0}.
- K sweep {1, 3, 5}.
- Throughput overhead measurement.
- Without-discriminator vs without-PGD-mining individual contributions.

### 5. Discussion (~0.5 page)
- Why discriminator + PGD compose: complementary failure modes.
- When v10 alone is enough vs when CAFM is needed.
- Limitations: B/2 80ep regime (not XL/2 1400ep paper scale); EqM-only (SiT head-to-head deferred to ICLR full paper).

### 6. Related work (~0.5 page)
- Dense citation map: AFM/CAFM (Lin), DAT (Wu), AEBM-Diff (Geng), ACT (Kong), VeCoR (Hong), Briglia, Rob-GAN, Madry, DCD (Luo).

## Figures (workshop)
1. **Method diagram**: vanilla EqM → +CAFM discriminator → +v10 PGD mining. Show losses.
2. **Ablation table** (Section 4.2 — the headline).
3. **Diagnostics curves**: L_base / L_hard / ratio / ||δ|| over training.
4. **CIFAR sanity**: FID curves over epochs, 4 conditions.
5. **Scaling preview** (S/2 IN-100): conditions × seeds → FID dots.

## What we need by submission (Aug 29)

- [ ] Phase 1b CAFM-only result (1 seed B/2 IN-256).
- [ ] Phase 2 v10+CAFM result (1 seed B/2 IN-256).
- [ ] Phase 3 3-seed ablation table (B/2 IN-256, 4 conditions × 3 seeds).
- [ ] Phase 3 EqM-S/2 IN-100 (3 conditions × 3 seeds).
- [ ] CIFAR sanity + diagnostics curves.
- [ ] Method writing.

## ICLR extension (Oct 1)

Adds:
- **SiT head-to-head** (Phase 5): apply v10+CAFM to SiT-B/2 IN-256 using Lin's pretrained ckpt. 3 seeds. Direct head-to-head with Lin's published CAFM numbers.
- **Larger scaling**: EqM-L/2 IN-100, 3 seeds × 4 conditions.
- **Full hyperparameter ablation**: λ × K × ε × mine_every grid (using S/2 IN-100 for compute).
- **Theoretical analysis**: equivariance vs invariance, connection to Lipschitz/Wasserstein bounds.
- **Negative results discussion**: which prior variants (v01-v09) failed and why.

## Open writing risks

1. Reviewer: "Why not just do v10+CAFM on SiT directly?" → Phase 5 head-to-head answers.
2. Reviewer: "B/2 80ep is too far from SOTA (1.90 XL/2 1400ep)." → Honest framing as ablation-regime improvement + Lin CAFM also reported B/2 results.
3. Reviewer: "Is v10 just dropout-equivalent / augmentation-equivalent?" → VeCoR comparison shows adversarial mining > augmentation negatives empirically.
4. Reviewer: "Briglia argues invariance fails — why doesn't v10?" → §3.3 + small-λ + ablation showing collapse at large λ.

## Hard deadlines

- 2026-08-15: 3-seed B/2 IN-256 ablation complete (critical).
- 2026-08-22: workshop draft complete.
- 2026-08-29: submit.
