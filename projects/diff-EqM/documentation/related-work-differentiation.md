# Related-Work Differentiation Memo

**Locked**: 2026-05-19
**Purpose**: Position v10 (and broader "adversarial training for regression-based implicit-energy generative models") cleanly against the two highest-overlap prior works.

## Prior work #1 — Wu et al. 2025 (DAT)

**Citation**: "Scalable Energy-Based Models via Adversarial Training: Unifying Discrimination and Generation". arxiv 2510.13872 (Oct 2025).

### Method
- Interprets classifier as classical EBM (JEM-style).
- Replaces SGLD with **PGD-generated contrastive samples**.
- Generative loss: Binary Cross-Entropy

  `L_BCE = -E_{x~p_data}[log σ(-E(x))] - E_{x~p_θ}[log(1 - σ(-E(x)))]`

  where σ = logistic sigmoid, contrastive x sampled by PGD on energy.
- PGD steps: T=15–70, step size η=2.0, **unconstrained perturbation** (used as sampling).
- Additionally has adversarial robustness loss `L_AT-CE` with L2-bounded PGD (ε=3.0 IN, ε=0.5 CIFAR).

### Results
- CIFAR-10: FID 7.57 (T=50)
- ImageNet-256: FID 5.39 (WRN-50-4, T=65); FID 6.64 (ResNet-50, T=15)
- Beats BigGAN-deep (6.95). First EBM-hybrid at ImageNet-256 scale.

### Sampling
- Inference = PGD-style gradient descent on energy. No regression / velocity field.

### Coverage
- Classical EBM only. No mention of equilibrium matching, flow matching, score-based, or any regression-based formulation.

## Prior work #2 — Geng et al. 2024 (ICML)

**Citation**: "Improving Adversarial Energy-Based Model via Diffusion Process". arxiv 2403.01666 (ICML 2024).

### Method
- Diffusion + EBM hybrid: separate EBM per denoising timestep.
- Generator-discriminator adversarial training (GAN-style), not PGD hard mining.
- Uses **symmetric Jeffrey divergence** + variational posterior for generator.
- No PGD; no hard-example mining.

### Results
- CIFAR-10 FID not extracted in our fetch (paper has detail; not blocking for differentiation).

### Coverage
- Adversarial-EBM-diffusion. Generator-driven adversarial loop, not hard-example mining.

## Other context

- Wang et al. 2025 EqM (2510.02300): the base model we extend. Establishes implicit-energy via regression on velocity-field target. FID 1.90 IN-256 with XL/2 full training. No adversarial training in the paper.
- Madry et al. 2017 (PGD adversarial training): foundational PGD-based hard-example training, but applied to classifier robustness, not generative modeling.

## Prior work #3 — Lin et al. 2026 (AFM + CAFM) — DOMINANT prior

**Citations**:
- Lin, Yang, Lin, Chen, Fan. *Adversarial Flow Models*. ICML 2026. arxiv 2511.22475.
- Lin, Yang, Lin, Chen, Fan. *Continuous Adversarial Flow Models*. arxiv 2604.11521 (Apr 2026).
- Affiliation: ByteDance Seed.
- Code: github.com/ByteDance-Seed/Adversarial-Flow-Models (MIT).

### Method (CAFM, the load-bearing version for our work)
- Post-training of pretrained flow matching models. Also from-scratch capable.
- JVP-based discriminator on the velocity field (discriminates in derivative space).
- Loss: least-squares GAN + centering penalty.
- N=16 discriminator updates per generator update. 10 ep post-training on top of 1400-ep FM baseline.
- IN-256 SiT-XL/2: FM 8.26 → CAFM 3.63 guidance-free, 1.53 with CFG.
- IN-256 JiT-H/16: 7.17 → 3.57 guidance-free, 1.80 with CFG.
- Pure GAN-style. **No PGD on input.** Gradient penalties are R1/R2 on discriminator (in AFM); CAFM eliminates them via continuous formulation.

### Coverage
- Standard flow matching (SiT, JiT, Z-Image). **No EqM application.** No PGD on input.

### Threat assessment
- **HIGH (dominant prior)**. Owns "adversarial training of regression-based flow matching" ground.
- Lin lab likely to publish follow-ups before Oct 1; weekly arxiv sweep mandatory.

### Why this is not a full scoop for our project
- They use GAN-style learned discriminator. We use PGD hard-example mining on the base regression target. **Mechanistically distinct.**
- They test SiT/JiT. We are first to apply CAFM-style AT to EqM family.
- They do not test combining PGD-on-input with discriminator AT. We do (Branch B).

## POST-2026-05-23 PIVOT: Discriminator-based vs Mining-based axis

Branch B-Both retired 2026-05-23 after CAFM-EqM Phase 1b FID 341.25 catastrophe (postmortem `postmortem-cafm-eqm-2026-05-23.md`). Project now **v10-only**.

The lit landscape (with AFM/CAFM/AAPT/V-PAE from Lin lab + EqM from Du group) partitions cleanly along a single axis:

| Axis | **Discriminator-based** | **Mining-based (ours)** |
|---|---|---|
| Method family | Adversarial 2-player (generator + discriminator) | Single-player regression with adversarially mined inputs |
| Examples | AFM, CAFM, AAPT, V-PAE, DAT, AEBM-Diff, Rob-GAN | **v10** (no prior on regression-target gen models) |
| Loss form | LSGAN / BCE / Jeffrey div on discriminator output | MSE on regression target evaluated at PGD-mined input |
| Collapse modes | Mode collapse, discriminator dominance, field-collapse | None (loss bounded below by 0; aux = base form) |
| Adversarial mechanism | Discriminator gradient | Input-space PGD on regression loss |
| EqM compatibility | **FAILS catastrophically** (CAFM-EqM FID 341.25 — fresh dis trivially crushes regression-trained gen) | **Works** (CIFAR 13.40 vs 14.17 PASS; IN-1K smoke-probe 78.91 at ckpt_65000 on healthy trajectory) |
| Best result | AFM XL/2 IN-256 FID 2.38 (Lin Nov 2025) | Phase 1 in flight: gate FID ≤ 30.41 vs vanilla 31.41 (B/2, 80ep) |
| Inference cost | Native 1NFE (AFM) or vanilla flow | Vanilla EqM gradient flow (no change) |

**Niche partition (post-2026-05-26 sweep)**:
- Lin lab (4 papers in 12mo) owns the discriminator-based niche outright.
- v10 owns the mining-based-regression-target niche — uncontested through workshop window (Aug 29).

**Methodological claim (revised post-pivot)**: First adversarial-style training for regression-target generative models that requires no discriminator, no two-player game, and cannot collapse to trivial solutions. NOT a FID-SOTA contender vs AFM 2.38; instead, a methodological contribution at the vanilla-EqM-comparable scale (B/2, 80ep).

The "v10+CAFM" composition columns below are HISTORICAL — preserved for record but no longer the current positioning. The v10-only row remains active.

---

## v10 — our positioning (Branch B-Both — HISTORICAL; see above for current post-pivot positioning)

### Method
- Same EqM regression target: `target = (ε - x) · c(γ)`, `L_base = ||f(x_γ) - target||²`.
- **Auxiliary** PGD hard-example loss: `δ* = argmax_{||δ||≤ε_rad} ||f(x_γ+δ) - target||²`, with K small PGA steps.
- Total: `L = L_base(x_γ) + λ · L_base(x_γ + δ*)`.
- L2-bounded perturbation (hard-example sense, not unconstrained sampling).
- K = 3, ε_rad ~ 0.3, λ ~ 0.1 (CIFAR-validated regime from v02).

### Differentiation table

| Axis | DAT (Wu 2025) | AEBM-Diff (Geng 2024) | AFM/CAFM (Lin 2026) | **v10+CAFM (ours, Branch B)** |
|---|---|---|---|---|
| Model family | Classical (classifier-as-EBM) | Timestep-conditioned diffusion EBM | Flow matching (SiT/JiT) | **EqM + SiT (Phase 5)** |
| Adversarial signal | BCE on PGD negatives | Generator-discriminator Jeffrey div | Learned discriminator (LSGAN) on velocity | **Discriminator (CAFM) + PGD hard mining on regression target (v10)** |
| PGD role | Sampling + generate negatives | none | none | **Hard-example mining for L_base** |
| PGD bound | Unconstrained | n/a | n/a | **L2-bounded (ε~0.3, K=3)** |
| Inference | PGD on energy | Reverse diffusion | Vanilla FM gradient flow | **Vanilla EqM gradient flow** |
| Tested models | Classical EBM only | Diff-EBM only | SiT/JiT only | EqM (Phases 1–4) + SiT (Phase 5) |

### Novelty claim (Branch B-Both, single sentence — POST-LIT-REVIEW REVISION)

> **First adaptive hard-negative mining for regression-target generative models.** We show that PGD-mined hard examples on the velocity-field regression target (v10) compose with CAFM-style discriminator post-training (Lin et al. 2026 CAFM) to compound FID gains on Equilibrium Matching (Wang & Du 2025). Discriminator catches global distributional mismatch; PGD-mining catches local regression failure. Phase 5 head-to-head on SiT confirms transfer to standard flow matching.

**External validation**: VeCoR (Hong et al. 2025, arxiv 2511.18942) §7 explicitly lists "adaptive hard-negative mining" as open future work for velocity contrastive regularization. v10 is precisely that direction. Citable verbatim in paper introduction.

**Mechanism note**: v10 is invariance-flavored vs Briglia 2025's equivariance prescription. v10 anchors to ground-truth target (not clean output), making it "robust regression" rather than pure invariance. Mitigation: small λ default, equivariant fallback in `v11_fallback_sketch.md`.

### Why this matters mechanically (paper-section sketch)

Three discriminating properties hold for v10 but not for DAT / Geng:
1. **Preserves the EqM target geometry**: the auxiliary loss is the base loss evaluated at perturbed inputs, so c(γ) decay near the data manifold is preserved (DAT's BCE loss has no c(γ) analog; Geng's per-timestep EBMs are differently parameterized).
2. **No extra inference-time cost**: v10 modifies training only; inference is vanilla EqM gradient flow. DAT's sampling = PGD on energy (T=15–70 steps per sample); Geng requires the learned generator.
3. **Compatible with single-pass training of regression models**: no generator network (vs Geng), no discriminative head (vs DAT). One regression model, one extra forward pass at each step.

### Risk: convergence of approaches

If a follow-up to DAT extends BCE-PGD to regression-target EBMs (e.g., score matching with PGD negatives), the novelty narrows to "EqM-specific instantiation + flow-matching transfer." Mitigation:
- Lock the differentiation language now (this memo).
- Get the empirical result locked by Aug 15 (workshop deadline buffer).
- Weekly arxiv sweep for parallel work (keyword: "adversarial" + "score matching" / "flow matching" / "equilibrium matching" / "regression EBM").

## arxiv weekly-sweep keywords

- "adversarial energy-based"
- "adversarial training" + "diffusion"
- "PGD" + "score matching"
- "hard example mining" + "generative"
- "equilibrium matching"
- "flow matching" + "adversarial"
- watch raywang4 (EqM author), Wu et al. (DAT authors) for follow-ups

## Citations to include in paper

- Wang et al. 2025, Equilibrium Matching (base model).
- Wu et al. 2025, DAT (closest PGD-EBM prior).
- Geng et al. 2024, AEBM-Diff (adversarial-EBM-diffusion prior).
- Madry et al. 2017, PGD adversarial training (foundational PGD).
- Lipman et al. 2023, Flow Matching (Phase 4 transfer model).
- Du & Mordatch 2019, Implicit generation and modeling with EBMs (classical EBM-CD).
- Grathwohl et al. 2020, JEM (classifier-as-EBM, basis for DAT).
- Luo 2023, Diffusion Contrastive Divergence (mining at noised inputs — related to our t-sampling).
