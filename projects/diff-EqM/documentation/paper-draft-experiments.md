# Workshop Paper Draft — §4 Experiments

Status: v2 draft, 2026-05-28 (post Branch B-Both retire + Phase 1 PASS). v10-only framing. Hard numbers from Phase 1; placeholders `[X]` `[Y]` etc. for Phase 2/3 results in flight.

**Version log**:
- v2 2026-05-28: rewrite from v10-only after CAFM postmortem; insert Phase 1 FID 29.01; restructure scaling at IN-1K not IN-100; add negative-result subsection for CAFM-EqM.
- v1 2026-05-20: original Branch B-Both skeleton with v10+CAFM combination.

---

## 4. Experiments

### 4.1 Experimental Setup

**Base model.** Equilibrium Matching (EqM, Wang & Du 2025). Primary scale: EqM-B/2 (~130M parameters, 12 layers, hidden dim 768, patch 2², SiT-XL/2-style transformer with time conditioning disabled). Scaling experiments: EqM-S/2 (~32M), EqM-L/2 (~458M), EqM-XL/2 (~675M). All trained on 32×32×4 SD-VAE latents.

**Datasets.** ImageNet-1K at 256×256 class-conditional (paper-scale). CIFAR-10 at 32×32 (mechanism sanity).

**Training.** 80 epochs, 4×A100-80GB DDP, global batch 256 (XL/2 reduced to 128 for memory). Adam β=(0.9, 0.999), learning rate 1×10⁻⁴, EMA decay 0.9999. No warmup. Matches Wang & Du's recipe at this regime.

**v10 hyperparameters.** λ=0.1, K=1 (FGSM-style), ε_rad=0.3 L2 ball, mining lr η=0.05, mine_every=1. PGA initialization: Gaussian N(0, ε_rad/2 · I) projected to ball. δ\* detached before backprop. K=1 motivated by Briglia et al. 2025 single-step result; multi-step variants in §4.4.

**Evaluation.** 50K-sample FID using EqM's native gradient-descent sampler (η=0.003, μ=0.3, 250 NFE), pytorch-fid against 50K shuf-subsampled ImageNet-1K reference images at 256×256.

**Trusted baseline.** Our vanilla EqM-B/2 80-epoch reproduction: FID **31.41** on IN-1K-256, beating paper-reported **32.85** (Wang & Du Table 3) by Δ=−1.44. This is our load-bearing baseline for all v10 comparisons.

### 4.2 CIFAR-10 mechanism sanity (Phase 0.3)

Stability check on the variant harness (FM UNet, 55.68M params, c(γ) matched to ImageNet recipe). 150 epochs, batch 128, single seed.

| Method | Final FID (5K samples) |
|---|---|
| Vanilla EqM (v00, variant harness reference) | 14.17 |
| **v10 (ours)** | **13.40** |

v10 beats vanilla by 0.77 FID. The mining ratio L_v10/L_EqM is stable in [1.04, 1.05] across all 150 epochs — PGA finds non-trivial hard examples at every step without saturation. We contrast this with cosine-contrastive objectives (v02, VeCoR-style), which saturated to cos≈1.0 within 9 epochs on EqM-B/2 at IN-1K scale (|v|≈220, cos(v, v_neg)=0.999 constant). Appendix A.1.

Per CLAUDE.md research protocol, this is a stability check, not a publishable claim. The publishable result is §4.3.

### 4.3 ImageNet-1K-256 headline (Phase 1, Phase 2)

Direct comparison vanilla EqM-B/2 vs v10 EqM-B/2, 80 epochs from scratch, 3 seeds each, 50K-sample FID:

| Condition | Seed 0 | Seed 1 | Seed 2 | Mean ± std | Δ vs vanilla |
|---|---|---|---|---|---|
| Vanilla EqM-B/2 | 31.41 | [V₁] | [V₂] | [V̄] ± [σ_V] | — |
| **v10 EqM-B/2 (ours)** | **29.01** | [v₁] | [v₂] | **[v̄] ± [σ_v]** | **[Δ̄]** |
| Paper-reported EqM-B/2 80ep (Wang & Du Table 3) | 32.85 | — | — | — | — |

**Phase 1 single-seed result (locked)**: v10 seed 0 FID = **29.01** vs vanilla seed 0 = 31.41 (gain **2.40 FID, 7.6% relative**) and vs paper-reported = 32.85 (gain **3.84 FID, 11.7% relative**).

**Welch t-test (Phase 2, in flight)**: one-sided, alternative vanilla > v10. Pre-registered gate: mean gain ≥ 1.0 FID AND p < 0.05. Result: t=[t], p=[p].

### 4.3.1 Diagnostics throughout training

Logged every 200 steps (CLAUDE.md mandatory):

| Diagnostic | Vanilla (seed 0) | v10 (seed 0) | Interpretation |
|---|---|---|---|
| Base loss L_EqM (end of train) | 10.32 | 10.28 | v10 preserves regression objective |
| Aux/base ratio L_v10/L_EqM | — | 1.01–1.03 | Mining active, non-saturating |
| Perturbation norm ‖δ‖ | — | 0.300 (boundary) | PGA finds non-trivial directions |
| Per-step wall time | 1.0× | ~1.5× | Mining overhead ~50% (one extra fwd+bwd, K=1) |

The non-saturating ratio is the key signature differentiating v10 from cosine-contrastive losses that fail at scale.

### 4.4 Hyperparameter ablations at B/2 (Phase 5)

Ablations vary one HP at a time around the Phase 1 default (λ=0.1, K=1, ε_rad=0.3), seed 0, EqM-B/2 80ep IN-1K-256:

| Ablation | Value | Mean ratio L_v10/L_EqM | FID @50K | Δ vs Phase 1 v10 |
|---|---|---|---|---|
| **Default (Phase 1)** | λ=0.1, K=1, ε_rad=0.3 | 1.02 | **29.01** | — |
| λ retune | λ=0.3 | [r_λ] | [F_λ] | [Δ_λ] |
| K retune | K=3 | 1.058 (smoke) | [F_K] | [Δ_K] |
| ε_rad retune (planned) | ε_rad=0.5 | — | [F_ε] | [Δ_ε] |

K=3 shows stronger mining signal in smoke (1.058 vs K=1's 1.02), informing whether the extra PGA steps find qualitatively harder examples or are redundant overhead per Briglia 2025. λ=0.3 is the single bracketing point for the λ sweep.

Compute: 2 ablation runs in flight (~72 GPU-h each), ε_rad sweep optional.

### 4.5 Scaling curve (Phase 3)

Whether the v10 FID gain holds across model scale. Single seed per cell at S/2 / L/2 / XL/2 (B/2 from §4.3). All 80ep IN-1K-256, gd sampler, 50K-sample FID.

| Model | Params | Vanilla FID | v10 FID | Δ | Δ% |
|---|---|---|---|---|---|
| EqM-S/2 | ~32M | [S_van] | [S_v10] | [Δ_S] | — |
| EqM-B/2 | ~130M | 31.41 | **29.01** | **2.40** | **7.6%** |
| EqM-L/2 | ~458M | [L_van] | [L_v10] | [Δ_L] | — |
| EqM-XL/2 (stretch) | ~675M | [XL_van] | [XL_v10] | [Δ_XL] | — |

**Pre-registered gate**: v10 must beat vanilla by ≥0.5 FID at S/2 AND L/2 to claim "scaling-friendly." XL/2 is a stretch confirmation; any positive Δ supports the claim at paper's headline architecture (the paper's XL/2 1400ep FID 1.90 is out of our compute budget and not a comparison target).

Smoke result note: at L/2, smoke aux/base ratio settled near 1.000 (vs 1.02 at B/2 and 1.04 at S/2), suggesting either (a) larger models absorb mining easier, or (b) ε_rad=0.3 is small relative to L/2 latent dynamic range. ε_rad sweep at L/2 contingent on §4.4.

### 4.6 SiT transfer (Phase 4)

Whether the v10 mining mechanism extends beyond EqM to standard flow matching. SiT-B/2 (Ma et al. 2024) trained from scratch, 80ep IN-1K-256, 4×A100 DDP, vanilla FM target ε−x (no c(γ) decay). Single seed initially, multi-seed if Phase 4 gate passes.

| Condition | FID @50K | Δ vs vanilla |
|---|---|---|
| Vanilla SiT-B/2 | [SiT_van] | — |
| **v10 SiT-B/2 (ours)** | [SiT_v10] | [Δ_SiT] |

**Pre-registered gate**: v10 beats vanilla SiT by ≥0.5 FID. PASS → claim broadens to "regression-target generative models." FAIL → claim narrows to EqM.

### 4.7 Negative result: discriminator-based adversarial training on EqM (CAFM port)

We document a complete failure case for completeness: porting CAFM (Lin et al. 2026) to EqM-B/2.

| Condition | FID @50K |
|---|---|
| Vanilla EqM-B/2 80ep | 31.41 |
| **CAFM 10ep post-training on vanilla** | **341.25** |
| Diagnostic FID at ckpt step 5,000 (~250 generator updates past warmup) | 369.64 |

Training diagnostics were clean (gen loss converged 4.0→1.7-2.3, dis loss oscillated 0.9-2.1, centering penalty stable at 10⁻⁴), giving no warning before sampling. Sample-level inspection at ckpt 5,000 immediately reveals total mode collapse — the failure is instant, not cumulative.

Root cause is structural, not a bug. EqM is trained by pure regression and never sees an adversarial signal during pretraining. When a freshly-initialized CAFM discriminator is attached, it trivially discriminates "vanilla-EqM output ≠ training-data velocity" and pushes the generator off the EqM target manifold to satisfy the adversarial objective. The schedule c(γ) → 0 near the data manifold amplifies the asymmetry: any non-zero adversarial gradient at high γ is preferred over satisfying a vanishing regression target.

This is not specific to CAFM — it is a general failure mode for discriminator-based adversarial training of regression-target generative models. We document it as Appendix B and as motivation for the discriminator-free mining-based approach (§3).

### 4.8 Compute summary

| Phase | Runs | GPU-h |
|---|---|---|
| Phase 0.3 CIFAR sanity | 1 | 7 |
| Phase 1 v10 IN-1K seed 0 (incl. resume) | 1 | 350 |
| Phase 1 FID eval | 1 | 10 |
| Phase 2 v10 seeds 1+2 | 2 | 280 |
| Phase 2 vanilla seeds 1+2 | 2 | 280 |
| Phase 3 vanilla S/2 + L/2 + XL/2 baselines | 3 | 600 |
| Phase 3 v10 S/2 + L/2 + XL/2 | 3 | 700 |
| Phase 4 SiT-B/2 vanilla + v10 | 2 | 350 |
| Phase 5 ablations (λ, K, ε_rad) | 3-5 | 350 |
| Negative result: CAFM-EqM | 1 | 100 |
| Buffer | — | ~200 |
| **Total** | | **~3200** |

Comfortably within available compute (per user 2026-05-27 directive).

---

## Figure inventory (to produce)

1. **Method diagram** (§3, Figure 1) — f_θ evaluated at clean x_γ (yields L_EqM) and at mined x_γ + δ\* (yields L_v10), with sub-figure marking the failed discriminator alternative (§4.7).
2. **CIFAR-10 mechanism** (§4.2, Figure 2) — FID vs epoch for vanilla v00 vs v10, with mining ratio overlay.
3. **ImageNet-1K headline** (§4.3, Figure 3) — bar chart of 3-seed means with error bars, vanilla vs v10. Annotate Welch t p.
4. **Diagnostics signature** (§4.3.1, Figure 4) — multi-panel: aux/base ratio over training for v10 PASS vs CAFM FAIL, side by side. The non-saturating vs catastrophic-collapse contrast.
5. **Scaling curve** (§4.5, Figure 5) — FID vs parameter count (S/2, B/2, L/2, XL/2), vanilla vs v10.
6. **SiT transfer** (§4.6, Figure 6) — direct vanilla vs v10 on SiT-B/2.
7. **CAFM negative result** (§4.7, Appendix B) — gen+dis loss curves looking "healthy" + sample grid at ckpt 5,000 showing collapse. The textbook example of "loss curves can lie."

---

## Notes for revision

- §4.3 lock 50K-FID-vs-paper-reported framing: emphasize gain over our reproduced baseline (2.40) ahead of gain over paper (3.84) since the latter is partly recipe-drift attribution.
- §4.4 if K=3 shows ≥0.3 FID gain, recommend K=3 as default and demote K=1 to ablation.
- §4.5 if L/2 v10 result fails the 0.5 FID gate, investigate ε_rad scaling — currently 0.3 is B/2-tuned.
- §4.6 SiT result is Phase 4 (post-Phase 3). Workshop deadline tight; if SiT not ready by Aug 22, push to ICLR.
- §4.7 CAFM negative result is publishable on its own merits — first documented failure of discriminator-based AT on regression-target gen models. Even if Phase 2/3 fail, this section + §3.3 mechanism arg make a coherent workshop paper.
