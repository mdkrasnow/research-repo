# Workshop Paper Draft — §4 Experiments

Status: skeleton v1, 2026-05-20. Numerical placeholders [X], [Y], [Z], [W] fill as Phase 1b/2/3 results arrive.

---

## 4. Experiments

### 4.1 Experimental Setup

**Base model.** Equilibrium Matching (EqM-B/2, Wang & Du 2025) — 130M parameters, SiT-XL/2-style transformer backbone with time conditioning disabled (input ``t = 0``), trained on 32×32×4 SD-VAE latents.

**Datasets.** ImageNet-1K at 256×256 (paper-scale) and CIFAR-10 at 32×32 (sanity).

**Training.** Vanilla EqM baseline: 80 epochs on ImageNet-1K, 4×A100 DDP, global batch 256. Adversarial post-training (CAFM-only, v10-only, v10+CAFM): 10 epochs initialized from the 80-epoch vanilla checkpoint. CAFM hyperparameters per Lin et al. 2026: Adam ``β=(0, 0.95)``, learning rate 1×10⁻⁵, EMA decay 0.99, N=16 discriminator updates per generator update, ``λ_cp=0.001`` centering penalty, ``λ_ot=0`` for post-training. v10 hyperparameters: ``λ=0.1``, ``K=1`` (FGSM-style), ``ε_rad=0.3``, ``η=0.05``.

**Evaluation.** Frechet Inception Distance (FID) on 50K samples using EqM's native NAG-GD sampler (η=0.0017, μ=0.3, 250 NFE) against ImageNet-1K reference statistics computed via Inception-V3.

**Trusted baseline.** Vanilla EqM-B/2 80-epoch FID = **31.41** on ImageNet-1K-256 (paper-comparison Δ = −1.44 vs paper-reported 32.85 at the same configuration).

### 4.2 CIFAR-10 sanity (Phase 0.3)

Stability check on CIFAR-10 with the variant harness (Wang & Du-style FM UNet, 55.68M parameters, ``c(γ)`` truncated decay matched to ImageNet). 150 epochs, batch 128, learning rate 2×10⁻⁴, single seed. FID computed on 5,000 EMA samples via Euler ODE 100 steps.

| Method | Final FID |
|---|---|
| Vanilla EqM (v00, reference) | 14.17 |
| **v10 (ours)** | **13.40** |

v10 beats vanilla by 0.77 FID. The mining ratio ``L_v10 / L_EqM`` remains stable in [1.04, 1.05] across all 150 epochs, indicating PGA finds non-trivial hard examples at every step without saturation. We compare this trajectory to the saturation signature of cosine-contrastive objectives (v02, Jiang et al.-style) in §A.1 of the appendix; on EqM-B/2 at ImageNet-1K scale, v02 saturated to ``cos ≈ 1.0`` within 9 epochs (``|v|=220``, ``cos(v, v_neg)=0.999`` constant), motivating our pivot to v10's invariance-flavored regression formulation.

Per CLAUDE.md research protocol, the CIFAR result is a stability check only, not a publishable claim. The headline result is the ImageNet-1K ablation (§4.3).

### 4.3 ImageNet-1K-256 ablation (Phase 1-3)

Four conditions, three seeds each, on EqM-B/2 with 10-epoch adversarial post-training initialized from the vanilla 80-epoch trusted baseline:

| Condition | Mean FID @50K | Std (3 seeds) | Δ vs vanilla |
|---|---|---|---|
| Vanilla EqM (no post-training) | 31.41 | — | — |
| v10-only (no discriminator) | [X] | [σ_X] | [Δ_X] |
| CAFM-only (Lin et al. recipe, EqM-adapted) | [Y] | [σ_Y] | [Δ_Y] |
| **v10 + CAFM (ours)** | **[Z]** | [σ_Z] | **[Δ_Z]** |

Welch t-test, v10+CAFM vs CAFM-only: p = [p_value], mean improvement [Δ_compound] FID. Pre-registered Phase 3 exit gate: compound mean improvement ≥ 0.5 FID at p < 0.10.

#### Diagnostic-level evidence of complementarity

For each condition we log: generator loss ``L_EqM``, discriminator loss ``L_D`` (CAFM-only and combined), mining loss ``L_v10`` (v10-only and combined), discriminator-on-EMA logits, and per-step mean ``‖δ‖``. Two key non-redundancy diagnostics:

1. **L_v10 remains nonzero in combined runs.** If the discriminator's gradient subsumed PGD-mining's signal, we would expect ``L_v10 → 0`` as training proceeds. We observe instead [behavior_TBD], indicating the two losses constrain different model errors.
2. **L_D trajectory differs between CAFM-only and v10+CAFM.** [trajectory_TBD] — if v10 closed the easy-to-find regression gaps, the discriminator's signal-to-noise should shift.

### 4.4 Hyperparameter ablations on EqM-S/2 ImageNet-100 (Phase 3.2)

Smaller-scale ablations using EqM-S/2 (33M parameters) on ImageNet-100 (100-class subset), 3 seeds each:

- **λ (v10 weight):** {0.03, 0.1, 0.3, 1.0}. Default 0.1. We expect FID minimum at the smallest λ that still produces meaningful mining signal.
- **K (PGA steps):** {1, 3, 5}. Default K=1 (FGSM); we test whether multi-step PGD finds qualitatively different hard examples or is redundant overhead per Briglia et al.'s single-step finding.
- **ε_rad (L2 perturbation budget):** {0.1, 0.3, 0.6}. Default 0.3 (CIFAR-validated). Larger ε explores farther from data; smaller stays closer.
- **mine_every (mining frequency):** {1, 4, 16}. Default 1 (every generator step); higher values amortize compute cost.

Compute: 4 sweep dimensions × 3 settings × 3 seeds = 36 runs at S/2 IN-100, ~6h each = ~216 GPU-h.

### 4.5 Scaling on EqM-L/2 ImageNet-100 (Phase 3.3)

Three conditions (vanilla, CAFM-only, v10+CAFM) × 3 seeds on EqM-L/2 (458M parameters). Tests whether the FID improvement persists at larger model scale or is B/2-specific. ~600 GPU-h.

### 4.6 SiT head-to-head (Phase 5, ICLR extended results)

Reproduce Lin et al.'s CAFM-only on SiT-XL/2 IN-256 using their pretrained checkpoint; then apply v10 mining on top of CAFM in the same recipe. Three seeds each. Direct head-to-head against Lin's published numbers:

| Method | FID guidance-free | FID with CFG |
|---|---|---|
| SiT-XL/2 1400ep (Lin reference) | 8.26 | 2.06 |
| + CAFM 10ep (Lin published) | 3.63 | 1.53 |
| + CAFM 10ep (our reproduction) | [Y'] | [Y''] |
| + **v10 + CAFM (ours)** | **[Z']** | **[Z'']** |

Pre-registered Phase 5 gate: v10+CAFM beats CAFM-only by ≥ 0.3 FID on at least one of guidance-free or CFG conditions. ICLR-only result.

### 4.7 Compute summary

| Phase | Hardware | Wall (per run) | Runs | GPU-h |
|---|---|---|---|---|
| Phase 0.3 v10 CIFAR sanity | 1× H200 | 6.5h | 1 | 6.5 |
| Phase 1a CAFM-EqM smoke | 4× A100 | 0.5h | 1 | 2 |
| Phase 1b CAFM-only seed 0 | 4× A100 | ~30h | 1 | 120 |
| Phase 2 v10+CAFM seed 0 | 4× A100 | ~35h | 1 | 140 |
| Phase 3.1 seeds 1, 2 × 2 conditions | 4× A100 | ~32h | 4 | 512 |
| Phase 3.2 S/2 IN-100 ablations | 4× A100 | ~6h | 36 | 864 |
| Phase 3.3 L/2 IN-100 × 3 conditions × 3 seeds | 4× A100 | ~20h | 9 | 720 |
| Phase 5 SiT-XL/2 × 2 conditions × 3 seeds | 4× A100 | ~40h | 6 | 960 |
| Buffer (~15%) | — | — | — | ~520 |
| **Total** | | | | **~3850** |

Within 4000 GPU-h headroom; Phase 5 (SiT) is the largest single phase and can be down-scoped if needed.

---

## Figure inventory (to produce)

1. **Method overview** (§3, Figure 1) — pipeline diagram: vanilla EqM → +CAFM discriminator branch → +v10 PGD branch, with arrows showing both gradients reaching the same generator.
2. **CIFAR-10 sanity** (§4.2, Figure 2) — FID-vs-epoch curves for v10 and v00 on the variant harness. Show training trajectory + mining ratio overlay.
3. **ImageNet-1K headline** (§4.3, Figure 3) — bar chart of 3-seed means with error bars for 4 conditions. Annotate Welch t p-value.
4. **Diagnostics** (§4.3, Figure 4) — multi-panel: L_v10 over training, L_D over training, mean ``‖δ‖``, ratio. One panel per condition.
5. **HP ablation heatmaps** (§4.4, Figure 5) — λ × K grid showing FID at S/2 IN-100.
6. **Scaling curve** (§4.5, Figure 6) — FID vs parameter count (S/2, B/2, L/2) for vanilla / CAFM / v10+CAFM.
7. **SiT head-to-head** (§4.6, Figure 7) — same as Figure 3 but on SiT-XL/2 (ICLR extended results).

---

## Notes for revision

- §4.2 v00 reference: use exact same harness (variant_pilot), not legacy cifar_seed_study (4.7 FID gap documented).
- §4.3 add 50K-FID-vs-1K-FID note: our 1K eval during training is rough; report 50K for all final numbers.
- §4.4 if Phase 1b times out, drop S/2 ablations to S/2 minimal (λ + K only).
- §4.6 SiT result is ICLR-only; clearly mark in workshop draft.
- Appendix A.1: cosine-saturation diagnostic from v02 IN-1K run 10198798 — copy figure from `debugging.md`.
