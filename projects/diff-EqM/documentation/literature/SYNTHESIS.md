# Literature Synthesis — Phase 0

**Written**: 2026-05-19
**Reads completed**: 13 papers + 4 watch-list entries (Round 1 + 2).
**Outstanding (low priority)**: DiT (Peebles), OHEM, Focal, FAIL, FMVP.

## 1. Landscape map

The "adversarial training for regression-target generative models" space has crystallized in the last 18 months. Three loci:

- **Discriminator-based AT on flow matching** (Lin lab ByteDance): AFM (ICML 2026, arxiv 2511.22475) + CAFM (arxiv 2604.11521). LSGAN-style. SOTA at IN-256 (SiT-XL/2 1.53 FID with CFG). Public MIT code. **3 papers in 12 months; high publication velocity**.
- **PGD-based AT on classifier-as-EBM** (Wu et al. DAT, MIT, arxiv 2510.13872). BCE-on-PGD-negatives. Classical EBM. SOTA in classical EBM family (5.39 FID IN-256). Not flow/EqM.
- **Discriminator-based AT on consistency models** (Kong ACT, CVPR 2024). Discriminator + consistency loss. No PGD on input.
- **Contrastive (augmentation-based) supervision on velocity** (VeCoR Hong et al. JIIOV, arxiv 2511.18942). Two-sided cosine. SiT-XL/2 22% relative FID gain. **Authors explicitly list "adaptive hard-negative mining" as open future work.**

Gap remaining: **adaptive (adversarial) hard-negative mining on the regression target itself**, AND combining it with discriminator-based AT.

Adjacent priors:
- Briglia 2025 (arxiv 2505.21742): theoretical argument that AT for diffusion must be **equivariance**, not **invariance**. Naive invariance loss collapses (FID 200+).
- Du & Mordatch 2019: classical EBM with MCMC negatives. Direct lineage to EqM (Wang+Du 2025).
- Rob-GAN 2018: oldest "PGD + GAN" combination prior. Defensive focus.

## 2. Updated positioning (≤ 50 words)

> First **adaptive hard-negative mining** for regression-target generative models (Equilibrium Matching, Flow Matching). PGD-mined hard examples on the velocity-field regression target (v10) compose with CAFM-style discriminator post-training (Lin 2026) to compound FID gains. Two losses attack complementary failure modes: discriminator catches global distributional mismatch; PGD-mining catches local regression failure.

**Why this framing wins:**
- VeCoR's own §7 future-work statement is direct support: they explicitly call out adaptive hard-negative mining as open. Citable in introduction.
- "Adaptive" distinguishes from VeCoR's heuristic augmentation negatives.
- "Regression-target" distinguishes from DAT's discriminative BCE negatives.
- "Compose with CAFM" distinguishes from any single-component prior.
- Two contributions: (i) v10 alone novel as ANM; (ii) v10+CAFM novel as composition.

## 3. Mechanism design adjustments

### 3.1 Add equivariant-v10 variant to fallback slate (Briglia threat)
Current v10 = invariance-flavored: `||f(x_γ + δ*) − target||²`, fixed target.

**Briglia warns invariance fails for diffusion.** v10 is technically anchored to ground-truth target so it's "robust regression" not pure invariance, but the failure mode they describe (Figure 3, FID 200+) could still materialize at aggressive λ.

**Decision**:
- Keep v10 invariance form as PRIMARY (since it's anchored to true target, not clean prediction).
- Use **small initial λ = 0.1** (Briglia's stable regime is λ=0.3 on noise-conditioned diffusion; ours has different target).
- Add **v10-equivariant** variant to v11 fallback list: `||f(x_γ + δ) − f(x_γ).detach() − J·δ||²` or simpler `||f(x_γ + δ) − f(x_γ).detach()||² + ...` paired with target loss. (Variants to explore if v10 collapses.)
- CIFAR sanity (Phase 0.C.2 — wait, we removed that; CIFAR sanity is now in Phase 1) must include early-collapse diagnostic.

### 3.2 Reduce K to 1 (FGSM) as fast-iteration option
Briglia uses single-step FGSM, not multi-step PGD. Computationally much cheaper.

**Decision**:
- Test v10 with K=1 (FGSM) as compute-saving option in CIFAR sanity.
- If K=1 mining produces useful signal (L_hard > L_clean), prefer it for IN-1K (reduces mining cost from 4× forward passes to 2×).
- If K=1 inert (mining gradient too weak), keep K=3.

### 3.3 EqM has no time conditioning — CAFM discriminator needs adaptation
EqM removes time conditioning (input t=0 fixed). Lin CAFM's discriminator is `D(x_t, t)` — time-conditioned.

**Decision**:
- Phase 1a CAFM-to-EqM port must adapt discriminator to either:
  - Take γ ∈ [0,1] as input (EqM's mixing coefficient, since γ is still a sampling axis even if not fed to f).
  - Or be γ-unconditional (simpler but may sacrifice gain).
- Recommend: γ-conditional discriminator with γ embedded via standard sinusoidal/learned embedding. Aligns with ACT/CAFM convention.

### 3.4 γ-dependent λ scaling (Briglia-inspired)
Briglia's `λ_t = c · σ(t)⁻¹` scales adversarial weight inversely with noise level. EqM analog: scale λ by `1/c(γ)` (upweight aux loss when target magnitude is small).

**Decision**:
- Default v10 uses constant λ.
- Add γ-weighted variant `λ(γ) = λ₀ / max(c(γ), 0.1)` to Phase 0.3 design doc for potential ablation.

### 3.5 Compute budget reality check
Lin CAFM uses **N=16 discriminator updates per generator update**. Vanilla EqM 80ep = ~24h × 4 A100s = 96 GPU-h. CAFM 10 ep post-training with N=16 ≈ 10×16 = 160 effective steps + base = ~120 GPU-h per seed. v10 + CAFM with K=3 PGA per gen step ≈ 1.3× more = ~160 GPU-h per seed.

3-seed × 4 conditions × ~160 GPU-h ≈ **1920 GPU-h** for the full Phase 3 ablation. Tight on 2400 budget; cushion via:
- K=1 if signal sufficient (saves ~25%).
- Drop "v10-only" arm (we measured at CIFAR already, IN-1K v10-only is mostly redundant).

## 4. New / sharpened risks

### 4.1 Update 2026-05-26 — post CAFM-EqM retire + AFM hit
- **Branch B-Both retired** 2026-05-23 (CAFM-EqM Phase 1b FID 341.25 catastrophe — postmortem). v10-only forward.
- **AFM (Lin Nov 2025, arxiv:2511.22475)** added to lit notes. Lin lab now 4 papers in 12mo (AFM/CAFM/AAPT/V-PAE), confirms discriminator-based adversarial-flow niche fully owned by that group. IN-256 XL/2 FID **2.38** one-step.
- **Niche partition**: discriminator-based (Lin lab) vs mining-based (v10). Mechanistically distinct; the two are not direct competitors on the same axis. v10 owns the mining-based-regression-target niche.

### 4.2 Risk table

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Lin lab publishes mining-based variant before Oct 1 | LOW (was M-H) | HIGH | They have 4-paper discriminator-track momentum; pivoting to mining unlikely in this window. Weekly arxiv sweep continues. |
| Du group publishes adversarial-EqM follow-up | M | HIGH | Du = senior author of EqM. Watch raywang4 + Yilun Du. EqM paper itself flagged "adversarial perturbations" as unexamined future work (external open-direction support for v10). |
| ~~v10 saturates / collapses at IN-1K scale~~ | **RESOLVED 2026-05-27** | n/a | Phase 1 gate PASS: FID 29.01 vs vanilla 31.41 (gain 2.40, gate cleared by 1.40). Mechanism transferred from CIFAR (gain 0.77) to IN-1K (gain 2.40) — scaled UP. |
| ~~Phase 1 gate fail (FID > 30.41)~~ | **RESOLVED 2026-05-27** | n/a | PASS by 1.40 FID margin. λ retune not triggered; λ=0.3 ablation running anyway for paper table. |
| Phase 2 Welch t fails (high seed variance) | M | HIGH | Seeds 1+2 v10 (16376973/74) + vanilla seeds 1+2 (16594167/73) all launched. If true seed sensitivity, v11 equivariant fallback. |
| Phase 3 scaling: mining doesn't transfer to S/2 or L/2 | M | MED | Smokes PASS at S/2 (aux/base 1.04) and L/2 (aux/base 1.000 — weak signal); L/2 may need ε_rad retune. XL/2 smoke confirms GBS=32 fits 4xA100-80GB. |
| ICLR reviewer asks "why B/2 80ep when paper does XL/2 1400ep?" | M | MED | Methodological framing: "first discriminator-free adversarial-style training for regression-target gen models". Cite AFM 2.38 as the discriminator-based SOTA point — different axis. |
| VeCoR adds adversarial mining in v2 | M | MED | Watch arxiv:2511.18942 versions. JIIOV has resources. |
| Bug invalidating CIFAR Phase 0.3 PASS or smoke-probe FID 78.91 | LOW | HIGH | Code path (`train_imagenet.py`) is trusted — same that produced vanilla FID 31.41. |

### 4.3 RESOLVED / OBSOLETE
- ~~"CAFM port to EqM doesn't transfer (time-conditioning mismatch)"~~ — CONFIRMED catastrophic 2026-05-23, branch retired
- ~~"Lin CAFM repo doesn't reproduce easily"~~ — moot; branch retired
- ~~"Our 31.41 baseline is invalid"~~ — RESOLVED (`eqm_baseline_reconciliation.md`)

## 5. Recommended edits

### CLAUDE.md
- Already updated with literature review protocol. No changes.

### summer-2026-plan.md — applied immediately below
- Sharpen claim to "First adaptive hard-negative mining for regression-target generative models" (per VeCoR future-work cite).
- Add γ-conditional discriminator to Phase 1a port design.
- Reduce default K to 1 in CIFAR sanity, K=3 only if K=1 inert.
- Add equivariant-v10 to v11 fallback (Task 0.4).
- Phase 3 ablation: drop "v10-only IN-1K" arm to save ~150 GPU-h; rely on CIFAR for v10-only signal.

### related-work-differentiation.md — applied below
- Strengthen "novelty claim" with VeCoR future-work citation.
- Add Briglia equivariance discussion.

### phase-0-spec.md — applied below
- Task 0.2 (CAFM-EqM port design) must address γ-conditional discriminator.
- Task 0.3 (v10+CAFM combination design) must address invariance-vs-equivariance choice + small-λ default.
- Task 0.4 must include equivariant-v10 as primary fallback.

### v10_hard_example_eqm_proposal.md
- Add Briglia citation in §Risks.
- Set primary λ_default = 0.1 (down from 0.1; same).
- Add `λ = γ-weighted` and `K = 1 (FGSM)` variants to HP sweep.

## 6. Citation map

| Paper | Intro | Related work | Method | Discussion |
|---|---|---|---|---|
| Wang+Du 2025 EqM | base model | base | base | — |
| Lin 2026 AFM/CAFM | competitor | primary prior | discriminator design | head-to-head |
| Wu 2025 DAT | PGD-EBM prior | discriminative AT family | — | classical-EBM comparison |
| Geng 2024 AEBM-Diff | adversarial-EBM prior | gen-disc EBM family | — | — |
| Kong 2024 ACT | discriminator+regression prior | consistency family | — | future work direction |
| VeCoR 2025 | direct motivation (future-work cite) | velocity contrastive | — | adaptive mining advance |
| Briglia 2025 AT-for-diffusion | mechanism reference | equivariance | invariance-vs-equivariance | v10-equivariant alt |
| Madry 2017 PGD | foundational | PGD foundational | — | — |
| Rob-GAN 2018 | combination prior | "PGD+GAN" lineage | — | — |
| Du 2019 implicit EBM | EBM lineage | — | — | — |
| Grathwohl 2020 JEM | classifier-EBM | — | — | — |
| Lipman 2023 FM | regression-target lineage | — | — | — |
| Ma 2024 SiT | architecture | — | — | Phase 5 |
| Song 2023 Consistency | regression family | — | — | future work |
| Luo 2023 DCD | mining prior | — | — | — |
| Lin 2025 AAPT | Lin lab pattern | — | — | watch |

## 7. Workshop / ICLR paper outline (draft)

**Title (draft)**: "Adaptive Hard-Negative Mining for Regression-Target Generative Models"

**Sections**:
1. Intro — VeCoR future-work statement + Lin CAFM SOTA → gap = adaptive mining + composition.
2. Background — EqM, flow matching, CAFM, VeCoR, ACT.
3. Method
   - v10: PGD on L_base, L2-bounded.
   - v10+CAFM combination: combined loss + training schedule.
   - EqM-specific adaptations: γ-conditional discriminator, c(γ) considerations.
4. Experiments
   - CIFAR sanity (stability + diagnostics).
   - EqM-B/2 IN-256 80ep ablation table (4 conditions × 3 seeds).
   - Scaling on EqM-S/2 IN-100 (3 conditions × 3 seeds).
   - SiT head-to-head (Phase 5, ICLR only).
5. Discussion — invariance-vs-equivariance, when discriminator vs mining helps, limitations.
6. Related work.

## 8. Stop / continue decision

**CONTINUE Branch B-Both.** All findings align with the planned framing. Mechanism is mechanistically distinct from all 13 read priors. VeCoR future-work statement is strong external validation.

**Adjustments locked**: equivariant fallback + smaller-K default + γ-conditional discriminator + sharpened positioning per VeCoR.

**Next concrete action**: apply edits to plan + spec + memo. Then Phase 0.C.1 (clone Lin repo) and Phase 0.2 (CAFM-EqM port design doc).
