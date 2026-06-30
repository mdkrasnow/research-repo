---
name: diff-eqm-framing-branch-b-both
description: "diff-EqM project framing locked 2026-05-19 as Branch B-Both — first adaptive hard-negative mining for regression-target generative models, combining PGD-on-EqM-target (v10) with CAFM-style discriminator post-training (Lin 2026), primary on EqM-B/2 IN-256, secondary SiT head-to-head Phase 5."
metadata: 
  node_type: memory
  type: project
  originSessionId: f3229205-c8f9-42de-862b-1e264830d515
---

## Locked framing

**Sentence**: First adaptive hard-negative mining for regression-target generative models. PGD-mined hard examples on the velocity-field regression target (v10) compose with CAFM-style discriminator post-training (Lin et al. 2026) to compound FID gains on Equilibrium Matching (Wang & Du 2025). Two losses attack complementary failure modes: discriminator catches global distributional mismatch; PGD-mining catches local regression failure.

**Why this framing wins**:
- Direct external validation: VeCoR (Hong 2025) §7 explicitly lists "adaptive hard-negative mining" as open future work. v10 is exactly that.
- Second external cite: Wang+Du EqM paper itself flags "sensitivity to adversarial perturbations... unexamined in the current work, suggesting this is an area for future research."
- Mechanistically distinct from all 17 read priors (Lin AFM/CAFM, Wu DAT, Geng AEBM-Diff, Kong ACT, Briglia, Madry, Rob-GAN, etc.).
- Two contributions support workshop + ICLR.

## Targets

- **NeurIPS 2026 workshop**: deadline 2026-08-29.
- **ICLR 2027 main**: deadline ~2026-10-01.
- **MISSED**: NeurIPS 2026 main (May 6), ICML 2026 (Jan 28). ICML 2027 (Jan) is safety net.

## How to apply
- Any new variant proposal must answer: "does this fit the 'adaptive hard-negative mining on regression target' frame?"
- Paper drafts at `projects/diff-EqM/documentation/paper-draft-{intro,background,method,experiments}.md`.
- Workshop outline at `projects/diff-EqM/documentation/workshop-paper-outline.md`.
- Phase 5 SiT plan at `projects/diff-EqM/documentation/phase-5-sit-headtohead.md`.
- Differentiation memo at `projects/diff-EqM/documentation/related-work-differentiation.md`.

Related: [[diff-eqm-phase-0-v10-pass]] for current empirical state, [[diff-eqm-variant-findings]] for prior variant history.
