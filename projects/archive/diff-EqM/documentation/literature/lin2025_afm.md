# AFM: Adversarial Flow Models (Lin et al.)

**Citation**: S. Lin, C. Yang, Z. Lin, H. Chen, H. Fan. *Adversarial Flow Models*. arxiv 2511.22475 (Nov 2025). ByteDance Seed.

## Method
- Unifies adversarial models + flow models. Generator = deterministic noise→data mapping, **same optimal-transport coupling as flow matching**. No regression target — adversarial objective only.
- Native one-step OR multi-step generation. No intermediate-timestep supervision.
- Trained with discriminator + adversarial loss; OT coupling stabilizes adv training.
- Architecture: DiT B/2, XL/2; 56-layer and 112-layer single-pass variants.

## Headline numbers
- **IN-256 XL/2, 1NFE: FID 2.38** (new best for one-step gen at this scale).
- 56-layer 1NFE: 2.08. 112-layer 1NFE: 1.94 (surpasses 28-layer 4NFE).
- B/2 1NFE approaches XL/2 consistency-model performance.

## Relevance to diff-EqM (v10 branch)

- **Same Lin lab** as CAFM (2604.11521, Apr 2026), AAPT (2506.09350, NeurIPS 2025), V-PAE (2508.21019, Aug 2025). 4 adversarial-flow papers in ~12 months. Highest scoop watch.
- Method is DISCRIMINATOR-BASED with OT-coupled generator. v10 is MINING-BASED with regression target preserved. Orthogonal niches.
- AFM occupies "one-step adversarial flow" niche. v10 occupies "multi-step gradient-descent EqM-style sampling + adversarial robustness during training" niche.
- AFM CANNOT be compared head-to-head FID with v10: different sampling regime (1NFE vs gradient descent), different objective family, different scales.

## Differentiation argument vs AFM (for paper §2)
- AFM: discriminator-based, two-player game, deterministic generator, no regression target. Collapse risk = mode collapse / dis-crush (same family as CAFM Phase 1b FID 341 failure on EqM).
- v10: single-objective regression on PGD-mined inputs. No discriminator. No two-player game. Loss bounded by EqM target so cannot collapse to trivial solutions. Compatible with any regression-target gen model.

## Implication for branch positioning
- The Lin lab now CLEARLY owns the discriminator-based adversarial-flow niche (4 papers, 2.38 FID SOTA on IN-256 XL/2). v10 should NOT claim to compete on FID-SOTA grounds.
- v10's claim is methodological: first adversarial-style training for regression-target gen models without a discriminator. Workshop paper §1 should explicitly frame this.

## Action
- Cite in workshop §2 related work as the dominant discriminator-based adversarial-flow baseline.
- Use FID 2.38 as the "what discriminator-based methods achieve at scale" point for §2.
- Do NOT scope v10 against AFM's headline numbers — wrong axis.

## Watch
- Lin lab's next paper likely Q3 2026 by their cadence. If they add "PGD-input mining" or "hard-negative regression target" — scoop. Until then v10 niche safe.
