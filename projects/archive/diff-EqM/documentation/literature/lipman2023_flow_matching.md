# Flow Matching for Generative Modeling

**Citation**: Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling*. ICLR 2023. arxiv 2210.02747.

## Method
Simulation-free training of continuous normalizing flows: regress on velocity field `u_t(x)` of a fixed conditional probability path. Generalizes diffusion via interpolant framework.

## Relevance
- Foundational regression-target generative model.
- Direct ancestor of SiT (Ma 2024) and CAFM (Lin 2026).
- v10's "regression-target adversarial training" applies here too.

## Action
Cite as foundational. Phase 5 SiT head-to-head uses this model family.
