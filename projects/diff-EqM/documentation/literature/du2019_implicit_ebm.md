# Implicit Generation and Modeling with EBMs (Du & Mordatch 2019)

**Citation**: Du, Y. & Mordatch, I. *Implicit Generation and Modeling with Energy-Based Models*. NeurIPS 2019. arxiv 1903.08689.

## Setting
- Task: scale classical EBM training (MCMC-based) to ImageNet.
- Datasets: CIFAR-10, IN-32, IN-128, robotic trajectories.

## Method
Contrastive divergence with SGLD-based MCMC negatives. Tricks to stabilize MCMC sampling. Energy function E(x). Sampling via MCMC on energy.

## Relevance
- Yilun Du is co-author. Du is also senior author on EqM 2025.
- This paper IS the lineage Du → EqM.
- **No PGD-based mining**; SGLD only.
- **No regression-target formulation**.
- Compositional generation + OOD robustness claims preserved across Du's later EqM work.

## Action
- Cite as historical EBM training reference.
- **Watch Du's group for adversarial-EBM follow-ups** — they have the mechanism + the model + the lab to publish v10-style work themselves.
