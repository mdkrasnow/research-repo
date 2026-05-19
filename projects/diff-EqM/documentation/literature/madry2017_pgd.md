# Towards Deep Learning Resistant to Adversarial Attacks (PGD AT)

**Citation**: Madry, Makelov, Schmidt, Tsipras, Vladu. *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR 2018. arxiv 1706.06083.
**Read depth**: skim.

## Setting
- Task: classifier robustness to adversarial examples.
- Datasets: MNIST, CIFAR-10.
- Foundational PGD adversarial training reference.

## Method
Saddle-point formulation: `min_θ E[max_{||δ||≤ε} L(θ; x+δ, y)]`. PGD as multi-step inner maximization.

## Relevance to v10
- Foundational citation. v10 = "PGD applied to the regression target in generative modeling instead of cross-entropy in classifiers."
- Key principle: multi-step PGD finds stronger adversarial examples than single-step FGSM.
- **Caveat**: Briglia (2505.21742) argues iterative PGD overhead unnecessary for diffusion AT; single-step FGSM works. v10 design defaults K=3, may downscale to K=1 if compute-bound.

## Action
Cite as foundational PGD reference. State explicitly: "Madry-style PGD applied to the L2 regression target."
