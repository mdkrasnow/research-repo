# Rob-GAN (Liu & Hsieh 2019)

**Citation**: Liu, X. & Hsieh, C.-J. *Rob-GAN: Generator, Discriminator, and Adversarial Attacker*. CVPR 2019. arxiv 1807.10454.

## Method
Joint G + D + PGD attacker. Attacker perturbs real images to fool D; G generates fakes. D minimizes loss across fakes + adversarial reals.

## Relevance
- **Oldest "PGD + GAN" combination prior**.
- **Defensive focus** (classifier robustness) BUT claims sample-quality + GAN-convergence gains too.
- Cite as Rob-GAN-style "combine PGD with discriminator" prior.
- **Distinct from v10+CAFM** because:
  - Rob-GAN: PGD attacker perturbs real data; discriminator separates real vs fake images.
  - v10+CAFM: PGD mines hard examples for **regression loss** on flow target; discriminator separates real vs fake velocity fields.
  - Different mechanism, different objective.

## Action
Cite as nearest "combine PGD + GAN" prior. State explicitly: "Rob-GAN combines PGD perturbations with classifier-style discrimination of real-vs-fake images; we combine PGD-on-regression-target with velocity-field discrimination."
