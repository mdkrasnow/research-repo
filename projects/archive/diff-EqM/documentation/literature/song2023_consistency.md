# Consistency Models (Song et al. ICML 2023)

**Citation**: Song, Dhariwal, Chen, Sutskever. *Consistency Models*. ICML 2023. arxiv 2303.01469.

## Method
One-step generation via regression on consistency property. Two training modes: distillation (CD) and from-scratch (CT). Best FIDs: 3.55 CIFAR-10, 6.20 IN-64.

## Relevance
- Regression-target one-step generation. Adjacent family to EqM (both regression-based).
- ACT (Kong 2024) adds discriminator AT to this family — separate from us.
- Branch B applies to flow matching / EqM, not consistency. If we wanted to extend further: v10+CAFM-style transfer to consistency models would be a third contribution. Out of scope for ICLR.

## Action
Cite as adjacent family. Mention in discussion as natural future-work direction.
