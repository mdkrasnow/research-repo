# DiT: Scalable Diffusion Models with Transformers

**Citation**: Peebles & Xie. *DiT*. ICCV 2023 (best paper). arxiv 2212.09748.

## Method
Replace UNet with transformer on latent patches. AdaLN-zero modulation conditions on class + timestep. Patch size {2, 4, 8}.

## Best result
DiT-XL/2 IN-256 FID **2.27** (CFG, 250 steps).

## Relevance
- Direct backbone ancestor of SiT (Ma 2024) and EqM (Wang+Du 2025).
- Lin CAFM discriminator architecture inherits DiT block structure.
- Background citation only; no method overlap with v10.

## Action
Cite in §2.1 as architecture lineage.
