"""CAFM-to-EqM port: discriminator-based adversarial post-training of EqM.

Adapts Lin et al. 2026 CAFM (arxiv 2604.11521) from SiT/JiT/Z-Image to EqM.
Key adaptations:
- EqM is time-unconditional (input t=0 fixed); discriminator becomes γ-conditional
  via the existing time embedding interface, with γ ∈ [0,1] playing the role of t.
- EqM target = (ε − x) · c(γ); CAFM's velocity_real = ε − x (no c(γ) scaling).
  We multiply by c(γ) so the discriminator compares the energy-compatible target.
- Latent space SD-VAE 4×32×32 — identical to CAFM's SiT setup, no VAE pipeline change.

Modules:
- `eqm_target`: c(γ) truncated decay + target computation matching transport.py.
- `generator_wrapper`: wraps our EqM-B/2 model with t=0 frozen.
- `v10_mining`: PGD hard-example mining on the EqM regression loss.

Phase 1a deliverable. See `documentation/cafm-eqm-port-design.md`.
"""
