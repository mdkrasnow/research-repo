# Debugging — DG-ANM for EqM

## Active Issues

### CIFAR-10 vanilla FID 497 vs paper's 3.36 (ROOT CAUSE IDENTIFIED, 2026-04-18)
- **Symptom**: vanilla EqM CIFAR 80ep → FID 497.55 (job 3053350); DG-ANM → FID 497.04 (job 3104472). Both stuck at pathological floor, no method separation.
- **Root cause**: wrong architecture. Paper Appendix B.1 uses U-Net on CIFAR from the Flow Matching (Lipman et al., 2024) codebase. We used EqM-S/2 transformer (patch=4) because our vendored `eqm-upstream/` only contains the SiT-style transformers from the ImageNet branch — no U-Net.
- **Full audit + fix plan**: `documentation/stage-a5-audit.md`
- **Action**: do not run any more CIFAR experiments on the current stack. Decide fix path (port FM UNet / DDPM UNet smoke / defer CIFAR) before more compute.
- **Good news**: Stage B (ImageNet-256) is NOT affected — our upstream's transformer models are what the paper used on ImageNet. Stage B can proceed in parallel with the CIFAR fix.

## Resolved
(none yet)

---
## Common Failure Modes (Preemptive Checklist)
- [ ] Import errors (timm, diffusers not installed on cluster)
- [ ] OOM with geometry estimation (P_T/P_N are BxDxD — may need batching for large D)
- [ ] Mining gradient explosion (adversarial search can produce NaN)
- [ ] EqM field returns NaN at large perturbations
- [ ] Checkpoint path mismatch between train and eval configs
- [ ] CIFAR-10 download fails on cluster (use pre-cached data_dir)
