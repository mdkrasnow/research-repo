# Debugging — DG-ANM for EqM

## Active Issues

### R2 v03/v06 OOM on gpu_test 20G card (RESOLVED 2026-04-27)
- **Symptom**: jobs 8898137 (v03) and 8898147 (v06) FAILED at 2-3 min wall-time on `gpu_test` partition. `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of 19.62 GiB of which 32.88 MiB is free`.
- **Root cause**: mining variants need 3-4x the activation memory of vanilla (PGA does multiple forward+backward per step). bs=128 fits in 40G A100 but not in the 20G test card. v00_vanilla ran fine on `gpu` (40G) at the same bs because it has no mining.
- **Fix**: resubmit mining variants only to `gpu` partition. v02 (8898150) and v04 (8898152) already there and unaffected.
- **Resubmits**: v03→8941251, v06→8941252 on `gpu`.
- **Future rule**: never submit mining variants (v01-v04, v06) to `gpu_test`. Add to variant_pilot.sbatch header comment, or split sbatches per-partition. Non-mining variants (v00, v05) can still use gpu_test for queue speed.



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
