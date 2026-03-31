# Debugging — DG-ANM for EqM

## Active Issues
(none yet)

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
