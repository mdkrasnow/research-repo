# Results Summary — IRED Baseline (Official Release)

## Research Question

Can we reproduce official IRED results on the matrix inversion task using the released codebase from Yilun Du?

## Project Overview

This project validates the official IRED (Iterative Reasoning through Energy Diffusion) implementation as the baseline for understanding energy-based reasoning with diffusion models.

## Completed Experiments

(Results will be populated as experiments complete)

### Q-001: Baseline Validation (Pilot)
- **Status**: Pending
- **Config**: 10×10 matrices, 1000 steps, gpu_test partition
- **Expected Duration**: 15 minutes
- **Key Metrics**: MSE (mean squared error), energy convergence

### Q-002: Baseline (Standard Configuration)
- **Status**: Pending
- **Config**: 20×20 matrices, 100K steps, gpu_test partition
- **Expected Duration**: 2 hours
- **Key Metrics**: MSE, per-element accuracy, convergence speed

### Q-003: Addition Task
- **Status**: Pending
- **Config**: Addition reasoning task, 50K steps
- **Expected Duration**: 1.5 hours
- **Key Metrics**: Task accuracy, generalization

## Key Findings

(TBD - to be updated as results come in)

## Reproducibility Notes

- Base commit: (to be filled in when Q-001 runs)
- Seed: 42 (fixed for reproducibility)
- Device: GPU (CUDA 11.8.0-fasrc01 on cluster)
- Python: 3.10.13-fasrc01

## Performance Baselines

(To be populated from Q-001 and Q-002 results)

## Next Steps

1. Submit Q-001 pilot to cluster
2. If Q-001 succeeds: validate output format and metrics
3. Submit Q-002 for production baseline
4. Compare against custom IRED variants in other projects
5. Document any discrepancies from expected results

## References

- Official Code: https://github.com/yilundu/ired_code_release
- Paper: https://arxiv.org/abs/2406.11179
- IRED Website: https://energy-based-model.github.io/ired/
