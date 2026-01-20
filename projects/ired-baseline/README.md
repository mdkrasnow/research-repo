# IRED Baseline — Official Yilun Du Release

## Overview

This project uses the official [IRED (Iterative Reasoning through Energy Diffusion)](https://github.com/yilundu/ired_code_release) codebase from Yilun Du to establish baseline results and validate the framework.

**Paper**: [Learning Iterative Reasoning through Energy Diffusion](https://arxiv.org/abs/2406.11179)

## Key Contribution

IRED formulates reasoning and decision-making problems using energy-based optimization and diffusion models. The framework learns energy functions to represent constraints between input conditions and desired outputs.

## Project Goals

1. **Baseline Establishment**: Run official IRED implementation on matrix inversion task
2. **Validation**: Confirm code execution, reproducibility, and output format
3. **Comparison Baseline**: Establish performance metrics to compare against optimized variants
4. **Documentation**: Create reproducible SLURM workflow for official IRED

## Core Components

- **diffusion_lib/**: Core diffusion infrastructure (denoising, sampling, training loops)
- **dataset.py**: Dataset implementations (Inverse, Addition, LowRank, etc.)
- **models.py**: Energy-based models (EBM, DiffusionWrapper variants)
- **train.py**: High-level training orchestration
- **experiments/**: Experiment runner scripts

## Quick Start

### Local Validation
```bash
python -c "import torch; from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D; print('✓ Imports successful')"
```

### Configuration
See `configs/` for experiment configurations.

### Execution
See `documentation/queue.md` for experiment queue and resource allocations.

## Project Status

- **Phase**: TEST (local validation)
- **Implementation**: ✓ Complete (official release)
- **Configuration**: ⏳ In progress
- **SLURM Integration**: ⏳ In progress

## Next Steps

1. Complete local validation tests
2. Create experiment configs
3. Submit pilot run to SLURM cluster
4. Validate results and reproducibility

## References

- Official IRED Repository: https://github.com/yilundu/ired_code_release
- Paper: https://arxiv.org/abs/2406.11179
- Project Website: https://energy-based-model.github.io/ired/
