# Research Question

**Does adversarial negative mining improve performance on the matrix inversion task?**

## Background

The IRED (Iterative Reasoning Energy Diffusion) codebase implements diffusion-based approaches to continuous-space reasoning tasks, including matrix inversion. The existing implementation includes:

- Matrix inversion dataset (Inverse class in dataset.py)
- Diffusion-based training infrastructure (GaussianDiffusion1D, Trainer1D)
- Energy-based models with gradient-based optimization (opt_step method)
- Existing negative mining infrastructure (noise contrastive estimation, lines 605-699 in denoising_diffusion_pytorch_1d.py)

## Research Goal

Investigate whether gradient-based adversarial negative mining improves model performance on matrix inversion compared to:
1. Baseline: No negative mining (standard diffusion training)
2. Random negative mining: Current implementation with random perturbations
3. Adversarial negative mining: Gradient-based hard negative generation using opt_step

## Constraints

- GPU access: 1-4 GPUs
- Time per experiment: 1-2 hours
- Focus: Quick, iterative experiments with clear experimental comparison
