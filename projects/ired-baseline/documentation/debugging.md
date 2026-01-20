# Debugging — IRED Baseline (Official Release)

## Active Issues

(none yet)

## Resolved

(none yet)

---

## Common Failure Modes (Preemptive Checklist)

- [ ] **Import errors**: Missing pytorch, numpy, or diffusion_lib dependencies
  - Fix: Ensure Python 3.10.13-fasrc01 and required packages installed on cluster

- [ ] **Config format mismatches**: JSON parsing errors or missing required fields
  - Fix: Validate config schema before training

- [ ] **Path errors**: Relative paths not working in SLURM jobs
  - Fix: Use absolute paths or work from repo root

- [ ] **Resource limits**: Out of memory (OOM) or timeout
  - Fix: Adjust batch size, matrix rank, or time limit in sbatch

- [ ] **SLURM module loading**: Missing CUDA or python modules
  - Fix: Update module names to match cluster (e.g., python/3.10.13-fasrc01, cuda/11.8.0-fasrc01)

- [ ] **Data generation**: Random seed not set - non-reproducible results
  - Fix: Ensure seed parameter in dataset initialization

- [ ] **Energy computation**: NaN or Inf values during training
  - Fix: Check energy scale, gradient clipping, and model architecture

---

## Investigation Steps

1. Check SLURM log files: `slurm/logs/<job_name>_<job_id>.{out,err}`
2. Review pipeline.json event log for context
3. Check dataset generation (Inverse class in dataset.py)
4. Validate model energy computation (models.py EBM)
5. Confirm diffusion loop parameters (denoising_diffusion_pytorch_1d.py)
