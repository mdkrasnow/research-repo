# Implementation TODO — Adversarial Negative Mining for Matrix Inversion

## Ready

### Phase 1: Core Infrastructure (Estimated: 1.5h)
- [x] (T1) Modify GaussianDiffusion1D to add `mining_config` parameter
  - Location: `diffusion_lib/denoising_diffusion_pytorch_1d.py` lines 605-699
  - Add support for: 'none' (baseline), 'random' (random negatives), 'adversarial' (gradient-based hard negatives)
  - Refactor existing noise contrastive code to be strategy-aware
  - **COMPLETED**: Added mining_config parameter and three mining strategies

- [ ] (T2) Update Trainer1D to pass mining_config from initialization
  - Ensure mining strategy is configurable at trainer level
  - Add validation/logging for mining strategy
  - **NOTE**: This may not be needed - mining_config is already passed directly to GaussianDiffusion1D in the experiment script

### Phase 2: Experiment Script (Estimated: 1.5h)
- [x] (T3) Create `experiments/matrix_inversion_mining.py`
  - Implement configuration-based experiment orchestration
  - Support baseline, random, and adversarial mining strategies
  - Reuse existing Trainer1D and Inverse dataset
  - Add argparse interface with --mining-strategy flag
  - **COMPLETED**: Script created during /make-project

- [x] (T4) Create configuration files for each strategy
  - `configs/q001_baseline.json` (no negative mining)
  - `configs/q002_random.json` (random negatives)
  - `configs/q003_adversarial.json` (gradient-based hard negatives)
  - **COMPLETED**: All 4 configs created (including q004_pilot.json)

### Phase 3: SLURM Integration (Estimated: 30min)
- [x] (T5) Create SLURM sbatch templates
  - `slurm/jobs/q001.sbatch` (baseline)
  - `slurm/jobs/q002.sbatch` (random mining)
  - `slurm/jobs/q003.sbatch` (adversarial mining)
  - Configure for 1-2 hour runtime, GPU allocation
  - **COMPLETED**: All 4 sbatch files created (including q004_pilot.sbatch)

### Phase 4: Local Testing (Estimated: 30min)
- [x] (T6) Test local execution with small config
  - Run baseline experiment locally with reduced steps (100 steps)
  - Verify training loop, loss computation, checkpointing
  - Check output format and result logging
  - **COMPLETED**: Validated imports, config loading, trainer init. MPS float64 limitation prevents full local training (not a blocker for SLURM)

### Phase 5: Pilot Runs (Estimated: 30min GPU time)
- [ ] (T7) Submit pilot runs to SLURM
  - Quick validation experiments (10K steps)
  - Verify all three mining strategies execute correctly
  - Check SLURM logs for errors

## Blocked
(none)

## Completed
- [x] (T1) Modified GaussianDiffusion1D with mining_config parameter and three strategies
- [x] (T3) Created experiment script `experiments/matrix_inversion_mining.py`
- [x] (T4) Created all configuration files (q001-q004)
- [x] (T5) Created all SLURM sbatch templates (q001-q004)

---

## Notes
- **Hybrid approach chosen**: Reuse Trainer1D/GaussianDiffusion1D infrastructure, add custom orchestration
- **Mining strategies**:
  - Baseline: Standard diffusion training without adversarial negatives
  - Random: Randomly sampled matrices as negatives
  - Adversarial: Gradient ascent on energy landscape using existing `opt_step()`
- **Matrix rank**: 20x20 matrices (configurable via --rank flag)
- **Evaluation metric**: MSE (mean squared error) on matrix inversion accuracy
