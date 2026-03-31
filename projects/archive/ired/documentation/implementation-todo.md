# Implementation TODO — Adversarial Negative Mining for Matrix Inversion

## Ready

### Investigation Phase: Results Analysis and Diagnosis

#### Phase 0: Results Analysis (Immediate) [COMPLETED]
- [x] (T0.1) Correct MSE interpretation in completed experiments
  - **Finding**: Baseline (0.0096761) ≈ Random (0.00968396) - difference is only 0.08%
  - **Finding**: Adversarial (0.00982675) is ~1.5% worse than baseline
  - **Action**: Documented in debugging.md and research-plan.md
  - **Status**: COMPLETED - All three experiments analyzed

- [x] (T0.2) Review existing adversarial mining configuration
  - **Reviewed**: configs/q003_adversarial.json
  - **Parameters**: mining_opt_steps=2, mining_noise_scale=3.0, learning_rate=0.0001
  - **Documented**: Current hyperparameters recorded for Phase 3 sweeps
  - **Location**: projects/ired/configs/q003_adversarial.json
  - **Status**: COMPLETED

#### Phase 1: Multi-Seed Validation (Priority: HIGH) [IN PROGRESS]
- [x] (T1.1) Create multi-seed experiment configurations
  - configs/q101_multiseed_baseline.json (seeds 0-9)
  - configs/q102_multiseed_random.json (seeds 0-9)
  - configs/q103_multiseed_adversarial.json (seeds 0-9)
  - **Note**: Can use array jobs for efficient parallel execution
  - **Status**: COMPLETED - All three config files created

- [x] (T1.2) Create SLURM array job templates
  - slurm/jobs/q101_multiseed.sbatch with SLURM_ARRAY_TASK_ID as seed
  - Similar for q102, q103
  - **Benefit**: Run all 10 seeds in parallel instead of sequential
  - **Status**: COMPLETED - All three sbatch array job scripts created
  - **Details**: Uses --array=0-9%2 (throttled to 2 concurrent tasks), passes seed via --seed $SLURM_ARRAY_TASK_ID
  - **Note**: Switched from gpu_test to gpu partition to avoid QOSMaxSubmitJobPerUserLimit

- [x] (T1.0) Add seed support to experiment script
  - Added --seed parameter to experiments/matrix_inversion_mining.py
  - Implements random seed initialization (torch, numpy, random, cuda)
  - Command-line seed overrides config file seed
  - **Status**: COMPLETED

- [x] (T1.5) Submit multi-seed validation experiments
  - **Q-101**: Job ID 56216344 (array job with 10 tasks, 0-9%2 throttle)
  - **Q-102**: Job ID 56216358 (array job with 10 tasks, 0-9%2 throttle)
  - **Q-103**: Job ID 56216364 (array job with 10 tasks, 0-9%2 throttle)
  - **Status**: COMPLETED - All 3 array jobs submitted successfully
  - **Submitted**: 2026-01-21T17:48:48Z
  - **Git SHA**: 294cd74
  - **Run IDs**: q101_20260121_124610, q102_20260121_124613, q103_20260121_124900
  - **Expected Completion**: ~7.5h (5 sequential batches of 2 tasks at 1.5h each)
  - **Early Poll**: 2026-01-21T17:50:22Z (60s after submission)

- [ ] (T1.3) Create analysis script for multi-seed results
  - scripts/analyze_multiseed.py
  - Compute mean ± std for train_mse and val_mse
  - Generate confidence intervals
  - Test rank consistency (which strategy performs best across seeds)
  - **Output**: Statistical summary tables and plots

- [ ] (T1.4) Implement learning curve tracking
  - Modify experiments/matrix_inversion_mining.py to save checkpoints frequently
  - Add checkpoint_interval parameter (default: 5000 steps)
  - Save train_mse and val_mse at each checkpoint
  - **Purpose**: Identify when training diverges or plateaus

#### Phase 2: Diagnostic Instrumentation (Priority: HIGH)
- [ ] (T2.1) Add matrix conditioning analysis
  - Create analysis/matrix_conditioning.py
  - Track determinant, condition number, smallest singular value
  - Log distributions for positive samples vs adversarial negatives
  - **Key metric**: Compare cond(A) for positives vs negatives
  - **Hypothesis**: Adversarial negatives are ill-conditioned (high cond number)

- [ ] (T2.2) Add energy gap profiling
  - Modify experiments/matrix_inversion_mining.py to log:
    - E(pos) - E(neg) gap at each mining step
    - Gradient norms during adversarial mining
    - Distance between consecutive adversarial samples (diversity metric)
  - Create analysis/energy_profiling.py for visualization
  - **Output**: Energy gap plots over training, gradient norm statistics

- [ ] (T2.3) Implement false negative detection
  - Compute similarity between adversarial negatives and true positives
  - Flag negatives that are "too close" (potential false negatives)
  - Track false negative rate over training
  - **Threshold**: MSE(neg, pos) < threshold (e.g., 0.01)

#### Phase 3: Hyperparameter Sweep Infrastructure (Priority: MEDIUM)
- [ ] (T3.1) Implement mining_opt_steps sweep
  - Create configs for mining_opt_steps = [1, 3, 5, 10, 20]
  - Create SLURM array job or separate jobs
  - configs/q301_opt_steps_sweep/
  - **Current value**: Need to check q003_adversarial.json

- [ ] (T3.2) Implement ascent LR sweep
  - Add mining_ascent_lr parameter to mining_config
  - Modify GaussianDiffusion1D to use configurable ascent LR
  - Create configs for ascent_lr = [0.01, 0.05, 0.1, 0.5, 1.0]
  - **Note**: May need to modify opt_step() in diffusion code

- [ ] (T3.3) Implement noise scale sweep
  - Add mining_noise_scale parameter to mining_config
  - Apply Gaussian noise to adversarial samples: neg + noise_scale * randn()
  - Create configs for noise_scale = [0.01, 0.1, 0.5, 1.0, 2.0]
  - **Purpose**: Prevent degenerate adversarial samples

- [ ] (T3.4) Implement mixed strategy
  - Add 'mixed' mining strategy to GaussianDiffusion1D
  - Sample 70% negatives randomly, 30% adversarially
  - Create config configs/q304_mixed_strategy.json
  - **Location**: diffusion_lib/denoising_diffusion_pytorch_1d.py

#### Phase 4: Training Stability (Priority: HIGH)
- [ ] (T4.1) Add cosine LR schedule
  - Modify experiments/matrix_inversion_mining.py
  - Add --use-cosine-lr flag
  - Start decay at step 30K (or configurable)
  - **Library**: torch.optim.lr_scheduler.CosineAnnealingLR

- [ ] (T4.2) Add gradient clipping
  - Add torch.nn.utils.clip_grad_norm_() to training loop
  - Make max_norm configurable (default: 1.0)
  - Log clipped gradient norms for monitoring

- [ ] (T4.3) Add EMA (Exponential Moving Average) weights
  - Use ema_pytorch library (already in dependencies)
  - Apply EMA to model weights during training
  - Evaluate using EMA weights instead of raw weights
  - **Decay**: 0.995 (typical value)

- [ ] (T4.4) Implement early stopping
  - Track best validation MSE
  - Stop if no improvement for N steps (default: 10K)
  - Save best checkpoint separately from final checkpoint
  - Add --early-stopping flag

#### Phase 5: Visualization and Reporting (Priority: MEDIUM)
- [ ] (T5.1) Create learning curve visualization script
  - scripts/plot_learning_curves.py
  - Plot train_mse and val_mse over time for all strategies
  - Highlight divergence points
  - Compare best checkpoint vs final checkpoint

- [ ] (T5.2) Create matrix conditioning visualization
  - scripts/plot_conditioning.py
  - Histograms of det(A), cond(A), σ_min
  - Side-by-side comparison: positives vs adversarial negatives
  - **Output**: PDF figures for paper

- [ ] (T5.3) Create energy gap visualization
  - scripts/plot_energy_gaps.py
  - Time series of E(pos) - E(neg)
  - Gradient norm distributions
  - Negative sample diversity over training

- [ ] (T5.4) Create comprehensive results summary
  - scripts/summarize_results.py
  - Aggregate all experiments into single table
  - Generate LaTeX tables for paper
  - Include: mean ± std, best configs, key findings

---

### Phase 1: Core Infrastructure (Estimated: 1.5h) [COMPLETED]
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
