# Experiment Queue — IRED Baseline (Official Release)

## Research Question

Can we reproduce official IRED results on the matrix inversion task using the released codebase?

## Evaluation Metrics

- **Primary**: MSE (mean squared error) between predicted and true inverse matrices
- **Secondary**: Per-element accuracy, convergence speed, energy landscape quality

---

## READY

### Q-003: Baseline with Different Task
- **Hypothesis**: Validate IRED generalizes to alternative reasoning tasks (addition)
- **Config**: `configs/q003_addition.json`
- **Task**: Addition task with energy-based diffusion
- **Parameters**:
  - Input range: -10 to 10
  - Diffusion steps: 10
  - Training steps: 50,000
  - Batch size: 2048
  - Learning rate: 1e-4
- **Resources**: 1 GPU, 16GB RAM, 1.5 hours, gpu_test partition
- **Priority**: MEDIUM
- **Dependencies**: Q-001 must complete successfully
- **Notes**: Demonstrates IRED flexibility across task types

---

## IN_PROGRESS

### Q-002: Baseline (Standard Configuration) - RESUBMITTED
- **Job ID**: 56186268
- **Run ID**: q002_20260121_075508
- **Submitted**: 2026-01-21T12:55:37Z
- **Status**: Running
- **Git SHA**: 7770702133ed39020ae4a0424e6b600ec7a10c4b
- **Config**: `configs/q002_baseline.json`
- **Task**: Matrix inversion with standard IRED configuration
- **Parameters**:
  - Rank: 20 (20×20 matrices)
  - Diffusion steps: 10
  - Training steps: 100,000
  - Batch size: 2048
  - Learning rate: 1e-4
- **Resources**: 1 A100 GPU, 2 CPUs, 16GB RAM, partition=gpu_test, 2-hour time limit
- **Expected Runtime**: ~1.5 hours (based on scaling from Q-001 pilot: 40s for 1k steps → ~4000s for 100k steps)
- **Notes**: Resubmitted after adding matplotlib to pip install line. Early poll scheduled for 2026-01-21T12:56:37Z (60s after submission) to verify matplotlib import succeeds.

---

## DONE

### Q-001: Baseline Validation (Pilot) - SUCCESSFUL
- **Job ID**: 56162316
- **Run ID**: q001_20260121_160730
- **Submitted**: 2026-01-21T16:07:30Z
- **Started**: 2026-01-21T16:07:24Z
- **Completed**: 2026-01-21T16:12:40Z
- **Runtime**: 40 seconds
- **Node**: holygpu7c26105
- **Git SHA**: 7770702133ed39020ae4a0424e6b600ec7a10c4b
- **Config**: `configs/q001_pilot.json`
- **Task**: Matrix inversion validation run
- **Parameters**: Rank=5 (5×5 matrices), 1000 training steps, diffusion_steps=10
- **Resources**: 1 GPU, 2 CPUs, 8GB RAM, gpu_test partition
- **Results**:
  - Training completed successfully: 1000 steps
  - Final MSE: 0.0596854
  - Results persisted to: `/n/home03/mkrasnow/research-repo/projects/ired-baseline/runs/q001_pilot/results.json`
- **Validations Confirmed**:
  - ✅ Dataset API fix (commit a2edff8): Inverse class correctly uses (split, rank, ood) signature
  - ✅ Diffusion model fix (commit a2edff8): GaussianDiffusion1D correctly uses seq_length parameter
  - ✅ Result persistence fix (commit 7770702): Results successfully copied before /tmp cleanup
  - ✅ Git workflow: Automated clone and checkout working with pushed commits
  - ✅ CUDA/GPU: Successfully loaded cuda/11.8.0-fasrc01 and ran on A100
  - ✅ End-to-end pipeline: Training → results → persistence all operational
- **Notes**: MAJOR MILESTONE - First successful end-to-end run with all infrastructure validated. Ready to proceed with full baseline experiments.

---

## FAILED

### Q-002: Baseline (Standard Configuration) - First Attempt
- **Job ID**: 56185047
- **Run ID**: q002_20260121_074043
- **Submitted**: 2026-01-21T12:40:48Z
- **Started**: 2026-01-21T12:40:43Z
- **Failed**: 2026-01-21T12:42:39Z (Runtime: 1m56s)
- **Node**: holygpu7c26106
- **Git SHA**: 7770702133ed39020ae4a0424e6b600ec7a10c4b
- **Error**: ModuleNotFoundError: No module named 'matplotlib'
- **Root Cause**: dataset.py imports matplotlib.pyplot but matplotlib was not included in pip install command in sbatch script. The import appears unused (leftover from development).
- **Resolution**: Add matplotlib to dependency installation in all sbatch scripts (q001_pilot.sbatch and q002_baseline.sbatch)
- **Notes**: Environment setup completed successfully (PyTorch, CUDA, GPU verification all passed). Failure occurred during Python import phase. Early polling successfully caught initialization error within 2 minutes.

### Q-001: Baseline Validation (Pilot) - First Attempt
- **Job ID**: 56017331
- **Run ID**: q001_20260120_083826
- **Submitted**: 2026-01-20T08:38:26Z
- **Failed**: 2026-01-20T08:40:58Z (Runtime: 2m34s)
- **Git SHA**: 75df3cf948d3c2f55a88339bba0c402af27b0413
- **Error**: Python TypeError - Inverse.__init__() got unexpected keyword argument 'num_samples'
- **Root Cause**: Dataset API signature mismatch and incorrect diffusion model parameter
- **Resolution**: Fixed in commit a2edff8 (dataset API) and 7770702 (result persistence), resubmitted as job 56162042

### Q-001: Baseline Validation (Pilot) - Second Attempt
- **Job ID**: 56162042
- **Run ID**: q001_20260121_160000
- **Submitted**: 2026-01-21T16:00:00Z
- **Failed**: 2026-01-21T16:07:27Z (Runtime: 3 seconds)
- **Git SHA**: 7770702133ed39020ae4a0424e6b600ec7a10c4b
- **Error**: Git checkout failed - commit not available on remote repository
- **Root Cause**: Workflow issue - commit 7770702 wasn't pushed to origin/main before SLURM job submission. Automated job clones repo and checks out commit by SHA.
- **Resolution**: Commits a2edff8 and 7770702 now pushed to origin/main. Ready to resubmit immediately.
- **Notes**: This was a workflow/process error, not a code bug. The implementation fixes are correct.

---

## Future Extensions (After Baseline Validation)

- [ ] **Q-004**: Larger matrices (rank=50) to test scalability
- [ ] **Q-005**: Different diffusion schedules (steps: 20, 50, 100)
- [ ] **Q-006**: Planning task validation (from planning_dataset.py)
- [ ] **Q-007**: Reasoning task validation (from reasoning_dataset.py)
- [ ] **Q-008**: Comparison with hard negative mining variants
- [ ] **Q-009**: Multi-GPU training validation
- [ ] **Q-010**: Energy landscape visualization and analysis
