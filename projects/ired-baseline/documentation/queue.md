# Experiment Queue — IRED Baseline (Official Release)

## Research Question

Can we reproduce official IRED results on the matrix inversion task using the released codebase?

## Evaluation Metrics

- **Primary**: MSE (mean squared error) between predicted and true inverse matrices
- **Secondary**: Per-element accuracy, convergence speed, energy landscape quality

---

## READY

### Q-001: Baseline Validation (Pilot)
- **Hypothesis**: Official IRED implementation executes successfully with minimal configuration
- **Config**: `configs/q001_pilot.json`
- **Task**: Matrix inversion with energy-based diffusion (no hard negative mining)
- **Parameters**:
  - Rank: 10 (10×10 matrices, small for quick validation)
  - Diffusion steps: 5
  - Training steps: 1,000
  - Batch size: 256
  - Learning rate: 1e-4
- **Resources**: 1 GPU, 8GB RAM, 15 minutes, gpu_test partition
- **Priority**: HIGHEST (run first to catch issues)
- **Notes**: Quick validation before committing to longer runs

### Q-002: Baseline (Standard Configuration)
- **Hypothesis**: Establish reproducible baseline performance on matrix inversion
- **Config**: `configs/q002_baseline.json`
- **Task**: Matrix inversion with standard IRED configuration
- **Parameters**:
  - Rank: 20 (20×20 matrices)
  - Diffusion steps: 10
  - Training steps: 100,000
  - Batch size: 2048
  - Learning rate: 1e-4
- **Resources**: 1 GPU, 16GB RAM, 2 hours, gpu_test partition
- **Priority**: HIGH (must run after Q-001 succeeds)
- **Dependencies**: Q-001 must complete successfully
- **Notes**: Standard configuration from official release

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

(none)

---

## DONE

(none)

---

## FAILED

(none)

---

## Future Extensions (After Baseline Validation)

- [ ] **Q-004**: Larger matrices (rank=50) to test scalability
- [ ] **Q-005**: Different diffusion schedules (steps: 20, 50, 100)
- [ ] **Q-006**: Planning task validation (from planning_dataset.py)
- [ ] **Q-007**: Reasoning task validation (from reasoning_dataset.py)
- [ ] **Q-008**: Comparison with hard negative mining variants
- [ ] **Q-009**: Multi-GPU training validation
- [ ] **Q-010**: Energy landscape visualization and analysis
