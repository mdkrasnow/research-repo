# Decision Summary — IRED Baseline (Official Release)

## Project Creation Decision

### Problem Statement

Need to establish a reproducible baseline for IRED (Iterative Reasoning through Energy Diffusion) using the official codebase from Yilun Du, separate from the custom IRED implementation.

### Options Considered

**Option A: Reuse local IRED project**
- Pros: Code already available, familiar patterns
- Cons: Mixed with custom modifications, unclear which changes are original vs. custom

**Option B: Create new project from Yilun Du's official release ✓ CHOSEN**
- Pros: Clean separation, reproducible with official code, independent validation
- Cons: Requires new project structure

### Rationale

Official release provides:
1. **Reproducibility**: Exact code from published paper
2. **Comparison**: Clean baseline to compare custom variants against
3. **Independence**: Separate slug (ired-baseline) avoids interference with IRED project
4. **Documentation**: Reference implementation for understanding energy-based reasoning

### Design Decisions

#### 1. Project Structure
- Slug: `ired-baseline` (distinct from `ired`)
- Located at: `projects/ired-baseline/`
- Includes full diffusion_lib, models, and dataset implementations from official release

#### 2. Experiment Design
- **Q-001 (Pilot)**: 10×10 matrices, 1000 steps, 15 min runtime
  - Purpose: Catch execution issues early, validate setup
  - Quick turnaround for rapid iteration

- **Q-002 (Baseline)**: 20×20 matrices, 100K steps, 2 hour runtime
  - Purpose: Establish performance metrics for comparison
  - Standard configuration matching paper

- **Q-003 (Alternative Task)**: Addition task validation
  - Purpose: Demonstrate generalization beyond matrix inversion

#### 3. SLURM Integration
- Uses same automated git cloning pattern as IRED project
- Each job clones fresh from GitHub with specific commit SHA
- Ensures reproducibility and handles concurrent jobs safely

#### 4. Configuration Strategy
- JSON-based configs with all hyperparameters
- Base templates (q001_pilot, q002_baseline)
- Easy to create variations without code changes

### Approved By

Self-approved based on research needs for independent baseline validation.

### Timeline

- T1: Local validation (imports, dependencies)
- T2-T3: Configuration and experiment script
- T4-T5: SLURM integration and testing
- T6-T8: Pilot submission and validation

---

## Future Opportunities

After baseline validation, potential comparison projects:

1. **IRED-v2**: Custom improvements (hard negative mining, gradient clipping)
2. **IRED-Ablation**: Systematic ablation studies
3. **IRED-Scaling**: Investigate scalability (larger matrices, more training)
4. **IRED-Tasks**: Extend to planning, reasoning, SAT solving

## References

- Official Repository: https://github.com/yilundu/ired_code_release
- Paper: https://arxiv.org/abs/2406.11179
- Project Website: https://energy-based-model.github.io/ired/
