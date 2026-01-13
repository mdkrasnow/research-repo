# Adversarial Negative Mining for Matrix Inversion - Project Documentation

**Research Question:** Does adversarial negative mining improve performance on the matrix inversion task?

---

## Quick Start

This project investigates whether gradient-based adversarial negative mining improves IRED's performance on matrix inversion compared to random perturbation mining or no mining.

**Current Status:** Implementation phase - ready to begin coding

---

## Documentation Index

### Core Documents

1. **[decision-summary.md](decision-summary.md)** - **START HERE**
   - Executive summary of the structured debate
   - Final decision: Hybrid Approach (Position C)
   - Implementation plan overview

2. **[debate.md](debate.md)** - Full 3-round debate transcript
   - Position A: Minimal Adaptation (reuse train.py)
   - Position B: Custom Experiment Script
   - Position C: Hybrid Approach (SELECTED)
   - Detailed argumentation and rebuttals

3. **[experimental-design.md](experimental-design.md)** - Complete experimental protocol
   - Research hypothesis
   - Three experimental conditions (baseline, random, adversarial)
   - Dataset and model architecture specifications
   - Evaluation metrics and success criteria
   - Detailed implementation approach

4. **[implementation-todo.md](implementation-todo.md)** - Development checklist
   - Phase 1: Modify GaussianDiffusion1D (1.5h)
   - Phase 2: Create experiment script (1.5h)
   - Phase 3: Pilot experiments (30min)
   - Phase 4: Full experiments (3-6h GPU time)

5. **[queue.md](queue.md)** - Experiment queue and log
   - Planned experiments
   - Run configurations
   - Experiment status tracking

### Research Question

**Primary:** Does adversarial negative mining improve performance on matrix inversion?

**Context:** The existing IRED codebase includes:
- Matrix inversion dataset (Inverse class)
- Energy-based diffusion models
- Gradient optimization infrastructure (opt_step)
- Random perturbation mining (current approach)

**Goal:** Compare three training strategies:
1. Baseline (no mining)
2. Random mining (current IRED approach)
3. Adversarial mining (gradient-based hard negatives)

---

## Decision Summary

After structured debate, we selected the **Hybrid Approach**:

### Why Hybrid Won

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Research Clarity | ⭐⭐⭐⭐⭐ | Configuration dicts make experimental design explicit |
| Time to Results | ⭐⭐⭐⭐ | 3-4h setup is acceptable for long-term benefits |
| Reproducibility | ⭐⭐⭐⭐⭐ | Config-based approach + version control = clear record |
| Extensibility | ⭐⭐⭐⭐⭐ | Easy ablations via config changes, no code modification |
| **Overall** | **4.5/5** | **Best balance of speed, clarity, and flexibility** |

### Key Architecture

```
Reuse Proven Infrastructure          +    Custom Experiment Orchestration
(Trainer1D, GaussianDiffusion1D)          (Clean experimental narrative)
         ↓                                          ↓
    Fast development                         Clear research design
    Low bug risk                             Easy ablations
    Automatic updates                        Self-documenting
```

### Implementation Approach

1. **Add `mining_config` parameter to GaussianDiffusion1D**
   - Supports three modes: 'none', 'random', 'adversarial'
   - Configurable opt_steps and scale_factor

2. **Create `experiments/matrix_inversion_mining_comparison.py`**
   - Clean orchestration script
   - Configuration-based strategy selection
   - Automatic results comparison

3. **Run experiments**
   - Pilot: 10K steps (30min) for validation
   - Full: 100K steps per strategy (1-2h each)

---

## Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| **Implementation** | 3-4 hours | Modify diffusion code + create experiment script |
| **Pilot** | 30 minutes | Quick validation (10K steps) |
| **Full Experiments** | 3-6 hours | Three strategies × 100K steps |
| **Analysis** | 1-2 hours | Compare results, plot curves |
| **Ablations** | Variable | If adversarial mining shows promise |

**Total to First Full Results:** ~8-12 hours (dev + experiments + analysis)

---

## Expected Outcomes

### Scenario 1: Adversarial Mining Succeeds ✅
- MSE improvement >5% over baseline
- **Next:** Run ablations (opt_steps, scale_factor, matrix size)
- **Next:** Extend to other reasoning tasks
- **Next:** Write paper

### Scenario 2: Random Mining Best
- Random perturbations sufficient
- **Next:** Investigate why gradient-based negatives underperform

### Scenario 3: Baseline Best
- Negative mining provides no benefit
- **Next:** Try alternative regularization approaches

### Scenario 4: Training Instability
- Adversarial mining causes divergence
- **Next:** Tune hyperparameters (opt_steps, scale_factor, loss weighting)

---

## File Structure

```
projects/ired/
├── research_question.md              # Research question definition
├── documentation/
│   ├── README.md                     # This file
│   ├── decision-summary.md           # Executive summary
│   ├── debate.md                     # Full debate transcript
│   ├── experimental-design.md        # Detailed experimental protocol
│   ├── implementation-todo.md        # Development checklist
│   └── queue.md                      # Experiment queue
├── .state/
│   └── pipeline.json                 # Project state (phase: IMPLEMENT)
├── diffusion_lib/
│   └── denoising_diffusion_pytorch_1d.py  # TO MODIFY: add mining_config
├── experiments/
│   └── matrix_inversion_mining_comparison.py  # TO CREATE
├── train.py                          # Existing training infrastructure
├── dataset.py                        # Inverse dataset (no changes needed)
└── models.py                         # EBM models (no changes needed)
```

---

## Key Insights from Debate

### Why NOT Position A (Minimal Adaptation)?
- Flag explosion for future ablations
- Clutters already complex train.py (318 lines, 15+ datasets)
- Poor extensibility

### Why NOT Position B (Custom Script)?
- 6-8 hours development time (2× Position C)
- Code duplication (training loop, validation, checkpointing)
- Maintenance burden (changes don't propagate)

### Why Position C (Hybrid)?
- **Best extensibility:** Configuration-based ablations
- **Proven infrastructure:** Reuses battle-tested Trainer1D
- **Research clarity:** Orchestration script shows experimental design
- **Acceptable development time:** 3-4h (only 1h more than Position A)
- **Low risk:** No reimplementation, no breaking existing code

---

## Next Actions

1. **Begin Implementation**
   - Modify `diffusion_lib/denoising_diffusion_pytorch_1d.py`
   - Create `experiments/matrix_inversion_mining_comparison.py`

2. **Run Pilot**
   - Quick validation with 10K steps
   - Verify all strategies work

3. **Execute Full Experiments**
   - Baseline, random, adversarial (100K steps each)

4. **Analyze and Decide**
   - Compare MSE and reconstruction error
   - Determine if ablations are warranted

---

## Resources

- **Compute:** 1-4 GPUs
- **Time per experiment:** 1-2 hours (100K steps)
- **Storage:** ~1.5GB total (3 strategies × ~500MB)
- **Development time:** 3-4 hours
- **SLURM Partition:** `gpu_test` (all experiments < 24h; better queue priority)

---

## References

- **IRED Paper:** "Learning Iterative Reasoning through Energy Diffusion" (ICML 2024)
- **Codebase:** Projects/ired/ (cloned from GitHub)
- **Existing Infrastructure:**
  - Trainer1D: `diffusion_lib/denoising_diffusion_pytorch_1d.py`
  - Inverse Dataset: `dataset.py` lines 420-465
  - Gradient optimization: `opt_step` method lines 373-405
