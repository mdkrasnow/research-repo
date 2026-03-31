# Decision Summary: Project Structure for Adversarial Negative Mining Research

**Date:** 2026-01-13
**Status:** DECIDED
**Research Question:** Does adversarial negative mining improve performance on the matrix inversion task?

---

## Executive Summary

After conducting a structured 3-round debate examining three alternative project structures, we have decided to implement a **Hybrid Approach** that reuses existing training infrastructure (Trainer1D, GaussianDiffusion1D) while creating a focused experiment orchestration script for clarity and extensibility.

**Decision: Position C (Hybrid Approach)**

---

## Debate Positions Evaluated

### Position A: Minimal Adaptation (Reuse existing train.py)
- Add command-line flags to train.py for mining strategies
- Fastest initial implementation (1-2 hours)
- Risk: Clutters already complex codebase, poor extensibility

### Position B: Custom Experiment Script
- Create standalone experiment script with reimplemented training loop
- Clearest experimental narrative
- Risk: Code duplication, longer development time (4-6 hours), maintenance burden

### Position C: Hybrid Approach (SELECTED)
- Reuse Trainer1D infrastructure with custom orchestration script
- Configuration-based mining strategy selection
- Best balance of speed, clarity, and extensibility

---

## Decision Criteria and Evaluation

| Criterion | Position A | Position B | Position C | Winner |
|-----------|-----------|-----------|-----------|---------|
| **Research Clarity** | Medium | High | High | B & C (tie) |
| **Time to First Results** | 2-3h | 6-8h | 3-4h | A |
| **Reproducibility** | Medium | High | High | B & C (tie) |
| **Extensibility** | Low | Medium | High | **C** |
| **Overall Score** | 2/4 | 3/4 | **4/4** | **C** |

---

## Key Factors in Decision

1. **Extensibility is Critical**: The research question will naturally lead to ablation studies (varying opt_steps, scale_factor, mining frequency). Position C makes these trivial via configuration changes.

2. **Proven Infrastructure**: Position C reuses battle-tested Trainer1D (validation, checkpointing, metrics, EMA) while Position B risks subtle bugs in reimplementation.

3. **Acceptable Time Investment**: Position C's 3-4 hour setup is justified by long-term benefits. The user's "1-2 hours per experiment" refers to GPU runtime, not development time.

4. **Research Clarity**: Position C matches Position B's clarity through explicit configuration dictionaries and clean orchestration script.

5. **Risk Management**: Lower risk than Position B (no reimplementation), more maintainable than Position A (no flag explosion).

---

## Implementation Plan

### Phase 1: Modify GaussianDiffusion1D (1.5 hours)

**File:** `diffusion_lib/denoising_diffusion_pytorch_1d.py`

1. Add `mining_config` parameter to `__init__()`:
```python
def __init__(self, ..., mining_config=None):
    self.mining_config = mining_config or {
        'mode': 'random',
        'opt_steps': 2,
        'scale_factor': 1.0,
    }
```

2. Refactor `p_losses()` method (lines 605-699):
```python
if self.supervise_energy_landscape:
    if self.mining_config['mode'] == 'none':
        # No negative mining
        pass
    elif self.mining_config['mode'] == 'random':
        # Random perturbations (current approach)
        xmin_noise = self.q_sample(x_start=x_start, t=t, noise=3.0*noise)
        xmin_noise_rescale = self.predict_start_from_noise(...)
    elif self.mining_config['mode'] == 'adversarial':
        # Gradient-based hard negatives
        xmin_noise = self.q_sample(x_start=x_start, t=t, noise=3.0*noise)
        xmin_noise = self.opt_step(
            inp, xmin_noise, t, mask, data_cond,
            step=self.mining_config['opt_steps'],
            sf=self.mining_config['scale_factor']
        )
        xmin_noise_rescale = self.predict_start_from_noise(...)
```

### Phase 2: Create Experiment Script (1.5 hours)

**File:** `experiments/matrix_inversion_mining_comparison.py` (NEW)

Key components:
1. `get_mining_config(strategy)`: Returns config dict for baseline/random/adversarial
2. `run_experiment(strategy, rank, batch_size, train_steps)`: Trains and evaluates one strategy
3. `main()`: Command-line interface with argparse
4. Results comparison and visualization

### Phase 3: Pilot Experiments (30 minutes)

Run quick validation:
```bash
python experiments/matrix_inversion_mining_comparison.py --pilot
```

### Phase 4: Full Experiments (3-6 hours GPU time)

Run three strategies with 100K steps each:
```bash
python experiments/matrix_inversion_mining_comparison.py \
    --strategies baseline random adversarial --train-steps 100000
```

---

## Expected Outcomes

### Success Scenario: Adversarial Mining Wins
- Validation MSE significantly lower (>5% improvement over baseline)
- Proceed with ablations (vary opt_steps, scale_factor)
- Extend to other reasoning tasks
- Write paper

### Alternative Scenarios:
- **Random Mining Best**: Gradient-based negatives may overfit
- **Baseline Best**: Negative mining may not help matrix inversion
- **Training Instability**: May need hyperparameter tuning

All scenarios provide valuable research insights.

---

## Why This Decision Beats Alternatives

### vs Position A (Minimal Adaptation)
- **Better extensibility**: Configuration-based vs flag explosion
- **Clearer experimental narrative**: Orchestration script shows design explicitly
- **Similar development time**: 3-4h vs 2-3h (marginal difference)

### vs Position B (Custom Script)
- **Faster development**: 3-4h vs 6-8h (reuse Trainer1D infrastructure)
- **Lower risk**: No reimplementation bugs
- **Better maintenance**: Core infrastructure improvements propagate automatically
- **Similar clarity**: Configuration dicts + orchestration = explicit design

---

## Key Design Principles Applied

1. **Separation of Concerns**: Infrastructure (Trainer1D) vs experimental design (orchestration script)
2. **Configuration over Code**: Mining strategies defined declaratively, not procedurally
3. **Leverage Proven Code**: Reuse battle-tested components
4. **Optimize for Change**: Easy to add ablations without code changes
5. **Explicit over Implicit**: Configuration dictionaries make experimental design obvious

---

## Next Actions

1. Implement mining_config in GaussianDiffusion1D
2. Create experiment orchestration script
3. Run pilot experiments (10K steps)
4. Execute full experiments (100K steps)
5. Analyze results and decide on ablations

**Current Phase:** IMPLEMENT
**Estimated Time to First Results:** 4-4.5 hours (3-4h dev + 30min pilot)

---

## Documentation References

- Full debate transcript: `documentation/debate.md`
- Experimental design: `documentation/experimental-design.md`
- Implementation checklist: `documentation/implementation-todo.md`
- Experiment queue: `documentation/queue.md`
