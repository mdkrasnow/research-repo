# IRED-DCD Investigation Plan

## Overview

This investigation implements a corrected version of IRED that treats gradient-optimized negatives as **model samples** (short-run MCMC) rather than discriminative noise contrastive negatives. This aligns the training objective with the actual sampling process and addresses the fundamental mismatch that caused adversarial mining to underperform baseline.

## Core Problem Identified

**Current Implementation Issue**:
- Negatives are produced by **minimizing** the model's energy (short-run samples)
- But training objective treats them like "noise negatives" in a **discriminative NCE classifier**
- This creates a mismatch: the model is trained to discriminate against its own samples

**Solution**: Use Contrastive Divergence (CD) / Diffusion Contrastive Divergence (DCD) objective that is consistent with EBM training using short-run sampling.

## Key Changes

### 1. Negative Generation: Langevin-style Sampling + Replay Buffer

**Current**: Deterministic gradient descent with uphill reject
**New**: Langevin dynamics with Gaussian noise per step

```python
# Langevin sampling step:
x ← x - η∇_x E(x,t) + σ·N(0,I)
```

**Parameters**:
- `k_steps`: 10 (increased from 2)
- `noise_scale_init`: 1.0-1.5 (reduced from 3.0)
- `langevin_sigma`: √(2η) as starting heuristic

**Replay Buffer**:
- FIFO buffer per timestep bucket (or global)
- Size: 10,000 samples
- `p_replay`: 0.95 (95% from buffer, 5% fresh init)
- Improves mixing and stabilizes training

### 2. Energy Loss: CD/DCD-style Objective

**Current**: Binary cross-entropy (discriminative NCE)
```python
L_NCE = CrossEntropy(-energy_stack, target=0)
```

**New**: Contrastive Divergence energy difference
```python
L_CD = E[E(x_pos, t)] - E[E(x_neg, t)]
```

- Stop-gradient through sampling trajectory (practical choice)
- Matches "data energy down, model-sample energy up" logic of ML-trained EBMs
- Optionally use DCD with diffusion transitions for better theoretical grounding

### 3. False-Negative Filtering: Task-Aware Residual Check

**Problem**: Hard negatives can be false negatives (too close to correct solutions)

**Solution**: Use matrix inversion residual to detect near-correct negatives
```python
r(X) = ||AX - I||_F
if r(x_neg) < τ(t): drop from energy loss (or downweight)
```

**Parameters**:
- `τ(t)`: Percentile-based threshold (e.g., 30th percentile of residuals)
- Stricter at small t where closeness matters more

### 4. Gentle Energy Loss: Scheduling + Small Weight

**Current**: Fixed weight (0.5), applied to all timesteps

**New**:
- Warmup: 0 → 0.05-0.1 over first 20% of training
- Max weight: 0.05-0.1 (not 0.5)
- Optional: Only apply for mid timesteps t ∈ [0.2T, 0.8T]
- Prevents energy term from fighting denoising early

## Ablation Plan

We run 4 configurations to isolate the contribution of each component:

| Config | CD Loss | Replay | Residual Filter | Schedule | Expected Improvement |
|--------|---------|--------|----------------|----------|---------------------|
| Q-201 (Baseline) | ✗ | ✗ | ✗ | ✗ | Reference (current NCE) |
| Q-202 (+CD) | ✓ | ✗ | ✗ | ✗ | Fix objective mismatch |
| Q-203 (+Replay) | ✓ | ✓ | ✗ | ✗ | Better mixing |
| Q-204 (Full) | ✓ | ✓ | ✓ | ✓ | All features (best shot) |

**All experiments**:
- 10 seeds each (40 total experiments)
- Same hyperparameters: rank=20, batch=2048, 100K steps
- Reduced noise_scale: 1.5 (from 3.0)
- Increased opt_steps: 10 (from 2)

## Why This Should Beat Baseline

**Baseline MSE** (from original investigation): 0.0096761

**Mechanisms for Improvement**:

1. **CD/DCD objective** aligns training with sampling process
   - No more fighting against model's own samples
   - Proper EBM gradient flow

2. **Replay buffer** improves sample quality
   - Persistent chains mix better
   - More informative negatives (not just noise)

3. **Residual filtering** removes false negatives
   - Prevents "punishing correct behavior"
   - Cleaner gradient signal

4. **Energy scheduling** improves stability
   - Denoising learns first, then energy shaping
   - Prevents early-stage interference

**Target**: Beat baseline by 1-5% (MSE < 0.0096)

## Implementation Checklist

### Code Changes Required

- [ ] **Langevin sampling function** (`sample_negatives` in diffusion code)
  - Add Langevin noise: `σ·N(0,I)`
  - Configurable k_steps, sigma multiplier

- [ ] **Replay buffer class**
  - FIFO buffer with per-timestep or global storage
  - Sample with probability p_replay
  - Initialize fresh samples with reduced noise_scale

- [ ] **CD/DCD loss function**
  - Replace CrossEntropy with energy difference
  - Stop-gradient on x_neg
  - Optional: DCD with diffusion transitions

- [ ] **Residual filter**
  - Compute `||AX - I||_F` for negatives
  - Mask out samples below threshold
  - Apply mask to energy loss

- [ ] **Energy loss scheduler**
  - Warmup from 0 to max_weight
  - Optional timestep range filtering
  - Integration with training loop

### Config Files

- [x] `q201_dcd_baseline.json` - Current NCE (reference)
- [x] `q202_dcd_cdloss.json` - CD loss + Langevin
- [x] `q203_dcd_replay.json` - + Replay buffer
- [x] `q204_dcd_full.json` - + Residual filter + Schedule

### SLURM Scripts

- [ ] `q201_dcd_baseline.sbatch`
- [ ] `q202_dcd_cdloss.sbatch`
- [ ] `q203_dcd_replay.sbatch`
- [ ] `q204_dcd_full.sbatch`
- [ ] Multi-seed variants for all 4 configs

### Analysis Scripts

- [ ] `scripts/analyze_dcd_results.py` - Compare all ablations
- [ ] Statistical comparison against baseline MSE
- [ ] Ablation contribution breakdown

## Success Criteria

### Minimum Success
- Any configuration beats baseline MSE (0.0096761)
- Understanding which components are necessary

### Moderate Success
- Q-204 (full) beats baseline by 1-3%
- Clear ablation story showing contribution of each component

### Maximum Success
- Q-204 beats baseline by 3-5%
- Publishable result: "How to properly train EBMs with short-run sampling"
- Clear guidelines for future IRED-style methods

## Literature Support

**Key Papers**:

1. **Contrastive Divergence** (Hinton, 2002)
   - Foundation for training EBMs with short-run MCMC
   - Matches our CD objective

2. **Diffusion Contrastive Divergence** (Gao et al., 2021)
   - Replaces Langevin with diffusion transitions
   - Addresses bias from short-run MCMC
   - Directly applicable to our setup

3. **False Negative Debiasing** (Chuang et al., NeurIPS 2020)
   - Hard negatives need debiasing
   - Our residual filter is task-specific debiasing

4. **Persistent Contrastive Divergence** (Tieleman, 2008)
   - Replay buffer / persistent chains
   - Standard for scalable EBM training

## Timeline Estimate

**Implementation**: 3-5 days
- Langevin sampling: 1 day
- Replay buffer: 1 day
- CD loss: 1 day
- Residual filter + schedule: 1-2 days

**Compute**: ~60 GPU-hours (40 experiments × 1.5h)
- Can parallelize to ~6-12h wall-clock time

**Analysis**: 2-3 days
- Result aggregation
- Statistical testing
- Ablation contribution analysis

**Total**: ~1.5-2 weeks

## Risk Mitigation

**Risk 1**: Implementation bugs in complex changes
- **Mitigation**: Test each component incrementally
- Start with Q-202 (simplest), validate before moving to Q-203/204

**Risk 2**: Full implementation still doesn't beat baseline
- **Mitigation**: This is still valuable (negative result)
- Documents what DOESN'T work for IRED-style methods
- Still publishable

**Risk 3**: Replay buffer memory issues
- **Mitigation**: Buffer size configurable (default 10K)
- Can reduce if needed, monitor memory usage

## Next Steps (Immediate)

1. **Implement Langevin sampling** (highest priority)
   - Modify `opt_step` or create new `sample_negatives` function
   - Add noise parameter to config

2. **Implement replay buffer** (high priority)
   - Create ReplayBuffer class
   - Integrate with training loop

3. **Implement CD loss** (high priority)
   - Replace CrossEntropy with energy difference
   - Test on Q-202 config

4. **Create SLURM scripts** (medium priority)
   - Based on existing q001-q004 templates
   - Multi-seed support

5. **Run pilot experiment** (validation)
   - Quick test (10K steps) to validate implementation
   - Check for NaNs, numerical issues

---

**Document Created**: 2026-02-16
**Investigation Status**: SETUP (ready for implementation)
