# IRED-DCD Investigation Setup Summary

**Created**: 2026-02-16
**Status**: ✓ Infrastructure Complete, Ready for Implementation

---

## What Was Set Up

### 1. Configuration Files ✓

Created 4 experiment configs for ablation study:

| Config | Description | Key Features |
|--------|-------------|--------------|
| `q201_dcd_baseline.json` | Baseline (current NCE) | opt_steps=10, noise_scale=1.5, NCE loss |
| `q202_dcd_cdloss.json` | CD loss + Langevin | + use_langevin, use_dcd_loss, k=10 |
| `q203_dcd_replay.json` | + Replay buffer | + replay_buffer (size=10K, p=0.95) |
| `q204_dcd_full.json` | Full implementation | + residual_filter + energy_schedule |

**Location**: `projects/ired/configs/q20*.json`

### 2. SLURM Batch Scripts ✓

Created 4 sbatch scripts for cluster execution:

- `q201_dcd_baseline.sbatch` - Baseline reference
- `q202_dcd_cdloss.sbatch` - CD + Langevin
- `q203_dcd_replay.sbatch` - + Replay buffer
- `q204_dcd_full.sbatch` - Full IRED-DCD

**Location**: `projects/ired/slurm/jobs/q20*.sbatch`

**Resource allocation**:
- Partition: `gpu_test` (for 2h runtime)
- GPU: 1x GPU
- CPU: 2 cores
- RAM: 16GB
- Time: 2 hours

### 3. Documentation ✓

Created comprehensive documentation:

| Document | Purpose |
|----------|---------|
| `ired-dcd-plan.md` | High-level investigation plan, rationale, literature support |
| `ired-dcd-implementation.md` | Detailed code specifications, API designs, integration points |
| `ired-dcd-setup-summary.md` | This file - setup checklist and next steps |

**Location**: `projects/ired/documentation/`

### 4. Pipeline State ✓

Updated `pipeline.json` to track new investigation:

```json
{
  "phase": "IMPLEMENT",
  "current_investigation": "ired-dcd",
  "ired_dcd_investigation": {
    "status": "SETUP",
    "target_baseline_mse": 0.00969245,
    "implementation_tasks": {
      "langevin_sampling": "pending",
      "replay_buffer": "pending",
      "cd_loss": "pending",
      "residual_filter": "pending",
      "energy_schedule": "pending"
    }
  }
}
```

---

## Experiment Design

### Ablation Study

4 configurations, each run with 10 seeds (40 total experiments):

```
Q-201 (Baseline)    →  Current NCE (for comparison)
       ↓
Q-202 (+CD)         →  + CD loss + Langevin sampling
       ↓
Q-203 (+Replay)     →  + Replay buffer (persistent chains)
       ↓
Q-204 (Full)        →  + Residual filter + Energy schedule
```

### Key Hyperparameters

**Changed from original investigation**:
- `mining_opt_steps`: 2 → **10** (more refinement)
- `mining_noise_scale`: 3.0 → **1.5** (less aggressive init)

**New parameters**:
- `langevin_sigma_multiplier`: 1.414 (√2)
- `replay_buffer_size`: 10,000
- `replay_sample_prob`: 0.95
- `residual_filter_percentile`: 0.3 (30th percentile)
- `energy_loss_warmup_steps`: 20,000
- `energy_loss_max_weight`: 0.1 (reduced from 0.5)

### Success Criteria

**Target**: Beat baseline MSE of **0.00969245** (from original investigation)

- **Minimum**: Any config beats baseline
- **Moderate**: Q-204 beats baseline by 1-3%
- **Maximum**: Q-204 beats baseline by 3-5% (MSE < 0.0092)

---

## Implementation Roadmap

### Phase 1: Core Components (3-5 days)

Priority order:

1. **Langevin Sampling** (Day 1)
   - [ ] Add `sample_negatives_langevin()` method
   - [ ] Test: Check negatives are diverse, energies decrease

2. **Replay Buffer** (Day 2)
   - [ ] Create `diffusion_lib/replay_buffer.py`
   - [ ] Integrate with GaussianDiffusion1D
   - [ ] Test: Buffer fills, samples reused at 95% rate

3. **CD/DCD Loss** (Day 3)
   - [ ] Replace NCE with energy difference loss
   - [ ] Add conditional logic (use_dcd_loss flag)
   - [ ] Test: Loss values reasonable (not NaN/Inf)

4. **Residual Filter** (Day 4)
   - [ ] Add `compute_matrix_residual()` helper
   - [ ] Integrate filtering in p_losses
   - [ ] Test: ~70% negatives kept (30th percentile)

5. **Energy Schedule** (Day 4-5)
   - [ ] Add warmup + timestep range logic
   - [ ] Pass global_step from trainer (requires trainer mod)
   - [ ] Test: Warmup curve looks correct

### Phase 2: Pilot Experiments (1 day)

Quick validation with 10K steps:

```bash
# Run pilot for each config (locally or on cluster)
python experiments/matrix_inversion_mining.py --config configs/q201_dcd_baseline.json --train-steps 10000
python experiments/matrix_inversion_mining.py --config configs/q202_dcd_cdloss.json --train-steps 10000
python experiments/matrix_inversion_mining.py --config configs/q203_dcd_replay.json --train-steps 10000
python experiments/matrix_inversion_mining.py --config configs/q204_dcd_full.json --train-steps 10000
```

**Validation checklist**:
- [ ] All 4 configs run without errors
- [ ] Loss curves are reasonable
- [ ] Q-202/203/204 show improvement trend vs Q-201
- [ ] No NaN/Inf values

### Phase 3: Full Experiments (1-2 days)

Submit all 4 configs to cluster with 10 seeds each:

**Compute estimate**: 40 experiments × 1.5h = **60 GPU-hours**
**Wall-clock time**: ~6-12h (with parallelization)

```bash
# Via dispatch or manual submission
scripts/cluster/submit.sh projects/ired/slurm/jobs/q201_dcd_baseline.sbatch
scripts/cluster/submit.sh projects/ired/slurm/jobs/q202_dcd_cdloss.sbatch
scripts/cluster/submit.sh projects/ired/slurm/jobs/q203_dcd_replay.sbatch
scripts/cluster/submit.sh projects/ired/slurm/jobs/q204_dcd_full.sbatch

# Then multi-seed variants (need to create)
```

**Note**: May need to create multi-seed versions (q201_multiseed, etc.) similar to q101_multiseed pattern.

### Phase 4: Analysis (2-3 days)

- [ ] Aggregate results from all 40 experiments
- [ ] Statistical comparison (t-tests, effect sizes)
- [ ] Ablation contribution analysis
- [ ] Learning curve visualization
- [ ] Document findings

---

## File Modifications Needed

### New Files to Create

1. `diffusion_lib/replay_buffer.py` - ReplayBuffer class
2. `scripts/analyze_dcd_results.py` - Results aggregation (optional)

### Existing Files to Modify

1. **`diffusion_lib/denoising_diffusion_pytorch_1d.py`** (main changes)
   - Add `sample_negatives_langevin()` method (~20 lines)
   - Add replay buffer integration (~30 lines)
   - Replace NCE with CD loss (~15 lines)
   - Add residual filtering (~25 lines)
   - Add energy scheduling (~20 lines)
   - **Total**: ~110 lines of changes

2. **`experiments/matrix_inversion_mining.py`** (minor changes)
   - Add CLI args for new config params (optional, configs already handle this)
   - **Total**: ~20 lines (optional)

### No Changes Needed

- `dataset.py` - Matrix inversion dataset unchanged
- `models.py` - EBM architecture unchanged
- `train.py` - Trainer unchanged (unless we add global_step passing)

---

## Quick Start: Implementation

### Option 1: Sequential Implementation

Work through components one at a time:

```bash
# Day 1: Langevin
# - Implement sample_negatives_langevin()
# - Test Q-202 with 1000 steps

# Day 2: Replay Buffer
# - Create replay_buffer.py
# - Integrate with diffusion
# - Test Q-203 with 1000 steps

# Day 3: CD Loss
# - Replace NCE with energy difference
# - Test Q-202/203 with 1000 steps

# Day 4: Residual Filter + Schedule
# - Add filtering logic
# - Add scheduling logic
# - Test Q-204 with 1000 steps

# Day 5: Pilot experiments
# - Run all 4 configs with 10K steps
# - Validate results
```

### Option 2: Parallel Implementation

Use the Task tool to parallelize independent components:

```bash
# Launch 3 parallel implementation tasks:
# 1. Langevin sampling
# 2. Replay buffer
# 3. CD loss

# Then integrate and add:
# 4. Residual filter + schedule
```

---

## Testing Commands

### Local Testing (CPU/MPS)

```bash
# Quick sanity check (10 steps)
python experiments/matrix_inversion_mining.py \
    --config configs/q201_dcd_baseline.json \
    --train-steps 10

# Pilot run (1000 steps, ~5 min)
python experiments/matrix_inversion_mining.py \
    --config configs/q202_dcd_cdloss.json \
    --train-steps 1000
```

### Cluster Testing

```bash
# Submit pilot job (10K steps, ~15 min)
scripts/cluster/submit.sh projects/ired/slurm/jobs/q201_dcd_baseline.sbatch

# Check status
scripts/cluster/status.sh <job_id>

# Fetch logs
scripts/cluster/remote_fetch.sh ired
```

---

## Risk Mitigation

### Implementation Risks

**Risk 1**: Langevin sampling causes NaN/Inf
- **Mitigation**: Clamp values, reduce sigma, add gradient clipping

**Risk 2**: Replay buffer memory issues
- **Mitigation**: Buffer size is configurable (default 10K is ~40MB for 20×20 matrices)

**Risk 3**: CD loss unstable
- **Mitigation**: Start with small energy_loss_weight (0.05-0.1), monitor loss curves

**Risk 4**: Residual filter too aggressive
- **Mitigation**: Percentile is configurable, can relax to 50th percentile

### Experimental Risks

**Risk 1**: Full implementation still doesn't beat baseline
- **Mitigation**: Publishable negative result - documents what doesn't work
- Expected: At least one ablation (Q-202 or Q-203) shows improvement

**Risk 2**: Ablations show no clear trend
- **Mitigation**: Re-examine hyperparameters, try different schedules
- Fallback: Document failure modes for future work

---

## Next Steps (Immediate)

### Ready to Start Implementation?

**Option A**: Start with Langevin sampling (highest impact)
- Implement `sample_negatives_langevin()` in diffusion code
- Test with Q-202 config (1000 steps)
- Verify negatives are diverse and energies decrease

**Option B**: Start with replay buffer (infrastructure)
- Create `replay_buffer.py`
- Integrate with diffusion
- Test buffer fill rate and sampling

**Option C**: Do a quick end-to-end pilot first
- Run Q-201 baseline (current code) with 10K steps
- Verify infrastructure works before implementing changes
- Establishes clean baseline for comparison

### Recommended Approach

1. **Run Q-201 baseline pilot first** (validate infrastructure)
2. **Implement Langevin + CD loss** (Q-202)
3. **Add replay buffer** (Q-203)
4. **Add residual filter + schedule** (Q-204)
5. **Run all 4 pilots** (10K steps each)
6. **Submit full experiments** (100K steps, 10 seeds)

---

## Summary Checklist

### Infrastructure ✓

- [x] 4 config files created (q201-q204)
- [x] 4 SLURM scripts created (q201-q204)
- [x] Documentation created (plan + implementation guide)
- [x] Pipeline state updated
- [x] Multi-seed configs (TODO: create later if needed)

### Implementation ⚠

- [ ] Langevin sampling
- [ ] Replay buffer
- [ ] CD/DCD loss
- [ ] Residual filtering
- [ ] Energy scheduling

### Validation ⚠

- [ ] Component tests
- [ ] Pilot experiments (10K steps)
- [ ] Full experiments (100K steps, 10 seeds)
- [ ] Results analysis

---

**Status**: Ready to begin implementation!

**Estimated time to first results**: 1 week (implementation + pilots)
**Estimated time to completion**: 1.5-2 weeks (full experiments + analysis)

**Questions?** See `documentation/ired-dcd-plan.md` for theoretical background or `documentation/ired-dcd-implementation.md` for detailed code specifications.
