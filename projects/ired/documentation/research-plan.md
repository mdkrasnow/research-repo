# Research Plan — Adversarial Negative Mining Investigation

## Project Status: Results Analysis & Investigation

**Current Phase**: ANALYZE
**Last Updated**: 2026-01-21

---

## Executive Summary

### Phase 0 Results (COMPLETED - 2026-01-21)

Initial experiments (Q-001, Q-002, Q-003) revealed **unexpected results** that contradict the original hypothesis:

| Strategy | Train MSE | Val MSE | Δ vs Baseline | Status |
|----------|-----------|---------|---------------|--------|
| **Baseline** (none) | 0.0096721 | 0.0096761 | — | ✓ Converged |
| **Random** mining | 0.00968831 | 0.00968396 | +0.08% | ✓ Converged |
| **Adversarial** mining | 0.00980961 | 0.00982675 | **+1.56%** | ✓ Converged (worse) |

**Key Finding**: Adversarial hard negative mining **degrades** performance by ~1.5% compared to baseline, contrary to expectations.

**Adversarial Configuration** (from q003_adversarial.json):
- `mining_opt_steps`: 2 (gradient ascent steps per negative sample)
- `mining_noise_scale`: 3.0 (noise added to prevent degenerate samples)
- `learning_rate`: 0.0001
- All experiments: 100K iterations, batch_size=2048, 20×20 matrices

### Investigation Plan

The investigation phase aims to:
1. **Validate** these results with multi-seed experiments (confirm statistical significance)
2. **Diagnose** why adversarial mining fails (matrix conditioning, false negatives, diversity loss)
3. **Rescue** adversarial mining via hyperparameter tuning (if possible)
4. **Stabilize** training with LR decay, gradient clipping, EMA, and early stopping

---

## Original Research Question

**Does adversarial negative mining improve performance on the matrix inversion task compared to baseline diffusion training and random negative mining?**

**Answer from initial results**: **NO** - Adversarial mining degraded performance by ~1.5%.

**New research question**: **Why does adversarial mining fail for matrix inversion, and can it be fixed?**

---

## Investigation Strategy

### Phase 1: Validate Results (Priority: HIGH)

**Goal**: Confirm adversarial underperformance is real, not a fluke

**Experiments**:
- **Q-101**: Multi-seed baseline (10 seeds)
- **Q-102**: Multi-seed random (10 seeds)
- **Q-103**: Multi-seed adversarial (10 seeds)
- **Q-104**: Learning curve analysis with frequent checkpointing

**Deliverables**:
- Mean ± std for each strategy
- Confidence intervals
- Rank consistency test (which strategy wins most seeds)
- Learning curves showing when/if adversarial diverges

**Expected Outcome**: Adversarial consistently underperforms across seeds (confirms real effect vs noise)

---

### Phase 2: Diagnose Root Cause (Priority: HIGH)

**Goal**: Understand mechanism of adversarial mining failure

**Hypothesis 1: Pathological Matrix Conditioning**
- Adversarial optimization pushes toward ill-conditioned (near-singular) matrices
- Matrix inversion has natural adversarial direction: high condition number
- Model learns "anti-inversion" patterns instead of true inversion

**Experiment Q-201**: Matrix conditioning analysis
- Track: det(A), cond(A), σ_min for positive vs adversarial negative samples
- Compare distributions
- Test: Do adversarial negatives have significantly higher condition numbers?

**Hypothesis 2: False Negatives / Diversity Loss**
- Adversarial negatives collapse to narrow region near positives
- Loss of diversity hurts generalization
- Some "hard negatives" may actually be false negatives (valid inverses)

**Experiment Q-202**: Energy gap and diversity profiling
- Track: E(pos) - E(neg) gap over training
- Measure: Distance between consecutive adversarial samples (diversity)
- Detect: Negatives that are "too close" to positives (false negative rate)
- Monitor: Gradient norms during mining

**Deliverables**:
- Statistical comparison of matrix properties (pos vs neg)
- Energy gap plots, gradient norm distributions
- False negative rate over training
- Diagnosis: Which hypothesis (or both) explains the failure

---

### Phase 3: Rescue Adversarial Mining (Priority: MEDIUM)

**Goal**: Test if adversarial mining can be made to work via hyperparameter tuning

**Rationale**: Hard negative mining is well-studied in literature. Failure modes exist, but often can be addressed by:
- Controlling hardness (fewer opt steps, smaller ascent LR)
- Adding diversity (noise injection, mixed strategies)
- Constraining negatives (projection to valid manifold)

**Experiments**:
- **Q-301**: Mining opt_steps sweep [1, 3, 5, 10, 20]
  - Hypothesis: Current setting may be too high, making negatives pathologically hard
- **Q-302**: Ascent LR sweep [0.01, 0.05, 0.1, 0.5, 1.0]
  - Hypothesis: Smaller step size may prevent ill-conditioned samples
- **Q-303**: Noise scale sweep [0.01, 0.1, 0.5, 1.0, 2.0]
  - Hypothesis: Adding noise prevents degenerate adversarial samples
- **Q-304**: Mixed strategy (70% random + 30% adversarial)
  - Hypothesis: Mixing preserves diversity while adding challenge

**Deliverables**:
- Performance vs hyperparameter curves
- Identification of best adversarial configuration (if any)
- Verdict: Can adversarial mining match baseline? If not, why?

---

### Phase 4: Training Stability (Priority: HIGH)

**Goal**: Fix late-stage training instability observed in validation run

**Observation**: Separate validation run showed best MSE at step ~38K, then divergence at 85K-99K

**Problem**: Current training setup lacks:
- Learning rate decay (constant LR for 100K steps)
- Gradient clipping (large gradients can destabilize)
- EMA for stable evaluation
- Early stopping (continues training even when degrading)

**Experiments**:
- **Q-401**: Stabilized training (cosine LR decay + grad clipping + EMA)
  - Apply to all three strategies
  - Expected: Improved final performance, no late divergence
- **Q-402**: Early stopping based on validation MSE
  - Stop when no improvement for 10K steps
  - Save best checkpoint separately

**Deliverables**:
- Stable learning curves for all strategies
- Cleaner performance comparisons (best checkpoint vs final checkpoint)
- Training time reduction (early stopping)

**Expected Outcome**: All strategies improve, but relative ranking likely unchanged

---

## Theoretical Framework

### Why Hard Negatives Can Fail

**Context from Literature**:

1. **False Negatives** (Chuang et al., NeurIPS 2020; Huynh et al., WACV 2022)
   - Hard negatives can accidentally be true positives in disguise
   - Model is punished for correct behavior
   - Well-documented failure mode in contrastive learning

2. **Pathological Hardness** (Robinson et al., 2020; Kalantidis et al., NeurIPS 2020)
   - "Harder is not monotonically better"
   - Overly hard negatives can be outliers/adversarial artifacts
   - Need to balance hardness with diversity

3. **EBM Training Instability** (Du et al., 2021; Yin et al., ECCV 2022)
   - Energy-based models with adversarial-style negative sampling are notoriously unstable
   - Gradient shortcuts and optimization pathologies
   - Requires careful design and stability techniques

### Task-Specific Considerations for Matrix Inversion

**Matrix inversion has unique adversarial geometry**:
- Near-singular matrices (high condition number) are naturally "adversarial"
- Gradient ascent on energy may push toward det(A) → 0
- This creates "anti-inversion" patterns rather than hard-but-valid examples

**Prediction**: Conditioning analysis will show adversarial negatives cluster at high cond(A)

---

## Success Criteria

### Minimum Success (Investigation Complete)
- Multi-seed validation confirms adversarial underperforms (statistical significance)
- Root cause identified via conditioning + energy analysis
- Documentation of failure mode suitable for publication

### Moderate Success (Adversarial Partially Rescued)
- Hyperparameter sweep finds configuration where adversarial matches baseline
- Understanding of when/why adversarial mining works for matrix inversion

### Maximum Success (Adversarial Fully Works)
- Adversarial mining reliably outperforms baseline with proper tuning
- Publishable result: "How to apply hard negative mining to structured prediction tasks"

### Publication-Worthy Outcomes (Any of)
1. **Negative result**: "Why harder ≠ better for matrix inversion EBMs" + geometry explanation
2. **Diagnostic methods**: Matrix conditioning analysis as evaluation tool for contrastive learning
3. **Rescue strategy**: Mixed negative sampling for structured tasks
4. **Training stability**: Best practices for EBM training with negative mining

---

## Timeline Estimate

**Phase 1 (Validation)**:
- Implementation: 2-3 days
- Compute: ~15h (can parallelize to 1.5h with array jobs)
- Analysis: 1 day

**Phase 2 (Diagnosis)**:
- Implementation: 3-4 days (instrumentation + analysis tools)
- Compute: ~4h
- Analysis: 2 days

**Phase 3 (Rescue Attempts)**:
- Implementation: 2-3 days (hyperparameter configs)
- Compute: ~30h (can parallelize significantly)
- Analysis: 2 days

**Phase 4 (Stability)**:
- Implementation: 2 days
- Compute: ~6h
- Analysis: 1 day

**Total**: ~2-3 weeks (implementation + compute + analysis)

---

## Risk Assessment

**Risk 1: Multi-seed validation reveals results are noise**
- Mitigation: If true, this is valuable (negative result: no strategy wins)
- Probability: Low (1.5% difference is large enough to likely survive replication)

**Risk 2: Root cause unclear from diagnostics**
- Mitigation: Even partial understanding is publishable
- Probability: Medium (may need additional probes beyond Q-201/202)

**Risk 3: Adversarial mining cannot be rescued**
- Mitigation: This is itself a strong result (fundamental limitation discovered)
- Probability: Medium-High (task geometry may fundamentally conflict with adversarial mining)

**Risk 4: Training instability affects all strategies**
- Mitigation: Phase 4 addresses this; improves all baselines
- Probability: Low (baseline already converges well, but stability helps)

---

## Next Steps (Immediate)

1. **Review existing adversarial config** (T0.2)
   - Check current values of mining_opt_steps, ascent LR, noise scale
   - Establish baseline for comparison

2. **Implement multi-seed infrastructure** (T1.1, T1.2)
   - Create configs for Q-101/102/103
   - Set up SLURM array jobs for parallel execution

3. **Start Phase 1 experiments** (Q-101, Q-102, Q-103)
   - Submit multi-seed validation runs
   - Expected: 1.5h compute time (parallelized), results within same day

4. **Implement diagnostic instrumentation** (T2.1, T2.2) in parallel
   - Matrix conditioning analysis
   - Energy gap profiling
   - Ready for Phase 2 as soon as Phase 1 completes

---

## References

**Key Papers Informing Investigation**:

1. **Debiased Contrastive Learning** (Chuang et al., NeurIPS 2020)
   - Problem: Random negatives can include false negatives
   - Solution: Debiasing techniques

2. **False Negative Cancellation** (Huynh et al., WACV 2022)
   - Problem: False negatives harm contrastive learning
   - Solution: Attract false negatives instead of repelling

3. **ProGCL** (Xia et al., ICML 2022)
   - Problem: Hard negatives in graphs are often false negatives
   - Solution: Careful selection and filtering

4. **Hard Negative Mixing** (Kalantidis et al., NeurIPS 2020)
   - Problem: Pure hard negatives can hurt
   - Solution: Mix hard and random negatives

5. **Contrastive Learning with Hard Negative Samples** (Robinson et al., 2020)
   - Key insight: Hardness is not monotonic; need careful control

6. **Improved Contrastive Divergence for EBMs** (Du et al., 2021)
   - Problem: EBM training is unstable
   - Solution: Initialization, architecture, optimization tricks

7. **Learning EBMs with Adversarial Training** (Yin et al., ECCV 2022)
   - Problem: Adversarial-style procedures have complex dynamics
   - Insight: Not straightforward "harder = better"

---

## Appendix: Key Metrics to Track

### Performance Metrics
- Train MSE (mean squared error on training set)
- Validation MSE (mean squared error on validation set)
- Best checkpoint MSE (lowest validation MSE across training)
- Final checkpoint MSE (MSE at step 100K or early stopping)

### Diagnostic Metrics (Phase 2)
- **Matrix conditioning**:
  - det(A) distribution (positive vs negative samples)
  - cond(A) distribution
  - σ_min (smallest singular value) distribution
- **Energy landscape**:
  - E(pos) - E(neg) gap over time
  - Gradient norm during mining
  - False negative rate (similarity between neg and pos)
- **Diversity**:
  - Inter-sample distance for adversarial negatives
  - Coverage of negative sample space

### Training Metrics (Phase 4)
- Learning rate schedule
- Gradient norm (pre-clipping, post-clipping)
- EMA decay and divergence from raw weights
- Early stopping trigger point

---

## Document History

- **2026-01-21**: Document created after completing Q-001, Q-002, Q-003 and discovering adversarial underperformance
