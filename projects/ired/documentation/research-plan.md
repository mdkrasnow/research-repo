# Research Plan — Adversarial Negative Mining Investigation

## Project Status: Results Analysis & Investigation

**Current Phase**: ANALYZE / DESIGN
**Last Updated**: 2026-02-19

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

---

## Scalar Energy Head Investigation [2026-02-19]

### Summary of Findings

The scalar energy head (`E = g(h) = Linear(h, 1)`) is **unstable without energy scale control**. The root cause and proposed fixes are documented below.

#### Confirmed facts
- q101 (vector energy, no mining): val MSE → 0.009 ✅
- q207 (scalar energy, adversarial mining): val MSE → 0.098 stable (plateau, not divergence)
- q209 (scalar energy, no mining): val MSE 0.11 @ 4k → **2.2 @ 11k** (diverges)
- q201 (vector energy, adversarial mining): val MSE 0.091 @ 8k → **1.10 @ 100k** (diverges)
- Energy magnitudes at step 1k: q209 E_pos≈337k, q207 E_pos≈3k — 100x difference
- Mining acts as accidental regularizer in q207 (counterpressure via margin loss)

#### The Core Problem

The scalar head has no natural scale: `E = w^T h + b` where `w, b` are unconstrained. Without mining or regularization, the network can minimize NCE loss by making all energies large and positive (since the gradient-based denoising signal pushes pred up, not down on energy). The result is runaway energy magnitudes that corrupt the `∇_y E` denoising gradients.

---

### Brainstorm: Approaches to Fix Energy Scale

#### Approach 1: Energy Regularization (`energy_reg_weight > 0`)
**Idea**: Add `λ * E_pos^2` (or `λ * |E_pos|`) to the loss directly.
- **Pro**: Simplest 1-line fix, already has a config knob (`energy_reg_weight`)
- **Pro**: Directly attacks the root cause (unbounded magnitude)
- **Con**: Hyperparameter sensitivity — too large λ collapses energies to zero; too small and instability remains
- **Con**: L2 regularization biases toward small energies, may conflict with NCE margin requirement
- **Tuning needed**: λ likely needs to be in [0.001, 0.1] range; requires sweep
- **Risk**: May just shift the plateau rather than fix it — unclear if it enables convergence to 0.009

#### Approach 2: Energy Normalization (LayerNorm / spectral norm on scalar head)
**Idea**: Apply LayerNorm to the features before the scalar head, or use spectral normalization on the linear layer weights.
- **Pro**: Architectural fix — constrains the scale of the linear layer without requiring a tuned λ
- **Pro**: LayerNorm is well-understood and stable; spectral norm clips the Lipschitz constant
- **Con**: Changes the expressivity of the energy function — may prevent the model from learning the right energy landscape
- **Con**: Spectral norm with a scalar output layer is unusual (constrains the norm of a vector, not a matrix)
- **Variant**: Normalize `h` before passing to scalar head, keeping the scalar head itself unconstrained
- **Risk**: May just push the problem into the backbone features

#### Approach 3: Energy Clamping / Gradient Clipping on Energy
**Idea**: Hard-clamp `E_pos` to a fixed range (e.g., [−100, 100]) or clip gradients from the energy head separately.
- **Pro**: Hard guarantee on energy magnitude
- **Con**: Non-differentiable clamp in the energy function corrupts backprop
- **Con**: Gradient clipping doesn't prevent forward-pass energy blowup, only limits weight updates
- **Verdict**: Gradient clipping is a reasonable auxiliary measure but not a primary fix

#### Approach 4: Whitened / Normalized Energy Target
**Idea**: Instead of predicting raw energy, predict a normalized score: `E = tanh(g(h))` or `E = sigmoid(g(h))` — bounded scalar outputs.
- **Pro**: Hard bounds on energy magnitude (tanh: (−1,1), sigmoid: (0,1))
- **Pro**: No extra hyperparameter
- **Con**: `tanh` saturates — gradients vanish for large inputs, which can kill learning in a different way
- **Con**: The EBM theory assumes energies can be arbitrary; bounding them changes the semantics
- **Con**: `∇_y E` is still well-defined, but the effective gradient field may be poorly conditioned near saturation
- **Verdict**: Interesting but risky; saturation is a known problem with bounded activations in deep EBMs

#### Approach 5: Stop-Gradient on Energy in Denoising Loss
**Idea**: When computing `∇_y E` for denoising, stop gradient through the energy head weights. The energy head is trained only by the NCE/contrastive loss, not pulled by denoising.
- **Pro**: Decouples the two objectives cleanly — energy head learns a landscape shape, denoiser uses that shape
- **Pro**: Prevents the denoising loss from destabilizing the energy parameterization
- **Con**: Changes the training dynamics significantly — the energy head may not adapt to the denoiser's needs
- **Con**: Requires careful implementation (stop_gradient in PyTorch = `.detach()`)
- **Verdict**: Architecturally interesting and theoretically motivated, but a bigger change to the training loop

#### Approach 6: Warm-Start Energy Head After Denoiser Convergence
**Idea**: Train the denoiser alone (no energy loss) for N steps until it converges, then introduce the energy head.
- **Pro**: Sidesteps the coupling problem — denoiser is already at a good point before energy is added
- **Pro**: Matches the q101 result as a starting point, then adds scalar energy on top
- **Con**: Two-phase training is more complex to implement and schedule
- **Con**: When energy is introduced, it still might destabilize the already-converged denoiser
- **Verdict**: Could work but is operationally complex; not the first thing to try

#### Approach 7: Use Mining as Intentional Regularizer (Controlled)
**Idea**: Accept that mining stabilizes the scalar head (as in q207) but weaken the mining enough that it doesn't degrade denoising. E.g., use random mining instead of adversarial, or use very short mining horizon.
- **Pro**: No architectural changes — just tuning existing knobs
- **Pro**: q207 proves mining-as-regularizer works; random mining was shown to be neutral vs baseline (q102: same MSE as baseline, p=0.088)
- **Con**: q207 with adversarial mining still plateaued at 0.098 — need random mining specifically
- **Next experiment**: scalar energy + random mining (no contrastive) — should be stable like q207 but without the adversarial-mining degradation
- **Verdict**: Highest probability of quickly reaching 0.009; lowest implementation risk

#### Approach 8: Energy L∞ or Soft-Clamp via Huber-style Loss
**Idea**: Use a Huber-style loss on energy: penalize `max(0, |E| - δ)^2` for some threshold δ.
- **Pro**: Only penalizes when energy exceeds threshold δ, leaving normal operation unconstrained
- **Pro**: Can set δ based on observed healthy energy scale (q207 E_pos ≈ 100k–140k, so δ ≈ 200k)
- **Con**: Requires knowing the "right" energy scale in advance (or tuning δ)
- **Verdict**: More principled than hard clamp but still requires tuning

---

### Recommended Experiment Order

**Priority 1** (lowest risk, highest probability of reaching 0.009):
- **q210**: Scalar energy + random mining (no adversarial, no contrastive)
  - Mining provides stability regularizer but without adversarial OOD degradation
  - Direct comparison to q207 (adversarial) and q101 (no mining)
  - If val MSE → 0.009: scalar head works fine, adversarial mining was the plateau cause
  - If val MSE → 0.098: random mining also stabilizes but also plateaus → mining itself is the issue

**Priority 2** (architectural fix, clean solution if it works):
- **q211**: Scalar energy + energy_reg_weight sweep (λ ∈ {0.001, 0.01, 0.1}) + no mining
  - Tests whether regularization alone fixes the instability without needing mining at all
  - Run 3 single-seed jobs first, pick best λ, then run full 10-seed

**Priority 3** (if both above fail):
- **q212**: Scalar energy + stop-gradient on energy head + no mining
  - Decouples the two objectives; more principled but larger code change

**Deprioritized**:
- Bounded activations (tanh/sigmoid) — saturation risk too high
- Warm-start — operationally complex, try regularization first
- Huber loss — adds another hyperparameter without clear advantage over L2 reg

---

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
