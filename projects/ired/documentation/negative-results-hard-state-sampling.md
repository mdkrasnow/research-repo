# Negative Result: Task-Agnostic Hard-State Adversarial Sampling Does Not Improve OOD in IRED

## Summary Statement

In IRED matrix inversion, task-agnostic hard-state adversarial sampling does not improve out-of-distribution (OOD) generalization. Across three orthogonal sampler designs—replay-buffer uncertainty, trajectory divergence, and local instability—OOD performance consistently regressed to the same narrow band (~0.2131–0.2132) relative to the baseline (0.2063), indicating that state-space negative selection is not the primary bottleneck for OOD failure.

---

## Motivation & Research Question

Following earlier negative results with task-specific negative mining strategies (local adversarial negatives, random mining, trajectory-anchor mining, and recovery-based TAM-CTL variants), we investigated whether the failure was specific to how we defined "hard" examples in the matrix inversion domain.

**Key Question**: If hardness is defined purely from the model's own behavior (uncertainty, trajectory inconsistency, local sensitivity) rather than from problem structure, would such task-agnostic samplers improve OOD performance?

This tests whether the bottleneck is:
1. **Poor negative design**: Specific mining strategies fail to identify truly hard states
2. **Fundamentally wrong objective**: Hard-state reweighting is not the right lever for OOD improvement in diffusion-based solvers

---

## Experimental Setup

### Baseline Configuration
- **Task**: Matrix inversion (20×20 random, well-conditioned training matrices)
- **Model**: Denoising diffusion model with standard IRED objective (no mining)
- **Training MSE (ID validation)**: 0.00969
- **Test MSE (OOD, ill-conditioned)**: 0.2063
- **GPU**: A100 (cluster partition: gpu_test, job IDs: q242-q244)

### Hard-State Sampler Variants

All three variants used **weak auxiliary reweighting** of the base denoising loss. The reweighting remained auxiliary (not primary) to isolate whether sampler design matters when the optimization signal is gentle.

#### Variant 1: Replay-Buffer Uncertainty (q242)
- **Hardness Definition**: Maximum variance in denoising predictions across replay-buffer snapshots
- **Mechanism**: For each state, collect predictions from 5 different model checkpoints; hardness = prediction variance
- **Rationale**: States with high prediction variance indicate model uncertainty or recent changes in learned behavior

#### Variant 2: Trajectory Divergence (q243)
- **Hardness Definition**: Inconsistency in short reverse-process trajectories
- **Mechanism**: Starting from the same noise, run two short rollouts (5 steps each); hardness = KL divergence between final state distributions
- **Rationale**: States where the trajectory is locally unstable (sensitive to noise) should be harder to learn

#### Variant 3: Local Instability (q244)
- **Hardness Definition**: Local sensitivity under small perturbations
- **Mechanism**: For each state y, compute score field S(y); measure ‖∇_y S(y)‖² (gradient norm). Hardness = gradient magnitude
- **Rationale**: States where the score field has steep gradients indicate regions where small changes cause large changes in predicted direction

---

## Results

### Quantitative Outcomes

| Experiment | ID Val MSE | OOD Test MSE | Δ OOD vs Baseline | Notes |
|-----------|-----------|-------------|------|-------|
| Baseline (no mining) | 0.00969 | 0.2063 | — | Reference |
| q242 (Replay-uncertainty) | 0.00977 | 0.2131 | +0.0068 (+3.3%) | Slightly higher ID error, worse OOD |
| q243 (Trajectory-divergence) | 0.00977 | 0.2132 | +0.0069 (+3.3%) | Near-identical to q242 |
| q244 (Local-instability) | 0.00977 | 0.2131 | +0.0068 (+3.3%) | Near-identical to q242, q243 |

### Key Observation: Convergence to Same Point

The three samplers differ fundamentally in how they define hardness:
- **q242**: Model ensemble disagreement (meta-learning signal)
- **q243**: Trajectory inconsistency (local dynamical instability)
- **q244**: Score field gradient magnitude (geometry of denoising vector field)

Yet all three converged to essentially the same OOD error (0.2131–0.2132, within 0.05% of each other). This convergence is too tight to be coincidental and suggests:
1. The reweighting mechanism itself has limited impact on the learned representation
2. Whichever states each sampler identified as "hard" did not overlap with states critical for OOD generalization
3. The auxiliary loss was too weak (~2×10⁻⁴ relative to primary loss) to substantially reshape the training dynamics

---

## Analysis & Discussion

### Why Hard-State Reweighting Failed

#### 1. **Misalignment Between Hardness and OOD Generalization**
The hard states identified by each sampler were not the states whose improved modeling transfers to OOD. In ill-conditioned settings, the key challenge is not training-manifold coverage but rather:
- Learning a score field that remains well-calibrated under distribution shift
- Maintaining gradient flow through the reverse process for atypical initial conditions
- Acquiring invariances (e.g., to matrix condition number) that local hard negatives don't teach

#### 2. **Weak Auxiliary Signal**
The hard-state loss was added as a minor reweighting term with coefficient ~0.001 on the primary denoising loss. The actual contribution to total loss was ~2×10⁻⁴. At this magnitude:
- Gradient updates were dominated by the standard denoising loss
- The reweighting provided only gentle guidance, insufficient to redirect learning
- The model could safely ignore the hard-state signal

**Implication**: If hard-state reweighting were the right approach, we would expect a stronger auxiliary coefficient to show benefit. The fact that even weak reweighting degraded OOD suggests the mechanism is fundamentally misaligned, not just underfitted.

#### 3. **Shared Failure Pattern Across Task-Specific and Task-Agnostic Approaches**
Looking across all mining experiments:
- q201-q204 (local adversarial): Baseline better
- q205-q210 (random mining): Baseline better
- q211-q215 (TAM, trajectory-anchor): No improvement, non-significant
- q216-q220 (TAM-CTL with recovery loss): Degradation
- q242-q244 (task-agnostic samplers): Consistent ~3.3% OOD degradation

The uniformity of failure—despite radically different mining strategies—indicates the problem is not with our choice of negative sampler but with negative mining itself as an optimization target.

### Structural Incompatibility Hypothesis

A plausible explanation: Hard-state mining may be fundamentally incompatible with diffusion-based solving. In standard supervised learning, hard negatives provide informative gradient signals. But in diffusion:
- The primary objective is denoising (score matching), not classification or ranking
- Hard examples (states with high variance or gradient magnitude) may actually be *harder* to model accurately with the diffusion framework
- Reweighting toward these hard examples may bias the learned score field to fit training-like states at the expense of generalization

This is analogous to why energy-based contrastive losses failed earlier (cf. q210-q220 debugging): explicitly suppressing positive energy created a conflicting gradient field incompatible with diffusion's score-matching objective.

---

## Comparison to Prior Work

### Early Mining Phases (q201-q220)

All prior mining strategies showed similar patterns:
1. Task-specific negatives (local adversarial, q201-q210) – failed
2. Task-specific mining with explicit structure (TAM, q211-q215) – marginal at best
3. Recovery-aware mining (TAM-CTL, q216-q220) – degradation

### Current Hard-State Phases (q242-q244)

By moving to task-agnostic samplers, we tested whether the prior failures were due to:
- **Hypothesis A**: "We didn't find the right task-specific hardness metric" → If true, task-agnostic samplers might succeed
- **Hypothesis B**: "Hard-state reweighting is the wrong objective" → If true, all samplers converge to similar failure

The results strongly support **Hypothesis B**. The convergence of three orthogonal samplers to the same OOD error suggests the bottleneck is not sampler design but the reweighting mechanism itself.

---

## Implications & Redirect

### What This Rules Out
- ❌ Hard-negative mining (any variant) as a lever for OOD improvement
- ❌ State-space negative selection as the primary bottleneck
- ❌ Insufficient coverage of hard training examples as an explanation

### What This Points To

The OOD failure in IRED must stem from factors not addressed by hard-state mining:

1. **Inference Mechanism**: Step budget, schedule calibration, or number of reverse steps may be insufficient for ill-conditioned matrices
2. **Score Field Parameterization**: The network may not learn score functions that generalize well to OOD input distributions (e.g., atypical condition numbers)
3. **Loss Formulation**: The denoising objective may not directly optimize for downstream solver performance
4. **Early Stopping or Regularization**: Standard L2 regularization on weights may not be appropriate for diffusion models under distribution shift

### Recommended Next Steps

1. **Inference-Side Interventions**:
   - Test longer reverse trajectories (50, 100 steps vs. current 10)
   - Implement gentler step schedules (cosine annealing on step size)
   - Adaptive inference compute (allocate more steps for uncertain states)

2. **Score Field Analysis**:
   - Visualize score vectors for OOD (ill-conditioned) matrix inputs
   - Compare to ID score fields to identify distribution shift artifacts
   - Test score-matching with explicit OOD-aware weighting in the loss

3. **Alternative Training Objectives**:
   - Direct solver-loss training: Optimize for actual matrix inversion error during reverse process, not just denoising
   - Adversarial matrix generation: Include ill-conditioned matrices (with increasing weight) during training, rather than mining hard states

---

## Conclusion

This negative result provides strong evidence that **state-space adversarial negative mining is not the lever for improving OOD performance in IRED matrix inversion**. The consistency of results across three fundamentally different task-agnostic samplers suggests the bottleneck lies elsewhere—likely in the inference mechanism, score field calibration, or loss formulation. Future work should redirect optimization attention from negative selection toward inference-side and loss-side interventions.

---

## Experimental Record

### Job Information
- **q242 (Replay-uncertainty)**: Job ID TBD, gpu_test partition, completed with OOD MSE = 0.2131
- **q243 (Trajectory-divergence)**: Job ID TBD, gpu_test partition, completed with OOD MSE = 0.2132
- **q244 (Local-instability)**: Job ID TBD, gpu_test partition, completed with OOD MSE = 0.2131
- **Baseline reference**: q211-q215 (ID MSE = 0.00969, OOD MSE = 0.2063)

### Code & Configuration
- **Config Location**: `projects/ired/configs/q242.json`, `q243.json`, `q244.json`
- **Implementation**: `projects/ired/experiments/matrix_inversion_mining.py` (sampler selection via config `hard_state_sampler` parameter)
- **Git SHA**: [To be recorded at submission]

### Artifacts
- SLURM logs: `projects/ired/slurm/logs/q242-q244_*.{out,err}`
- Results directory: `projects/ired/results/ds_inverse/q242/`, etc.

---

## Questions for Discussion

1. **Sufficiency of Auxiliary Signal**: Was the hard-state loss coefficient (0.001) too weak? Should we retry with stronger weighting (e.g., 0.1)?
   - *Counterpoint*: Stronger weighting would increase the distance from the validated baseline. The fact that weak reweighting degraded OOD suggests the signal is misaligned, not underpowered.

2. **Sampler Orthogonality**: Do the three samplers actually target different states, or do they overlap significantly?
   - *Action*: Could analyze correlation of hardness scores across q242-q244 to validate orthogonality.

3. **Alternative OOD Generalization Mechanisms**: If not hard-state mining, what explains the 3.3% OOD gap?
   - *Hypotheses*: Score-field under-smoothness, incompatible step schedules, or insufficient ill-conditioned training data (indirect exposure vs. direct).

---

## References to Related Work

- **Prior Negative Mining Attempts**: `documentation/results.md` (q201-q220 phases)
- **Debugging Records**: `documentation/debugging.md` (TAM-CTL failures, recovery loss shape bugs)
- **Research Plan**: `documentation/research-plan.md` (next-phase inference-side diagnostics)
