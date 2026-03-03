# Research Plan — IRED Matrix Inversion: Inference-First Improvement

## Project Status: INFERENCE-FIRST BRANCH

**Current Phase**: DESIGN → IMPLEMENT (inference-side interventions)
**Last Updated**: 2026-03-03

---

## Executive Summary

### Negative Mining Branch: CLOSED (2026-03-03)

After exhaustive investigation across 40+ experiments spanning local adversarial negatives, random mining, trajectory-anchor mining (TAM), recovery-based TAM-CTL, and three task-agnostic hard-state samplers (replay-uncertainty, trajectory-divergence, local-instability), **state-space adversarial negative mining is conclusively not the lever for improving OOD generalization in IRED matrix inversion**.

All task-agnostic samplers converged to the same OOD band (~0.2131–0.2132), which is 3.3% worse than baseline (0.2063). This uniformity across orthogonal sampler designs rules out sampler-specific failure and points to a fundamental misalignment between hard-state reweighting and the structure that matters for OOD.

Full negative result documentation: `documentation/negative-results-hard-state-sampling.md`

### Current Baseline Performance

| Metric | Value | Source |
|--------|-------|--------|
| ID validation MSE | 0.00969 | q211 baseline (8 seeds) |
| OOD MSE (ill-conditioned) | 0.2063 | q211 baseline |
| IRED paper same-difficulty MSE | 0.0096 | Table 2, 10 steps |
| IRED paper harder-difficulty MSE | 0.2063 | Table 2, 40 steps |

**Critical observation**: Our baseline (0.2063 at some step configuration) matches the IRED paper's best harder-difficulty result at 40 optimization steps. This demands a fidelity check before any new architecture work.

---

## Research Direction: Why Inference, Not Training

### Literature Grounding

The IRED paper (Du et al., ICML 2024, arXiv:2406.11179) is explicit about where gains come from on matrix inverse:

1. **Step count matters**: OOD MSE improves from 0.2110 (10 steps) → 0.2100 (20) → 0.2090 (30) → 0.2063 (40 steps). Same-difficulty MSE barely changes (0.0096 → 0.0095). The lever is inference compute on harder instances.

2. **Negatives are for landscape shaping, not hardness**: The paper's own negatives are simple (noise-corrupt the target, run two gradient steps). They explicitly found "more than 10 landscapes did not help." Negatives shape the energy surface; they are not a source of new hardness.

3. **Three explicit limitations point to inference-side improvements**:
   - "The inference time optimization procedure in IRED can still be improved because currently, it requires many steps of gradient descent to find an energy minima" → amortized initializers, guided optimizers
   - "Our sequence of annealed energy landscapes is defined through a sequence of added Gaussian noise increments — it would be further interesting to learn the sequence of energy landscapes to enable adaptive optimization" → learned schedules
   - "IRED in its current form does not leverage any additional memory" → stateful optimizers

4. **IRED was introduced to fix IREM's instability**: IREM required differentiating through the full optimization rollout (295% memory overhead, unstable training). IRED replaced this with denoising supervision + negative mining, gaining stability. Going back to train-through-optimization (IREM-style) would regress on this improvement.

### Path Independence Framework

The path independence literature (Anil et al., NeurIPS 2022, arXiv:2211.09961) provides the diagnostic framework:

- **Core finding**: Models only benefit from extra test-time iterations when their dynamics are sufficiently **path independent** (converge to the same fixed point regardless of initialization).
- **Necessary conditions**: Weight tying, input injection, randomized/adaptive depth at training time.
- **Diagnostic**: The **Asymptotic Alignment (AA) score** measures whether different initializations converge to the same attractor. High AA = path independent = benefits from more compute.
- **Per-example OOD diagnostic**: For a specific test input, if the solver converges to the same point from multiple initializations, trust the answer. If not, the model is likely wrong on that input.
- **Direct relevance to IRED**: IRED's annealed landscapes are a practical mitigation of path dependence (smooth-to-sharp to escape local minima), but there is no guarantee of path independence. If ill-conditioned matrices induce multiple local minima that the annealing doesn't resolve, more steps won't help—which is exactly what we observe.

### Improved CD Literature (Du et al., ICML 2021)

The improved-CD EBM paper's main robustness levers are **data augmentation** (as MCMC mode-mixing) and **multi-scale energy factorization**, not harder negatives. Key gaps relative to our implementation:

- **KL entropy term (L_ent)**: Maximizes diversity of MCMC samples via nearest-neighbor distance. Our Langevin chains may suffer from mode collapse that this addresses.
- **Data augmentation in Langevin chains**: Every 20 steps, apply random augmentation to help chains jump between modes. Not implemented in our codebase.
- **EMA (μ=0.99)**: Stabilizes training. Not explicit in current configs.
- **Backprop only through final Langevin step**: Memory-efficient, shown equivalent to full chain differentiation.

For matrix inversion, the clean analogue of these robustness levers is a **condition-number curriculum** (expose the model to progressively harder conditioning during training) and **coarse-to-fine energy factorization** (e.g., over spectral summaries of the matrix).

---

## Priority-Ordered Experiment Ladder

### Priority 1: Paper-Fidelity / Step-Semantics Check (q245)

**Rationale**: Our baseline OOD MSE (0.2063) exactly matches the IRED paper's best harder-difficulty result at 40 optimization steps. Before changing architecture, verify whether our "10-step reverse diffusion" actually corresponds to the paper's 10 total optimization steps or to something closer to their 40-step regime. If there is a fidelity gap, we may be debugging implementation rather than modeling.

**Experiment q245**: Reproduce IRED Table 2 step-count ladder
- Run inference at 10, 20, 30, 40 optimization steps on the existing trained baseline model
- Compare to paper's Table 2: {10: 0.2110, 20: 0.2100, 30: 0.2090, 40: 0.2063}
- Also reproduce Table 3 ablation (gradient descent alone vs. + refinement vs. + contrastive)
- **No retraining required**—just vary inference-time step count on existing checkpoint

**Hypothesis**: If our 10-step inference gives 0.2063 (matching paper's 40-step), we are already in the paper's best regime and the step semantics differ. If our 10-step gives ~0.2110, we match the paper and have room for step-count improvement.

**Success criteria**: Clear mapping between our step semantics and the paper's. Quantified headroom (if any) from step-count increases.

**Compute**: Minimal—inference only, no training. Single GPU, <1 hour.

---

### Priority 2: Inference Calibration (q246–q248)

**Rationale**: The IRED paper's central claim is that IRED generalizes to harder instances by using more computation at test time on a learned energy landscape. The first lever to pull after verifying step semantics is the inference loop itself.

**Experiment q246**: Step-count sweep on OOD
- Sweep per-landscape step count: {1, 2, 5, 10, 20, 50} steps per landscape × 10 landscapes
- Total optimization steps: {10, 20, 50, 100, 200, 500}
- Evaluate both ID and OOD MSE at each budget
- Plot the compute-performance frontier

**Experiment q247**: Adaptive step size (λ_k)
- Replace fixed step size with per-step adaptive λ:
  - Backtracking line search: halve λ if energy increases
  - Trust-region style: expand λ when energy decreases consistently, shrink when it doesn't
  - Momentum (β=0.9 Polyak-style): carry previous gradient direction
- Evaluate on same ID/OOD split

**Experiment q248**: Multi-start inference
- For each test input, run inference from N different random initializations (N ∈ {5, 10, 20})
- Select the solution with lowest final energy
- This directly tests path independence: if results vary widely across starts, the landscape has problematic local minima
- **Dual purpose**: Both a diagnostic (measures path dependence) and a potential OOD improvement

**Hypothesis**: If the energy landscape has multiple local minima for ill-conditioned matrices, multi-start will show high variance across initializations (path dependence) and the best-of-N selection will improve OOD. If the landscape is well-shaped, multi-start won't help but confirms path independence.

**Success criteria**:
- q246: Identify optimal step budget and whether more steps help on OOD
- q247: >5% OOD improvement from adaptive stepping
- q248: Variance across initializations quantifies path dependence; best-of-N improves OOD

**Compute**: q246 ~2h (inference sweeps), q247 ~4h (need to implement adaptive stepping), q248 ~2h (embarrassingly parallel).

---

### Priority 3: Learned Annealing Schedule (q249–q250)

**Rationale**: The IRED paper explicitly calls out the fixed Gaussian annealing schedule as a limitation: "it would be further interesting to learn the sequence of energy landscapes to enable adaptive optimization." If ill-conditioned matrices need a different annealing path, fixed noise increments are exactly the wrong place to stay rigid.

**Experiment q249**: Hardness-conditioned σ_k schedule
- Train a small network π(x) → {σ_1, ..., σ_K} that predicts the noise schedule for a given input x
- The schedule network is trained jointly with the energy model using the denoising objective
- At inference, ill-conditioned inputs may receive longer/gentler annealing paths automatically
- **Key insight**: This is a lightweight change—the energy model architecture is unchanged, only the schedule selection varies

**Experiment q250**: Learned λ_k policy
- Instead of fixed step size, train a policy λ(x, t, ∇E) → step size at each step
- Similar to adaptive compute in Universal Transformers (Dehghani et al., ICLR 2019)
- Can be trained with REINFORCE or straight-through estimation using the denoising loss as reward

**Hypothesis**: Ill-conditioned matrices benefit from longer annealing (more time in smooth landscape before transitioning to sharp), which fixed schedules cannot provide. A learned schedule should allocate more compute to harder inputs.

**Success criteria**: Learned schedule produces measurably different σ_k sequences for well-conditioned vs. ill-conditioned inputs. OOD MSE improves >5%.

**Compute**: ~8h training per variant (need to train schedule network jointly).

**Literature grounding**: Universal Transformers showed that adaptive computation time (halting when state stabilizes) enables variable-depth inference. The same principle applies here—allocate more annealing steps to harder inputs.

---

### Priority 4: Amortized Initializer / Guided Optimizer (q251–q252)

**Rationale**: The IRED paper explicitly suggests "an amortized neural network generator for generating initial solutions or guided optimizers can speed up this procedure." Starting from Gaussian noise is maximally uninformative. For matrix inversion, even a crude initial estimate (e.g., A^T / ‖A‖² or a learned preconditioner) places the solver much closer to the correct basin.

**Experiment q251**: Amortized initializer
- Train a small feedforward network g(A) → y_0 that predicts a rough inverse
- Use y_0 as the initialization for IRED inference instead of Gaussian noise
- The initializer is trained with simple MSE on the training set (no energy model needed)
- IRED then refines from y_0—the energy model acts as verifier/refiner

**Experiment q252**: Guided optimizer
- Replace plain gradient descent with a learned update rule:
  - y_{t+1} = y_t - f_θ(y_t, ∇E(y_t), t) instead of y_{t+1} = y_t - λ∇E(y_t)
  - f_θ is a small MLP that sees the current state, gradient, and step index
- This is a "learning to optimize" approach (Andrychowicz et al., NeurIPS 2016)
- The guided optimizer can learn to handle ill-conditioned landscapes (e.g., implicit preconditioning)

**Hypothesis**: Starting closer to the correct basin removes the burden of navigating from random noise through potentially path-dependent landscape regions. For ill-conditioned matrices, the initializer provides an OOD-aware warm start that the fixed-schedule IRED cannot.

**Success criteria**: >10% OOD improvement from amortized initialization. Guided optimizer converges in fewer steps than vanilla gradient descent.

**Compute**: q251 ~4h (train initializer + evaluate), q252 ~8h (joint training of optimizer).

---

### Priority 5: Memoryful Optimizer (q253)

**Rationale**: The IRED paper states "IRED in its current form does not leverage any additional memory. Therefore, for tasks that would benefit from explicitly using additional memory to store intermediate results... IRED might not be as effective." Matrix inversion of ill-conditioned systems is precisely such a task: memoryless gradient descent struggles when the landscape is stiff (high condition number → disparate eigenvalue scales → gradient oscillation).

**Experiment q253**: Stateful optimizer
- Replace memoryless gradient descent with an optimizer that carries state:
  - **Variant A**: Momentum (simple—carry exponential moving average of past gradients)
  - **Variant B**: Residual features (carry (y_t - y_{t-1}) as additional input to the energy function)
  - **Variant C**: Small recurrent hidden state h_t updated at each step alongside y_t
- The hidden state is reset at the start of each inference, so it's per-problem memory, not cross-problem

**Hypothesis**: Ill-conditioned matrices create stiff optimization landscapes where memoryless descent oscillates. Momentum or adaptive state allows the optimizer to implicitly precondition, damping oscillations along high-curvature directions.

**Success criteria**: Stateful optimizer shows faster convergence (fewer steps to same MSE) and better OOD performance than memoryless descent.

**Compute**: ~8h per variant (need to modify inference loop and possibly retrain with stateful dynamics).

---

### Priority 6: Path-Independence Measurement + Regularization (q254–q255)

**Rationale**: The path independence framework (Anil et al., NeurIPS 2022) provides both a diagnostic and a training intervention. Before adding more architectural complexity, measure whether our current model is path-independent on OOD inputs. If it isn't, that directly explains why more steps and harder states don't help.

**Experiment q254**: Path independence diagnostic (AA score)
- For each test input (both ID and OOD), run IRED inference from N=20 different random initializations
- Measure the Asymptotic Alignment (AA) score: cosine similarity between final states from different initializations
- Compute per-example and aggregate AA scores, stratified by condition number
- **Prediction**: ID inputs will show high AA (path independent), OOD ill-conditioned inputs will show low AA (path dependent)

**Experiment q255**: Path-independence regularization
- Add a training-time regularization that encourages path independence:
  - During training, for each (x, y) pair, run two forward passes from different noise initializations
  - Add a consistency loss: ‖y_final^{init_1} - y_final^{init_2}‖²
  - This directly penalizes path dependence
- Also test randomized depth during training (vary number of denoising steps per example)

**Hypothesis**: OOD failure correlates with low AA score (path dependence). Path-independence regularization improves OOD by ensuring the solver converges to the same answer regardless of initialization.

**Success criteria**:
- q254: Strong negative correlation between AA score and test MSE on OOD inputs
- q255: Path-independence regularization improves OOD MSE and raises AA scores

**Compute**: q254 ~2h (inference only), q255 ~8h (retraining with consistency loss).

**Key architectural implications from the literature**:
- Weight tying (same energy network at each step): Already present in IRED ✓
- Input injection (re-inject x at each step): Need to verify—check if the energy function receives the input matrix A at each optimization step, not just at initialization ⚠️
- Randomized depth at training: Not currently implemented—this is a training intervention that promotes path independence

---

### Priority 7: Condition-Number Curriculum (q256)

**Rationale**: If the training-side lever is not negative mining, the improved-CD literature (Du et al., ICML 2021) points to **data augmentation** and **multi-scale processing** as robustness levers. For matrix inversion, the clean analogue is a condition-number curriculum: expose the model to progressively harder conditioning during training.

**Experiment q256**: Condition-number curriculum
- Phase 1 (0–50K steps): Train on well-conditioned matrices only (condition number < 100)
- Phase 2 (50K–75K steps): Mix in moderately ill-conditioned matrices (condition number 100–1000)
- Phase 3 (75K–100K steps): Include highly ill-conditioned matrices (condition number > 1000)
- Compare to: (a) uniform random conditioning throughout, (b) baseline (well-conditioned only)

**Hypothesis**: Progressive conditioning exposure teaches the energy function to handle stiffer landscapes gradually, rather than forcing it to learn from a single conditioning regime. This is analogous to the improved-CD paper's data augmentation for MCMC mode mixing.

**Success criteria**: OOD MSE improves >10% vs. baseline. Training stability maintained (no divergence during phase transitions).

**Compute**: ~4h (single training run with curriculum scheduler).

**Note**: This is the only training-side intervention on the ladder. It is grounded in the improved-CD EBM literature's robustness principles (augmentation and multi-scale), not in negative mining.

---

### Priority 8: Task-Specific Energy Composition (q257)

**Rationale**: The IRED paper states "one may also add additional inference-time constraints by composing the learned IRED energy function with other energy functions." For matrix inversion, we can compose the learned energy with an analytic residual term. This is no longer task-agnostic, but it's the most likely near-term win if the goal is raw OOD performance.

**Experiment q257**: Composed energy with inverse-consistency penalty
- Define auxiliary energy: E_aux(A, Y) = ‖AY - I‖² (residual of the inverse equation)
- At inference, minimize E_total(A, Y) = E_IRED(A, Y) + α · E_aux(A, Y)
- Sweep α ∈ {0.01, 0.1, 1.0, 10.0}
- **No retraining**—this is purely an inference-time modification

**Hypothesis**: The analytic residual provides a global guidance signal that the learned energy landscape may lack for OOD inputs. Even if E_IRED has local minima for ill-conditioned matrices, E_aux is globally convex and pulls toward the correct inverse.

**Success criteria**: Substantial OOD improvement (>20%). The composed energy should dramatically reduce the gap between ID and OOD performance.

**Compute**: Minimal—inference only, <1h.

**Caveat**: This is task-specific and would not generalize to other IRED applications. It is appropriate if the research goal is understanding IRED's failure mode on matrix inversion, but not if the goal is general-purpose IRED improvement.

---

## Anti-Recommendations (What NOT to Do)

### Do NOT return to IREM-style train-through-optimization
IRED was introduced specifically to avoid IREM's instability (differentiating through the full optimization rollout has 295% memory overhead and unstable training). Going back would regress on IRED's core contribution. The IREM paper itself notes: "Direct recurrent backpropagation through such a number of iterative computations has been proven to be unstable to train."

### Do NOT pursue more negative mining variants
The negative mining branch is closed. Local adversarial, random, TAM, TAM-CTL, replay-uncertainty, trajectory-divergence, and local-instability all failed. The failure is sampler-agnostic, indicating the mechanism is wrong, not the specific sampler.

### Do NOT increase hard-state loss coefficient
The fact that even weak reweighting (L_hard ~ 2×10⁻⁴) degraded OOD suggests the signal is misaligned, not underpowered. Stronger coefficients would likely cause larger OOD degradation.

---

## Compressed Priority Order

| # | Experiment | Type | Key Question | Compute |
|---|-----------|------|-------------|---------|
| 1 | q245: Paper fidelity check | Diagnostic | Do our step semantics match the paper? | <1h |
| 2 | q246–q248: Inference calibration | Inference | More steps / adaptive λ / multi-start? | ~8h |
| 3 | q249–q250: Learned schedule | Training + Inference | Can we learn input-dependent annealing? | ~16h |
| 4 | q251–q252: Amortized init / guided opt | Architecture | Does warm start fix OOD? | ~12h |
| 5 | q253: Memoryful optimizer | Architecture | Does state help stiff landscapes? | ~24h |
| 6 | q254–q255: Path independence | Diagnostic + Training | Is path dependence the bottleneck? | ~10h |
| 7 | q256: Conditioning curriculum | Training | Does progressive exposure help? | ~4h |
| 8 | q257: Energy composition | Inference (task-specific) | Does analytic residual fix OOD? | <1h |

**Total estimated compute**: ~76 GPU-hours across all experiments

**Recommended parallelism**:
- q245 first (blocks everything—need to understand step semantics)
- q246–q248 can run in parallel after q245
- q254 (path independence diagnostic) can run in parallel with q246–q248
- q257 (energy composition) can run anytime (inference-only, no dependencies)

---

## Key References

### Primary Papers

1. **IRED**: Du et al., "Learning Iterative Reasoning through Energy Diffusion," ICML 2024. [arXiv:2406.11179](https://arxiv.org/abs/2406.11179)
   - Table 2: Step-count scaling on matrix inverse
   - Table 3: Ablation (gradient descent + refinement + contrastive)
   - Section 5: Three explicit limitations (inference speed, fixed schedule, no memory)

2. **Path Independence**: Anil et al., "Path Independent Equilibrium Models Can Better Exploit Test-Time Computation," NeurIPS 2022. [arXiv:2211.09961](https://arxiv.org/abs/2211.09961)
   - AA score diagnostic for path independence
   - Conditions: weight tying, input injection, randomized depth
   - Per-example OOD prediction via convergence consistency

3. **Improved CD**: Du et al., "Improved Contrastive Divergence Training of Energy-Based Models," ICML 2021. [arXiv:2012.01316](https://arxiv.org/abs/2012.01316)
   - Data augmentation as MCMC mode-mixing (not harder negatives)
   - Multi-scale energy factorization
   - KL entropy term for sample diversity

4. **IREM**: Du et al., "Learning Iterative Reasoning through Energy Minimization," ICML 2022. [arXiv:2206.15448](https://arxiv.org/abs/2206.15448)
   - Predecessor to IRED; replaced due to instability
   - 295% memory overhead from full backprop
   - Anti-recommendation: do not regress to IREM-style training

### Supporting Literature

5. **Deep Equilibrium Models**: Bai, Kolter, Koltun, NeurIPS 2019. [arXiv:1909.01377](https://arxiv.org/abs/1909.01377)
   - Fixed-point iteration as implicit infinite-depth networks

6. **Universal Transformers**: Dehghani et al., ICLR 2019. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)
   - Adaptive computation time with weight-tied transformer blocks

7. **Lyapunov-Stable DEQ**: AAAI 2024. [arXiv:2304.12707](https://arxiv.org/abs/2304.12707)
   - Stability guarantees via Lyapunov functions (analogous to convex energy)

8. **Learning to Optimize**: Andrychowicz et al., NeurIPS 2016.
   - Learned update rules for optimization (relevant to q252 guided optimizer)

---

## Historical Record

### Completed Phases (Negative Mining Branch — CLOSED)

#### Phase 0: Single-Run Pilot (2026-01-21)
- q001–q003: Baseline vs. random vs. adversarial mining
- Finding: Adversarial mining degrades by ~1.5%

#### Phase 1: Multi-Seed Validation (2026-01-24)
- q101–q103: 10 seeds each, statistical testing
- Finding: Baseline optimal (p<0.0001 vs. adversarial)

#### Scalar Energy Head Investigation (2026-02-19)
- q207, q209, q210: Scalar head instability, mining as accidental regularizer
- Finding: Energy scale problem, not mining benefit

#### TAM and TAM-CTL (2026-02-20 to 2026-02-24)
- q211–q225: Trajectory-anchor mining, recovery loss, hyperparameter sweeps
- Finding: TAM marginal at best, recovery loss degraded OOD
- Bugs fixed: DataLoader freeze, recovery config, shape bug, checkpoint path

#### Task-Agnostic Hard-State Sampling (2026-03-01 to 2026-03-03)
- q242–q244: Replay-uncertainty, trajectory-divergence, local-instability
- Finding: All converge to same OOD band (~0.2131), 3.3% worse than baseline
- **Conclusion**: Negative mining is not the lever. Branch closed.

---

## Document History

- **2026-01-21**: Document created after Phase 0 results
- **2026-02-19**: Added scalar energy head investigation
- **2026-03-03**: Major rewrite — closed negative mining branch, redirected to inference-first ladder grounded in IRED paper limitations, path independence framework, and improved-CD literature
