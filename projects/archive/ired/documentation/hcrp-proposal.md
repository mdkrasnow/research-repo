# Improving Out-of-Distribution Generalization in Energy-Based Iterative Reasoning through Inference-Time Optimization

## Research Proposal — Harvard College Research Program (HCRP), Summer 2026

---

## Objective, Significance, and Implications

Machine learning systems that can reason—solving novel problems by iteratively refining candidate solutions—represent a fundamental frontier in artificial intelligence. Unlike standard supervised models that produce outputs in a single forward pass, iterative reasoning systems allocate variable computation depending on problem difficulty, enabling them to generalize to harder instances than those seen during training. This capacity for *adaptive test-time computation* is increasingly recognized as a key ingredient for robust AI systems across scientific computing, robotics, and mathematical problem-solving.

This project investigates how to improve the out-of-distribution (OOD) generalization of IRED (Iterative Reasoning through Energy Diffusion), a state-of-the-art framework for learning iterative reasoning introduced by Du, Mao, and Tenenbaum at ICML 2024. IRED trains energy-based models (EBMs) to define optimization landscapes over candidate solutions, then solves problems at test time by performing gradient descent on these learned energy surfaces through a sequence of progressively sharpened (annealed) landscapes. IRED significantly outperforms prior methods on continuous reasoning tasks including matrix inversion, path planning, and physics simulation, and can generalize to more complex problem instances by simply running more optimization steps at inference.

However, IRED's OOD generalization has clear limits. On the matrix inversion benchmark—inverting random 20x20 matrices—IRED achieves excellent in-distribution performance (MSE = 0.0097) but degrades substantially on ill-conditioned matrices outside the training distribution (OOD MSE = 0.2063, a 21x increase). The IRED paper itself identifies three explicit limitations that all point to the inference mechanism as the primary bottleneck: (1) the optimization procedure "requires many steps of gradient descent to find an energy minima," suggesting amortized initializers or guided optimizers; (2) the annealing schedule is fixed rather than input-adaptive, suggesting learned schedules; and (3) the optimizer carries no memory across steps, suggesting stateful optimization for stiff landscapes.

My preliminary research, conducted over the past two months under Prof. Du's supervision, has systematically established that the bottleneck is *not* on the training side. Through over 40 controlled experiments spanning local adversarial negatives, random mining, trajectory-anchor mining (TAM), recovery-based convergence training (TAM-CTL), and three task-agnostic hard-state samplers (replay-uncertainty, trajectory-divergence, local-instability), I have conclusively demonstrated that state-space adversarial negative mining does not improve OOD generalization. All task-agnostic samplers converged to the same OOD error band (~0.2131, 3.3% *worse* than baseline), ruling out sampler-specific failure and pointing to a fundamental misalignment between hard-state reweighting and the structure that matters for OOD. This comprehensive negative result, documented across statistically rigorous multi-seed experiments (8-10 seeds per condition, p < 0.0001 for key comparisons), redirects the investigation squarely toward inference-time interventions—the direction the IRED paper itself anticipated.

The significance of this work is threefold. First, improving IRED's OOD generalization would demonstrate that learned energy landscapes can serve as reliable "verifiers" for iterative computation, a paradigm with broad implications for scientific computing where test-time problems routinely exceed training-distribution difficulty. Second, the path-independence diagnostic framework I will develop (measuring whether different initializations converge to the same solution) provides a principled, per-example metric for predicting when an iterative solver will fail on OOD inputs—a capability with direct practical value. Third, this research addresses an open question explicitly posed in the IRED paper: whether learned annealing schedules and amortized initializers can close the gap between in-distribution and out-of-distribution performance, contributing to the broader understanding of adaptive test-time computation in neural networks.

---

## Detailed Research Plan

### Research Design Overview

This project follows a systematic experimental ladder, progressing from low-cost diagnostics through increasingly ambitious architectural interventions. Each experiment builds on the previous result, with clear go/no-go criteria determining whether to proceed to the next stage. All experiments use the matrix inversion task (20x20 random matrices) as the primary benchmark, with a well-established baseline (ID MSE = 0.0097, OOD MSE = 0.2063) against which improvements are measured.

The computational infrastructure is already operational: I have a fully functional experimental pipeline on the Harvard FASRC cluster (A100 GPUs via SLURM), automated experiment submission with git-based reproducibility (each job checks out the exact commit used at submission time), and a structured results-tracking system that has managed over 60 experiments to date.

### Phase 1: Diagnostics and Calibration (Weeks 1–2)

**Experiment 1: Step-semantics fidelity check.** Before modifying the inference mechanism, I must verify that our implementation's "10 reverse diffusion steps" corresponds to the same computational budget as the IRED paper's reported step counts. Our baseline OOD MSE (0.2063) exactly matches the paper's best result at 40 optimization steps (Table 2), raising the possibility that our step semantics already correspond to the paper's most-compute regime. I will run inference at 10, 20, 30, and 40 total optimization steps on the existing trained model and compare to the paper's Table 2 ladder (10 steps: 0.2110, 20: 0.2100, 30: 0.2090, 40: 0.2063). This requires no retraining—only inference on existing checkpoints—and takes less than 1 GPU-hour.

**Experiment 2: Path-independence diagnostic.** For each test input (both in-distribution and OOD), I will run IRED inference from N=20 different random initializations and measure the Asymptotic Alignment (AA) score—the cosine similarity between final solutions from different starting points. Following the framework of Anil et al. (2022), high AA indicates path independence (the solver converges to a unique solution regardless of initialization), while low AA indicates problematic local minima. I predict that in-distribution inputs will show high AA while ill-conditioned OOD inputs will show low AA, directly explaining why additional optimization steps fail to help on harder instances. This diagnostic is inference-only (~2 GPU-hours) and provides the theoretical grounding for all subsequent interventions.

**Experiment 3: Compute-performance frontier.** I will sweep per-landscape step counts ({1, 2, 5, 10, 20, 50} steps per landscape across 10 landscapes) and plot both ID and OOD MSE as a function of total compute budget. This establishes the baseline scaling behavior and identifies whether more computation helps at all on OOD, or whether the landscape has fundamental path-dependence issues that prevent convergence.

### Phase 2: Inference-Time Interventions (Weeks 3–5)

**Experiment 4: Multi-start inference with best-of-N selection.** For each test input, I will run inference from N different random initializations (N in {5, 10, 20}) and select the solution with the lowest final energy. This serves dual purposes: as a diagnostic (if results vary widely across starts, the landscape has local minima) and as a practical intervention (best-of-N selection can improve OOD by escaping bad local minima). Success criterion: >5% OOD improvement and quantified path-dependence reduction.

**Experiment 5: Adaptive step-size optimization.** I will replace the fixed gradient descent step size with three adaptive variants: (a) backtracking line search (halve step size if energy increases), (b) trust-region style (expand when energy decreases consistently, shrink otherwise), and (c) Polyak-style momentum (beta=0.9, carrying exponential moving average of past gradients). Ill-conditioned matrices create stiff optimization landscapes where memoryless descent oscillates; momentum implicitly preconditions by damping high-curvature directions.

**Experiment 6: Composed energy with analytic residual.** At inference time, I will augment the learned energy with an analytic inverse-consistency penalty: E_total(A, Y) = E_IRED(A, Y) + alpha * ||AY - I||^2, sweeping alpha in {0.01, 0.1, 1.0, 10.0}. The analytic term is globally convex and provides gradient signal even where the learned energy has poor local structure. This requires no retraining and directly tests whether the learned landscape's OOD failure can be compensated by domain knowledge. While task-specific, this experiment isolates whether the bottleneck is in the energy landscape shape (fixable by training interventions) or in the optimizer's ability to navigate it (fixable by inference interventions).

### Phase 3: Learned Inference Components (Weeks 6–8)

Based on Phase 2 results, I will pursue the most promising direction among:

**Experiment 7: Hardness-conditioned annealing schedule.** Train a small network pi(x) that predicts a per-input noise schedule {sigma_1, ..., sigma_K}, allowing ill-conditioned matrices to receive longer, gentler annealing paths. The schedule network is trained jointly with the energy model using the standard denoising objective. This directly addresses the IRED paper's second stated limitation ("it would be further interesting to learn the sequence of energy landscapes").

**Experiment 8: Amortized initializer.** Train a feedforward network g(A) -> y_0 that predicts a rough initial inverse, replacing Gaussian noise initialization. The initializer is trained with simple MSE on the training set; IRED then refines from y_0, acting as a verifier/refiner. Starting closer to the correct basin removes the burden of navigating path-dependent landscape regions. This addresses the IRED paper's first limitation ("an amortized neural network generator for generating initial solutions").

**Experiment 9: Learned optimizer.** Replace plain gradient descent with a learned update rule y_{t+1} = y_t - f_theta(y_t, grad_E(y_t), t), where f_theta is a small MLP trained to optimize on the energy landscape. Following Andrychowicz et al. (2016), the learned optimizer can discover implicit preconditioning strategies that handle ill-conditioned landscapes. This addresses the IRED paper's third limitation ("does not leverage any additional memory").

### Phase 4: Path-Independence Regularization and Synthesis (Weeks 9–10)

**Experiment 10: Consistency regularization.** Add a training-time loss that encourages path independence: for each training pair (x, y), run two forward passes from different noise initializations and penalize ||y_final_1 - y_final_2||^2. Combined with randomized depth (varying the number of denoising steps per example during training), this directly promotes the conditions identified by Anil et al. (2022) as necessary for test-time compute scaling.

**Synthesis and writing.** Integrate the most successful interventions, run final multi-seed evaluations (8 seeds per condition) for statistical rigor, and prepare results for publication. The combination of comprehensive negative results (training-side mining) and positive results (inference-side interventions) constitutes a complete research narrative suitable for a workshop or conference paper.

### Methodology

All experiments follow the same rigorous protocol established in my preliminary work:

- **Statistical rigor**: 8-10 independent seeds per condition, with paired t-tests and reported effect sizes for all comparisons.
- **Reproducibility**: Every experiment is tracked with git commit SHA, JSON configuration files, SLURM job IDs, and automated result persistence. The experimental pipeline supports exact reproduction of any prior result.
- **Compute budget**: Estimated total of ~120 GPU-hours across all experiments (approximately 15 GPU-hours/week), well within FASRC allocation limits. All computation uses NVIDIA A100 GPUs on the Harvard FASRC cluster.
- **Incremental validation**: Each new technique is first tested with a 10K-step probe (single seed, ~30 minutes) before committing to a full 100K-step multi-seed run, catching implementation errors early and conserving compute.

---

## Time Frame

| Week | Phase | Activities | Deliverables |
|------|-------|-----------|-------------|
| 1 | Diagnostics | Step-semantics check; path-independence diagnostic | Verified step mapping; AA scores by condition number |
| 2 | Diagnostics | Compute-performance frontier; analyze Phase 1 | Scaling plots; go/no-go for Phase 2 directions |
| 3 | Inference | Multi-start inference; adaptive step size | OOD improvement quantification; path-dependence metrics |
| 4 | Inference | Composed energy; momentum optimizer | Best inference-only intervention identified |
| 5 | Inference | Ablation studies on best Phase 2 method | Robust multi-seed results for top intervention |
| 6 | Learned | Learned annealing schedule (joint training) | Trained schedule network; per-input schedule analysis |
| 7 | Learned | Amortized initializer or guided optimizer | Trained component; convergence speed comparison |
| 8 | Learned | Combine best learned component with Phase 2 winner | Combined system evaluation |
| 9 | Synthesis | Path-independence regularization; final evaluations | Complete multi-seed results across all methods |
| 10 | Writing | Synthesize results; prepare figures and manuscript | Draft workshop/conference paper; documented codebase |

---

## Faculty Involvement

This project is supervised by Professor Yilun Du, Assistant Professor of Computer Science at the Harvard John A. Paulson School of Engineering and Applied Sciences (SEAS) and Investigator at the Kempner Institute for the Study of Natural and Artificial Intelligence. Prof. Du is the lead author of the IRED framework that this project extends, as well as its predecessor IREM, making him uniquely qualified to guide this research. His expertise in energy-based models, iterative reasoning, and generative AI for robotic systems provides the theoretical foundation and methodological guidance for this work.

Prof. Du and I will meet weekly to review experimental results, discuss theoretical implications, and plan next steps. Prof. Du will provide guidance on: (1) interpreting diagnostic results in the context of the broader energy-based model literature, (2) architectural decisions for learned inference components (annealing schedules, amortized initializers), and (3) positioning findings relative to the growing literature on test-time computation scaling. As the creator of both IRED and its codebase, Prof. Du can provide insight into implementation details and design decisions that are not fully captured in the published paper—context that is especially valuable for the step-semantics fidelity check and for understanding the interaction between training objectives and inference-time behavior.

My preliminary work over the past two months has established a strong collaborative foundation: I have independently built the experimental infrastructure, executed over 40 experiments, discovered and documented a comprehensive negative result on training-side interventions, and formulated the inference-first research direction based on careful reading of the IRED paper's stated limitations and the path-independence literature. This summer's work will transition from exploration to targeted experimentation on the most promising directions, with the goal of producing publishable results that advance our understanding of adaptive test-time computation in energy-based reasoning systems.

---

## Works Cited

Andrychowicz, M., Denil, M., Gomez, S., Hoffman, M. W., Pfau, D., Schaul, T., Shillingford, B., & de Freitas, N. (2016). Learning to learn by gradient descent by gradient descent. In *Advances in Neural Information Processing Systems* (NeurIPS), 29, 3981–3989.

Anil, C., Pokle, A., Liang, K., Treutlein, J., Wu, Y., Bai, S., Kolter, J. Z., & Grosse, R. B. (2022). Path independent equilibrium models can better exploit test-time computation. In *Advances in Neural Information Processing Systems* (NeurIPS), 35.

Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep equilibrium models. In *Advances in Neural Information Processing Systems* (NeurIPS), 32, 688–699.

Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, L. (2019). Universal Transformers. In *Proceedings of the International Conference on Learning Representations* (ICLR).

Du, Y., Li, S., Tenenbaum, J. B., & Mordatch, I. (2021). Improved contrastive divergence training of energy-based models. In *Proceedings of the 38th International Conference on Machine Learning* (ICML), PMLR 139, 2837–2848.

Du, Y., Li, S., Tenenbaum, J. B., & Mordatch, I. (2022). Learning iterative reasoning through energy minimization. In *Proceedings of the 39th International Conference on Machine Learning* (ICML), PMLR 162, 5570–5582.

Du, Y., Mao, J., & Tenenbaum, J. B. (2024). Learning iterative reasoning through energy diffusion. In *Proceedings of the 41st International Conference on Machine Learning* (ICML), PMLR 235.

Graves, A. (2016). Adaptive computation time for recurrent neural networks. *arXiv preprint arXiv:1603.08983*.
