# Implementation TODO: IRED Energy Interpretability Project

## SETUP Phase

### Environment & Dependencies
- [ ] S1: Create Python virtual environment (3.9+)
- [ ] S2: Install PyTorch (2.0+, CUDA 12.x support)
- [ ] S3: Install core dependencies: numpy, scipy, scikit-learn, matplotlib, plotly
- [ ] S4: Install mechanistic interpretability packages: einops, wandb
- [ ] S5: Validate IRED codebase imports (diffusion_lib, dataset, models)

### Dataset Preparation
- [ ] S6: Load matrix inversion dataset from IRED (Inverse class, n=20x20)
- [ ] S7: Create dataset statistics notebook (conditioning, rank, norm distributions)
- [ ] S8: Prepare train/val/test splits (500/100/100 examples)
- [ ] S9: Implement data augmentation (noise, scaling variations)

### Baseline Model Training
- [ ] S10: Train baseline IRED model on matrix inversion (10 diffusion steps)
- [ ] S11: Validate model convergence and MSE performance
- [ ] S12: Save trained model checkpoint
- [ ] S13: Create inference pipeline for arbitrary matrices

### Analysis Infrastructure
- [ ] S14: Create `analysis/` submodule for interpretability tools
- [ ] S15: Implement energy landscape slicing utilities
- [ ] S16: Implement gradient computation wrapper (efficient batching)
- [ ] S17: Implement Hessian computation (finite differences + autograd)
- [ ] S18: Create visualization utilities (matplotlib + plotly integration)
- [ ] S19: Implement data logging/tracking (wandb integration)
- [ ] S20: Create test suite for analysis tools

### Documentation & Planning
- [ ] S21: Write Phase 1 detailed implementation plan
- [ ] S22: Create experimental design doc (matrix sampling strategy)
- [ ] S23: Set up experiment tracking spreadsheet
- [ ] S24: Prepare first set of SLURM job templates

---

## Phase 1: Energy Landscape Characterization

### Landscape Visualization
- [ ] P1-1: Implement 2D slicing along PCA directions
- [ ] P1-2: Generate energy heatmaps with filter normalization (cite: Goldstein et al.)
- [ ] P1-3: Create contour plots with minimum locations marked
- [ ] P1-4: Visualize ground truth location + prediction location on heatmap
- [ ] P1-5: Generate 50 slices from different random directions, document variation

### Gradient Field Analysis
- [ ] P1-6: Sample 1000 random predictions in valid range
- [ ] P1-7: Compute ∇E at each point (batched computation)
- [ ] P1-8: Create vector field visualization overlaid on energy landscape
- [ ] P1-9: Measure gradient field alignment with matrix properties:
  - [ ] Correlation(∇E, conditioning_number)
  - [ ] Correlation(∇E, frobenius_norm)
  - [ ] Correlation(∇E, largest_eigenvalue)
- [ ] P1-10: Create gradient magnitude distribution plots

### Hessian Analysis
- [ ] P1-11: Implement efficient Hessian computation (sample 100 points)
- [ ] P1-12: Compute eigenvalue decomposition of H at each point
- [ ] P1-13: Plot eigenvalue spectra (log-scale for large spreads)
- [ ] P1-14: Measure conditioning number of Hessian (curvature severity)
- [ ] P1-15: Identify most/least curved regions of landscape
- [ ] P1-16: Test hypothesis: minima are sharper than basin edges

### Quantitative Landscape Characterization
- [ ] P1-17: Compute convexity score (ratio of convex regions)
- [ ] P1-18: Measure basin sharpness (gradient magnitude at minimum)
- [ ] P1-19: Count spurious local minima (critical points with E > E_min + threshold)
- [ ] P1-20: Measure energy gap (max - min normalized by scale)
- [ ] P1-21: Compare across multiple random seeds (measure robustness)

### Visualization Suite
- [ ] P1-22: Create master figures for paper/presentation:
  - [ ] Figure 1a: Energy landscape slices (grid of 6 different angles)
  - [ ] Figure 1b: Gradient field overlay
  - [ ] Figure 1c: Hessian eigenvalue spectra
  - [ ] Figure 1d: Quantitative metrics summary table
- [ ] P1-23: Create interactive 3D visualization (plotly)
- [ ] P1-24: Document methodology (what do these plots mean?)

### Validation & Documentation
- [ ] P1-25: Write Phase 1 results document (findings, surprises, implications)
- [ ] P1-26: Create reproducibility notes (seeds, hyperparameters)
- [ ] P1-27: Identify next-phase questions based on Phase 1 results

---

## Phase 2: Sparse Autoencoder Decomposition

### Data Collection
- [ ] P2-1: Extract hidden activations from EBM forward pass
  - [ ] After fc1, fc2, fc3, and final energy output
- [ ] P2-2: Collect activations for 5000 (input, output, noise_level) examples
- [ ] P2-3: Compute activation statistics (mean, std, max per layer)
- [ ] P2-4: Analyze activation distribution (normal? heavy-tailed?)

### SAE Training Infrastructure
- [ ] P2-5: Implement SAE training loop (following ICLR 2025 best practices)
- [ ] P2-6: Implement loss function: reconstruction + L1 sparsity
- [ ] P2-7: Implement dead-feature detection and hunting
- [ ] P2-8: Implement learning rate scheduling and annealing
- [ ] P2-9: Create SAE validation metrics (interpretability, loss, sparsity)

### SAE Training Experiments
- [ ] P2-10: Train SAE for fc1 layer (latent_dim = 512, sparsity = 0.05)
- [ ] P2-11: Train SAE for fc2 layer (latent_dim = 512, sparsity = 0.05)
- [ ] P2-12: Train SAE for fc3 layer (latent_dim = 256, sparsity = 0.05)
- [ ] P2-13: Hyperparameter sweep: vary sparsity (0.01, 0.05, 0.1, 0.2)
- [ ] P2-14: Compare to baselines: PCA, ICA (nonlinear disentanglement)

### Feature Interpretation
- [ ] P2-15: Compute feature importance: sum(|∂E/∂z_i|) across dataset
- [ ] P2-16: Rank features by importance, select top 50 per layer
- [ ] P2-17: For each top feature, analyze activation patterns:
  - [ ] When does it activate? (high input norm, ill-conditioned, etc.?)
  - [ ] What does it encode? (create concept description)
- [ ] P2-18: Create feature-property correlation matrix:
  - [ ] Feature i vs. condition number, rank, norm, error, etc.
- [ ] P2-19: Visualize top features (activation maps, examples)

### Property Encoding Analysis
- [ ] P2-20: Can we predict matrix properties from SAE codes alone?
  - [ ] Train linear classifiers: z → cond_number (regression)
  - [ ] Train linear classifiers: z → rank (classification)
  - [ ] Train linear classifiers: z → error_magnitude (regression)
- [ ] P2-21: Measure R² for each property prediction
- [ ] P2-22: Identify which SAE features are most predictive

### Documentation & Interpretation
- [ ] P2-23: Create feature catalog (top 50 features per layer with descriptions)
- [ ] P2-24: Write interpretability report: what did the model learn?
- [ ] P2-25: Identify surprising/unexpected feature discoveries
- [ ] P2-26: Create visualizations: feature grid, property heatmaps

---

## Phase 3: Gradient Attribution & Interpretable Directions

### Integrated Gradients Implementation
- [ ] P3-1: Implement Integrated Gradients attribution method
  - [ ] Path: from random baseline to final prediction
  - [ ] Integration: numerical path integral of ∂E/∂x along path
- [ ] P3-2: Compute attributions for 100 test matrices
- [ ] P3-3: Create attribution heatmaps (input and output space)
- [ ] P3-4: Analyze: which input elements matter most? (ranking)
- [ ] P3-5: Analyze: which output elements are hardest to predict? (ranking)

### Concept Vector Discovery (CAV Extension)
- [ ] P3-6: Define concepts: "ill-conditioned", "rank-deficient", "large-norm", "noisy"
- [ ] P3-7: Create concept datasets by selecting matrices with known properties
- [ ] P3-8: Train concept classifiers (linear models on hidden layer outputs)
- [ ] P3-9: Extract CAVs: direction of concept classifier weights
- [ ] P3-10: Measure concept importance: sensitivity of energy to CAV direction
- [ ] P3-11: Create concept vector visualization

### Interpretable Direction Discovery (PCA in Gradient Space)
- [ ] P3-12: Compute ∇E for 1000 random predictions
- [ ] P3-13: Perform PCA on gradient matrix
- [ ] P3-14: Select top K directions (k=10)
- [ ] P3-15: For each direction, measure:
  - [ ] Energy impact (how much does E change when moving along direction?)
  - [ ] Property correlation (which matrix properties does direction affect?)
  - [ ] Semantic interpretability (can we name what this direction does?)
- [ ] P3-16: Rank directions by importance and interpretability

### Intervention Validation Experiments
- [ ] P3-17: For each discovered direction D:
  - [ ] Intervene: y' = y + t·D for t ∈ [-2, 2]
  - [ ] Measure: E(y'), error(y'), properties of y'
  - [ ] Visualize: effect curve
- [ ] P3-18: Hypothesis testing: direction should have predicted effect size
- [ ] P3-19: Create intervention report: which directions are truly causal?

### Causality Testing
- [ ] P3-20: Counterfactual analysis: what if we force model along direction D?
- [ ] P3-21: Comparison with random directions (sanity check)
- [ ] P3-22: Measure effect size vs. noise level
- [ ] P3-23: Test on different matrix types (well/ill-conditioned separately)

### Visualization & Documentation
- [ ] P3-24: Create attribution heatmap grid (input and output)
- [ ] P3-25: Create concept vector visualization
- [ ] P3-26: Create direction effect curves (grid of plots)
- [ ] P3-27: Create interpretable direction catalog with names and descriptions
- [ ] P3-28: Write Phase 3 synthesis: what are the "reasoning steps"?

---

## Phase 4: Riemannian Geometry Analysis

### SPD Manifold Setup
- [ ] P4-1: Implement SPD manifold distance (geodesic vs. Euclidean)
- [ ] P4-2: Implement Riemannian exponential/logarithmic maps
- [ ] P4-3: Implement tangent space projection
- [ ] P4-4: Validate: SPD operations on test matrices

### Landscape on Manifold
- [ ] P4-5: Recompute energy landscape using geodesic distances
- [ ] P4-6: Compare geodesic vs. Euclidean landscape (visual diff)
- [ ] P4-7: Measure: do discovered minima shift on manifold vs. Euclidean?
- [ ] P4-8: Analyze: does geometry explain landscape structure?

### Eigenvalue-Eigenvector Decomposition
- [ ] P4-9: For each prediction, decompose: Ŷ = V Λ V^T
- [ ] P4-10: Separate energy gradient into eigenvalue and eigenvector components
- [ ] P4-11: Analyze: does model learn eigenvalue/eigenvector structure separately?
- [ ] P4-12: Measure correlation with conditioning (eigenvalue spread)
- [ ] P4-13: Test hypothesis: model focuses on eigenvalues first, eigenvectors second?

### Optimal Transport Perspective
- [ ] P4-14: Compute Wasserstein distance: true vs. predicted
- [ ] P4-15: Correlate energy with transport cost
- [ ] P4-16: Analyze: does energy landscape align with OT geometry?
- [ ] P4-17: Compare different transport metrics (L2, Frobenius, Spectral)

### Geometric Property Analysis
- [ ] P4-18: Test which matrix properties dominate energy landscape:
  - [ ] Eigenvalue spread (conditioning)
  - [ ] Rank deficiency
  - [ ] Frobenius norm
- [ ] P4-19: Measure relative importance (variance explained)
- [ ] P4-20: Create 3D visualization (3 principal geometric properties)

### Documentation
- [ ] P4-21: Write geometric interpretability report
- [ ] P4-22: Compare Euclidean vs. Riemannian insights
- [ ] P4-23: Create geometric property importance rankings

---

## Phase 5: Layer-wise Compositional Analysis

### Linear Probing (ProbeGen Style)
- [ ] P5-1: Collect hidden layer outputs for 1000 examples
  - [ ] At each layer: after fc1, fc2, fc3
- [ ] P5-2: For each layer, train linear classifiers to predict:
  - [ ] Is final prediction well-conditioned? (binary, 0/1 threshold)
  - [ ] Error magnitude (regression)
  - [ ] Will optimization succeed? (binary, based on convergence metric)
  - [ ] Condition number of input (regression)
- [ ] P5-3: Measure linear separability: R², accuracy per layer
- [ ] P5-4: Plot curves: property separability vs. layer depth
- [ ] P5-5: Identify: at which layer do properties become separable?

### Gradient Flow Analysis
- [ ] P5-6: For 100 examples, trace ∇L backward through network
- [ ] P5-7: Measure gradient magnitude at each layer
- [ ] P5-8: Plot: gradient magnitude vs. layer depth
- [ ] P5-9: Identify: where does gradient attenuate? (information bottleneck)
- [ ] P5-10: Analyze: are gradients dominated by few layers?

### Energy Composition
- [ ] P5-11: Break down total energy by element: E = sum(f_i²)
- [ ] P5-12: Measure per-element contribution (f_i value)
- [ ] P5-13: Identify: which output dimensions are "responsible" for energy?
- [ ] P5-14: Analyze: do all output dimensions contribute equally?
- [ ] P5-15: Compute relative importance ranking

### Layer Criticality
- [ ] P5-16: Ablation: "remove" each layer by replacing with identity
- [ ] P5-17: Measure: how much does model performance degrade?
- [ ] P5-18: Rank layers by criticality
- [ ] P5-19: Identify: can model compensate if early layers fail?

### Compositionality Analysis
- [ ] P5-20: Do interpretable features (from Phase 2) compose through layers?
- [ ] P5-21: Analyze: are layer-wise features building blocks?
- [ ] P5-22: Identify: computation flow (feature interactions)

### Documentation
- [ ] P5-23: Create layer criticality report
- [ ] P5-24: Create gradient flow visualization
- [ ] P5-25: Write section: "How does the solution emerge?"

---

## Phase 6: Robustness Validation & Adversarial Testing

### Intervention Robustness
- [ ] P6-1: For top 20 discovered directions, repeat interventions on new seeds
- [ ] P6-2: Measure: are effects reproducible? (std dev of effect size)
- [ ] P6-3: Measure: effect consistency across matrix types
- [ ] P6-4: Create robustness score per direction

### Influence Function Analysis
- [ ] P6-5: Implement KFAC approximation for Hessian-inverse
- [ ] P6-6: Compute influence of each training example on test energy
- [ ] P6-7: Identify top-influential examples
- [ ] P6-8: Ablation: remove top 5% examples, retrain, measure interpretation changes
- [ ] P6-9: Measure: how much do interpretations change? (stability score)

### Sanity Checks
- [ ] P6-10: Orthogonal direction test: random direction ⊥ all discovered directions
  - [ ] Should have minimal energy impact
  - [ ] Measure: effect size should be statistically zero
- [ ] P6-11: Uninformative concept test: random concept vector
  - [ ] Should have minimal intervention effect
- [ ] P6-12: Gradient scrambling test: shuffle activation-energy associations
  - [ ] Should eliminate attributions

### Out-of-Distribution Generalization
- [ ] P6-13: Test on larger matrices (30×30, 40×40)
  - [ ] Do discovered directions transfer?
  - [ ] Do interpretable features (SAEs) still activate?
- [ ] P6-14: Test on different matrix types:
  - [ ] Ill-conditioned matrices
  - [ ] Low-rank matrices
  - [ ] Noisy matrices
- [ ] P6-15: Measure: what % of insights transfer to OOD? (transfer score)

### Edge Case Analysis
- [ ] P6-16: Singular matrices (rank < n)
  - [ ] Does energy function handle singularity gracefully?
  - [ ] Do interpretable directions still apply?
- [ ] P6-17: Pathological cases (near-singular, very large condition number)
- [ ] P6-18: Document failure modes

### Robustness Report
- [ ] P6-19: Summarize stability of all discovered interpretations
- [ ] P6-20: Confidence ratings per discovery
- [ ] P6-21: Create trustworthiness scoring framework

---

## Phase 7: Integration with Adversarial Mining

### Mining Strategy Baseline Setup
- [ ] P7-1: Train IRED with no negative mining (baseline)
- [ ] P7-2: Train IRED with random negative mining
- [ ] P7-3: Train IRED with adversarial (opt_step) negative mining
- [ ] P7-4: Save all three trained models

### Energy Landscape Comparison
- [ ] P7-5: Generate energy landscapes for all three models
- [ ] P7-6: Compute landscape metrics for each (convexity, sharpness, gap)
- [ ] P7-7: Visualize: side-by-side landscape comparison
- [ ] P7-8: Measure: do hard negatives sharpen minima?

### Interpretability Under Mining
- [ ] P7-9: Run Phase 2 (SAEs) on all three models
  - [ ] Compare feature importance rankings
  - [ ] Measure: are interpretable features more robust under mining?
- [ ] P7-10: Run Phase 3 (Gradients) on all three models
  - [ ] Do discovered directions persist across mining strategies?
  - [ ] Are effects stronger/weaker with mining?
- [ ] P7-11: Measure: interpretability consistency score (0-1)

### Mining Effectiveness Analysis
- [ ] P7-12: Measure test MSE for all three models
- [ ] P7-13: Compare: does interpretability improvement correlate with performance improvement?
- [ ] P7-14: Hypothesis: harder mining → more interpretable landscape → better generalization?

### Property-Guided Mining Strategy (Optional)
- [ ] P7-15: Use discovered interpretable properties to guide hard negative sampling
  - [ ] Eg: mine harder in high-conditioning-number regions
  - [ ] Eg: mine harder for rank-deficient matrices
- [ ] P7-16: Train IRED with property-guided mining
- [ ] P7-17: Compare performance and interpretability to baseline mining

### Final Synthesis
- [ ] P7-18: Write mining strategy recommendations
- [ ] P7-19: Create decision framework: which mining to use for which setting?
- [ ] P7-20: Propose best practices for interpretable energy-based learning

---

## Cross-Cutting Tasks

### Experiment Tracking & Logging
- [ ] Log all experiments to wandb (automatic)
- [ ] Create reproducibility manifest (seed, timestamp, code SHA)
- [ ] Archive all generated plots and data

### Documentation
- [ ] P-Doc-1: Methodology documentation (what we did and why)
- [ ] P-Doc-2: Results synthesis (major findings per phase)
- [ ] P-Doc-3: Limitations discussion
- [ ] P-Doc-4: Future work recommendations

### Code Quality
- [ ] Submit analysis code to repo with docstrings
- [ ] Create unit tests for key analysis functions
- [ ] Ensure all code is reproducible (fixed seeds, version pins)

### Paper/Presentation
- [ ] Collect all key figures into master presentation
- [ ] Write paper outline
- [ ] Create talking points for results
