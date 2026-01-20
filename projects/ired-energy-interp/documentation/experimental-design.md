# Experimental Design

## Overview

This document describes the experimental design choices, sampling strategies, and validation approaches for the IRED energy interpretability research project.

---

## Matrix Dataset Specification

### Training Data (Phase 0: Baseline IRED Training)
- **Source**: IRED Inverse class (dataset.py)
- **Size**: 20×20 symmetric positive definite (SPD) matrices
- **Count**: 10,000 training examples
- **Properties**:
  - Condition numbers: log-uniform in [1, 100]
  - Eigenvalues: sampled to achieve target conditioning
  - Random seed: 42 (fixed for reproducibility)

### Evaluation Dataset (Phases 1-7)
- **Size**: 500 test matrices (independent from training)
- **Properties**: Same conditioning distribution as training
- **Split**:
  - 400 for landscape/gradient analysis
  - 100 for robustness validation (separate seeds)

### OOD Test Sets (Phase 6)
- **Larger matrices**: 30×30, 40×40 (10 examples each)
- **Ill-conditioned**: condition number in [100, 1000] (20 examples)
- **Low-rank perturbed**: rank(A) = n-1 or n-2 (10 examples)
- **Total OOD**: ~60 test cases

---

## Baseline Model Configuration

### IRED Architecture
- **Diffusion steps**: 10 (as per original IRED)
- **Noise schedule**: Linear from 0.0 to 1.0
- **EBM architecture**: EBM class from models.py
  - Input dim: 400 (20×20 matrix vectorized)
  - Hidden dims: [512, 512] (fc1, fc2)
  - Time embedding: sinusoidal, 128 dim
  - Output: scalar energy (squared L2 norm of fc3)

### Training Configuration
- **Optimizer**: Adam (lr=1e-3)
- **Batch size**: 32
- **Epochs**: 50 (or until convergence)
- **Validation**: Every epoch, keep best checkpoint
- **No negative mining** (baseline, for comparison with Phase 7)

### Expected Performance
- Test MSE: ~0.01-0.05 (close to ground truth inverse)
- Convergence: ~200 optimization steps for inference

---

## Energy Landscape Sampling (Phase 1)

### 2D Slicing Strategy
- **Basis vectors**: Top 2 PCA directions of prediction space (computed on test set)
- **Grid**: 30×30 points per slice
- **Range**: ±3σ around ground truth (where σ = std of predictions)
- **Number of slices**: 50 random direction pairs (not just PCA)
- **Computation**: Batched energy evaluation (batch size=100)

### Gradient Field Sampling
- **Points**: 1000 random predictions in valid range
  - Sampling strategy: uniform grid + random perturbations
  - Validity check: cond(prediction) < 1000
- **Gradient computation**: Batched via PyTorch autograd
- **Normalization**: Per-example gradient norm scaling for visualization

### Hessian Sampling
- **Points**: 100 stratified samples (well-conditioned, moderate, ill-conditioned)
- **Computation**:
  - Finite differences (ε = 1e-5) with 2nd-order accuracy
  - OR: PyTorch Hessian computation with gradient checkpointing
- **Storage**: Save eigenvalues only (not full Hessian matrix)
- **Conditioning analysis**: Compute κ(H) for each point

### Quantitative Metrics
- **Convexity score**: % of landscape where ∇²E ≽ 0 (positive semi-definite)
- **Basin sharpness**: ||∇E|| at minimum / ||∇E|| at boundaries
- **Energy gap**: (E_max - E_min) / E_min
- **Spurious minima**: # critical points with E > E_min + 0.1 * (E_max - E_min)

---

## Sparse Autoencoder Training (Phase 2)

### Activation Collection
- **Layers**: 3 (after fc1, fc2, fc3)
- **Dataset**: 5000 examples (input, output, noise_level)
- **Activation normalization**:
  - Subtract layer mean
  - Divide by layer std
  - Clip outliers > 5σ

### SAE Architecture
- **Layer 1 (after fc1)**: 512 input → 512 latent, sparsity=0.05
- **Layer 2 (after fc2)**: 512 input → 512 latent, sparsity=0.05
- **Layer 3 (after fc3)**: 256 input → 256 latent, sparsity=0.05
- **Decoder**: Linear untied weights

### Training Configuration
- **Loss**: L_recon + λ_sparse * L1
- **λ_sparse ranges**: [0.01, 0.05, 0.1, 0.2]
- **Learning rate**: 1e-3 with cosine annealing
- **Batch size**: 256
- **Epochs**: Until convergence (typically 50-100)
- **Dead neuron handling**: Dead feature hunting, re-init < 1% activation

### Feature Interpretation
- **Importance ranking**: Sum of |∂E/∂z_i| across dataset
- **Activation pattern**: When does feature activate? (percentile decomposition)
- **Property correlation**: Pearson r with cond(A), ||A||_F, det(A), rank, error_mag
- **Significance testing**: FDR correction for multiple tests (α=0.05)

---

## Gradient Attribution (Phase 3)

### Integrated Gradients
- **Path**: Straight line from reference (zero matrix) to prediction
- **Steps**: 50 path points (numerical integration)
- **Aggregation**: Sum absolute attributions per element
- **Normalization**: Divide by total attribution magnitude

### Concept Activation Vectors (CAVs)
- **Concepts**:
  1. "Well-conditioned" (cond < 10)
  2. "Ill-conditioned" (cond > 50)
  3. "Large norm" (||A||_F > 1)
  4. "Rank-deficient" (simulated via perturbation)
- **Concept dataset**: 100-200 examples per concept
- **Probe**: Linear classifier on layer outputs
- **CAV**: Weight vector of trained probe

### Interpretable Directions
- **Basis**: PCA on gradient space (1000 random predictions)
- **Components**: Top 10-20 principal components
- **Ranking**:
  - Energy sensitivity (∂E / moving distance)
  - Property correlation (alignment with matrix properties)
  - Semantic interpretability (human rating)

### Intervention Validation
- **Perturbation magnitude**: t ∈ [-2, 2] (in units of discovered direction)
- **Measurements**:
  - Energy E(y + t·D)
  - Error ||A(y+t·D)⁻¹ - I||_F
  - Matrix properties: cond, norm, rank
- **Effect curves**: Fit polynomial, measure curvature
- **Hypothesis testing**: Direction should have predicted sign and magnitude

---

## Riemannian Geometry (Phase 4)

### SPD Manifold Metrics
- **Distance**: Affine-invariant Riemannian metric
  - d_geo(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F
- **Exponential map**: From tangent space to manifold
- **Logarithmic map**: From manifold to tangent space
- **Tangent projection**: Project vectors to tangent space

### Eigenvalue Analysis
- **Decomposition**: Each prediction Ŷ = V Λ V^T
- **Separate analysis**:
  - Eigenvalue error: ||λ_true - λ_pred||
  - Eigenvector error: principal angles between subspaces
  - Reconstruction: ||V Λ V^T - Ŷ||_F
- **Gradient decomposition**:
  - Part due to eigenvalue error
  - Part due to eigenvector error

### Optimal Transport
- **Distance metric**: Wasserstein-2 distance (empirical OT)
- **Computation**: Sinkhorn approximation for speed
- **Comparison**: L2, Frobenius, spectral distance alternatives
- **Alignment**: Measure energy correlation with transport cost

---

## Layer-wise Analysis (Phase 5)

### Linear Probing
- **Layer outputs**: After fc1, fc2, fc3, final energy
- **Probes trained to predict**:
  1. Is prediction well-conditioned? (binary, threshold=10)
  2. Error magnitude (regression, MSE loss)
  3. Will optimization converge? (binary, threshold = max 200 iterations)
  4. Input conditioning (regression)
- **Probe architecture**: Single linear layer
- **Evaluation**: R² (regression) or accuracy (classification)

### Gradient Flow
- **Backward pass**: Trace ∇L through network
- **Measurements per layer**:
  - Gradient norm ||∇||_2
  - Gradient signal ratio (std of gradient / mean)
  - Information bottleneck score
- **Visualization**: Gradient magnitude vs. layer depth

### Layer Criticality
- **Ablation**: Set layer to identity (bypass actual computation)
- **Metric**: Test MSE degradation
- **Interpretation**: How much does layer contribute to performance?

---

## Robustness Validation (Phase 6)

### Intervention Reproducibility
- **Multiple seeds**: Retrain model with 5 different random seeds
- **Measurement**: Repeat top interventions on each model
- **Stability metric**: Std dev of effect size across seeds
- **Threshold**: Effect is "robust" if std < 10% of mean effect

### Influence Functions
- **KFAC approximation**: Use Hessian inverse approximation
- **Computation**:
  1. Train model to convergence
  2. Compute KFAC approximation
  3. For test example z, compute influence: H^{-1} ∇_θ L(z)
  4. Multiply by training gradient: s_train · influence
- **Top influences**: Rank and analyze top 1% of training examples

### Sanity Checks
1. **Orthogonal direction**: Random direction ⊥ to discovered directions
   - Should have minimal energy impact (t-test H₀: effect = 0)
2. **Uninformative concept**: Random concept vector
   - Should not correlate with matrix properties
3. **Gradient scrambling**: Shuffle neuron activations
   - Attributions should vanish

### OOD Generalization
- **Test matrices**:
  - Larger (30×30, 40×40)
  - Different conditioning (100-1000)
  - Low-rank perturbations
- **Metrics**:
  - Do discovered features still activate?
  - Do discovered directions still apply?
  - Transfer score = % of insights that hold OOD

---

## Mining Strategy Comparison (Phase 7)

### Three Models Trained
1. **Baseline**: No negative mining
2. **Random**: Random matrices as negatives
3. **Adversarial**: Gradient-ascent hard negatives via opt_step

### Configuration
- **Same**: Architecture, hyperparameters, base dataset
- **Different**: Only mining strategy
- **Training duration**: Allow all to reach convergence
- **Test performance**: Report test MSE for all

### Landscape Comparison
- **Same analysis as Phase 1** for all three models
- **Metrics**:
  - Convexity: Does adversarial mining increase convexity?
  - Sharpness: Do minima become sharper?
  - Distinctness: Are spurious minima eliminated?

### Interpretability Comparison
- **SAE analysis** (Phase 2): Do features change?
- **Gradient analysis** (Phase 3): Do directions persist?
- **Consistency metric**: Overlap of top features/directions across strategies

### Property-Guided Mining (Optional)
- **Strategy**: Sample hard negatives more often in difficult regions
  - High conditioning number
  - Rank-deficient
  - Large Frobenius norm
- **Comparison**: vs. random mining, adversarial mining
- **Metric**: Performance, interpretability, efficiency

---

## Reproducibility Specifications

### Random Seeds
- **Model training**: 42 (base), then 43-47 for robustness
- **Data splitting**: 42 (fixed)
- **SAE training**: 100-104 (5 seeds for feature stability)
- **Initialization**: All layers use fixed seeds

### Code Versions
- **PyTorch**: 2.0.1 (or compatible)
- **NumPy**: 1.23+
- **SciPy**: 1.10+
- **All scripts**: Version-pinned via requirements.txt

### Computational Setup
- **GPU**: NVIDIA A100 or RTX 4090 (or compatible)
- **Memory requirement**: 16GB minimum
- **CPU**: Intel/AMD with 8+ cores for parallel analysis

### Documentation
- **Every experiment**: Record timestamp, seed, code SHA, command-line args
- **Artifacts**: Save figures, data, model checkpoints with experiment ID
- **Logging**: All to wandb with consistent naming

---

## Success Criteria

### Phase 1
- ✓ Energy landscape visualizations are interpretable
- ✓ Minima exist and are unique (not multi-modal)
- ✓ Gradients point toward minima
- ✓ Hessians are positive-definite at minimum

### Phase 2
- ✓ SAEs decompose features with < 10% dead neurons
- ✓ Top features correlate with matrix properties (r > 0.3)
- ✓ Features transfer across multiple seeds (> 80% overlap top 20)

### Phase 3
- ✓ Interventions have statistically significant effects (p < 0.05)
- ✓ Effect directions match hypothesis (sign correct)
- ✓ Discovered directions are stable across seeds (effect std < 15%)

### Phase 4
- ✓ Geodesic and Euclidean landscapes differ meaningfully
- ✓ Eigenvalue vs. eigenvector decomposition reveals structure
- ✓ Energy respects SPD manifold properties

### Phase 5
- ✓ Property separability increases with depth
- ✓ Gradient flow is continuous (no dead zones)
- ✓ Layer criticality ranking is meaningful

### Phase 6
- ✓ Robustness metrics > 70% (features/directions stable)
- ✓ OOD transfer score > 60% (most insights generalize)
- ✓ Sanity checks pass (orthogonal direction has ~0 effect)

### Phase 7
- ✓ Mining strategy affects landscape structure (measured metrics differ)
- ✓ Interpretability findings are consistent across mining strategies
- ✓ Actionable recommendations for mining are identified

---

## Timeline & Milestones

| Phase | Duration | Key Milestone | Go/No-Go |
|-------|----------|---------------|----------|
| SETUP | 1 week | Baseline model trained | Phase 1 can start |
| Phase 1 | 1 week | Energy landscapes characterized | Proceed if convex structure clear |
| Phase 2 | 2 weeks | SAE features found | Proceed if features >80% interpretable |
| Phase 3 | 2 weeks | Directions validated | Proceed if >70% causality validation |
| Phase 4 | 1 week | Geometry aligned | Optional, proceed to Phase 5 |
| Phase 5 | 1 week | Layer structure clear | Proceed if layers contribute differently |
| Phase 6 | 2 weeks | Robustness certified | Gate final claims on robustness metrics |
| Phase 7 | 2 weeks | Mining comparison complete | Final synthesis, recommendations |

**Total: ~12 weeks of research**

---

## Statistical Rigor

- **Multiple testing correction**: FDR correction (α=0.05) for feature-property correlations
- **Effect size reporting**: Always report effect size + confidence intervals, not just p-values
- **Power analysis**: Ensure n (samples) sufficient for small effect sizes (d=0.3)
- **Cross-validation**: Use k-fold (k=5) for all ML models (SAEs, probes, classifiers)
- **Replication**: Repeat key experiments across multiple seeds (n≥5)
