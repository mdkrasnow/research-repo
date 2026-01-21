# Implementation Roadmap

## Phase 0: Setup & Infrastructure

### Environment Setup
- [ ] Create requirements.txt with interpretability libraries
  - geomstats (Riemannian geometry)
  - pyhessian (Hessian analysis)
  - torch-hessian-eigenthings (alternative Hessian library)
  - umap-learn (dimensionality reduction)
  - plotly (interactive visualizations)
- [ ] Test IRED baseline training on matrix tasks
- [ ] Load pretrained IRED checkpoints for analysis

### Data Pipeline
- [ ] Create dataset loaders for matrix problems
  - Matrix completion (various ranks)
  - Matrix inversion (various condition numbers)
  - Matrix addition (baseline)
- [ ] Generate (x, y) sample pairs for analysis
  - Correct solutions
  - Perturbed solutions (various noise levels)
  - Systematic perturbations (rank, singular values, etc.)

## Phase 1: Energy Landscape Analysis

### 1.1 Hessian Analysis
- [ ] Implement Hessian computation wrapper
  - `compute_hessian(model, x, y, k)` for energy level k
  - Use PyHessian Lanczos method for top-k eigenvalues
- [ ] Batch computation across dataset
- [ ] Eigenspectrum visualization
  - Histogram of eigenvalues
  - Track across annealing levels k=1..10
- [ ] Eigenvector analysis
  - Project onto geometric bases (singular vectors, etc.)
  - Measure alignment with problem structure

### 1.2 Landscape Visualization
- [ ] Implement filter-normalized 2D slicing
- [ ] Geometric direction generator
  - Rank-preserving perturbations
  - Singular value perturbations
  - Subspace rotations
- [ ] Energy contour plotting
- [ ] Gradient field quiver plots

### 1.3 Mode Connectivity
- [ ] Geodesic path finder on Grassmannian (using Geomstats)
- [ ] Energy along paths
- [ ] Compare to straight-line interpolation

## Phase 2: Sparse Autoencoder Implementation

### 2.1 SAE Architecture
- [ ] Design sparse autoencoder
  ```python
  class GradientSAE(nn.Module):
      def __init__(self, input_dim, hidden_dim, sparsity_coef):
          # Encoder: gradient -> sparse features
          # Decoder: sparse features -> reconstructed gradient
  ```
- [ ] Training loop with L1 regularization
- [ ] Gradient collection pipeline
  - Sample (x, y) pairs
  - Compute ∇_y E^k(x, y) for all k
  - Store in dataset

### 2.2 Feature Analysis
- [ ] Feature visualization
  - Top activating examples for each feature
  - Geometric interpretation
- [ ] Feature-geometry alignment metrics
  - Correlation with rank
  - Correlation with singular value perturbations
  - Correlation with subspace angles

### 2.3 Activation Atlas
- [ ] Collect intermediate activations from E_θ network
- [ ] UMAP embedding
- [ ] Interactive visualization (plotly)

## Phase 3: Riemannian Geometry Analysis

### 3.1 Grassmannian Tools (Geomstats)
- [ ] Wrapper for matrix -> Grassmann point conversion
- [ ] Principal angle computation
- [ ] Riemannian distance on Gr(n, k)
- [ ] Geodesic interpolation

### 3.2 Tangent/Normal Decomposition
- [ ] Compute tangent space basis at y*
  - For completion: tangent to rank-r manifold
  - Use SVD-based projection
- [ ] Project gradient: ∇_y E = ∇_T E + ∇_N E
- [ ] Compute curvature ratios across dataset

### 3.3 Parallel Transport
- [ ] Implement parallel transport along optimization trajectory
- [ ] Track gradient evolution (intrinsic vs extrinsic)

## Phase 4: Score Function Analysis

### 4.1 Gradient Field SVD
- [ ] Collect gradient samples ∇_y E(x, y)
- [ ] Compute SVD: ∇_y E = UΣV^T
- [ ] Analyze principal components
  - Alignment with singular vectors
  - Alignment with eigenvectors

### 4.2 Implicit Bias Tracking
- [ ] Training curve analysis
  - Track effective rank during training
  - Compare IRED vs explicit low-rank regularization
- [ ] Solution rank distribution
- [ ] Nuclear norm analysis

## Phase 5: NTK Analysis

### 5.1 NTK Computation
- [ ] Implement empirical NTK using functorch
- [ ] Fast trace estimation (Hutch++)
- [ ] Compute across annealing levels

### 5.2 NTK Alignment
- [ ] Correlation with geometric structures
- [ ] Evolution during training

## Phase 6: Circuit Discovery

### 6.1 Activation Patching
- [ ] Implement ablation framework
  - Zero out specific layers/heads
  - Measure energy change
- [ ] Systematic sweep across architecture
- [ ] Causal importance scores

### 6.2 Circuit Extraction
- [ ] Identify critical computational paths
- [ ] Visualize circuit graphs
- [ ] Interpret circuit functions

## Experiments & Analysis

### Experiment Queue
1. **EXP-001**: Hessian eigenspectrum on matrix completion (rank 1, 2, 5)
2. **EXP-002**: Energy landscape visualization (2D slices)
3. **EXP-003**: SAE training on gradient field
4. **EXP-004**: Grassmannian distance vs energy correlation
5. **EXP-005**: Tangent/normal decomposition analysis
6. **EXP-006**: Score function SVD alignment
7. **EXP-007**: Implicit bias tracking during training
8. **EXP-008**: NTK alignment with geometry
9. **EXP-009**: Activation patching circuit discovery
10. **EXP-010**: Comprehensive annealing analysis (k=1..10)

### Analysis Scripts
- [ ] `analysis/hessian_analysis.py`
- [ ] `analysis/landscape_viz.py`
- [ ] `analysis/sae_training.py`
- [ ] `analysis/grassmann_geometry.py`
- [ ] `analysis/score_analysis.py`
- [ ] `analysis/ntk_computation.py`
- [ ] `analysis/circuit_discovery.py`

### Visualization Dashboard
- [ ] Jupyter notebook for interactive exploration
- [ ] Plotly dashboard for energy landscapes
- [ ] Comparative analysis across annealing levels

## Cluster Execution

### SLURM Scripts
- [ ] `slurm/train_ired.sbatch` - Train IRED models
- [ ] `slurm/hessian_analysis.sbatch` - Hessian computation (GPU)
- [ ] `slurm/sae_training.sbatch` - SAE training
- [ ] `slurm/full_analysis.sbatch` - Comprehensive analysis pipeline

### Resource Requirements
- GPU: A100 for training, inference, Hessian computation
- CPU: Sufficient for Geomstats computations
- Memory: ~32GB for large Hessian matrices
- Storage: ~50GB for checkpoints, gradient samples

## Documentation

### Papers to Write
- [ ] Technical report: IRED energy landscape interpretability
- [ ] Main paper: Geometric interpretability of energy-based reasoning
- [ ] Supplementary: Detailed methods and ablations

### Figures
- [ ] Hessian eigenspectrum plots
- [ ] Energy landscape contours
- [ ] SAE feature visualizations
- [ ] Grassmannian geometry diagrams
- [ ] Circuit diagrams
- [ ] Annealing progression analysis

## Timeline Estimate
- **Phase 1**: 2-3 weeks (landscape analysis)
- **Phase 2**: 2 weeks (SAE implementation)
- **Phase 3**: 2 weeks (Riemannian geometry)
- **Phase 4**: 1 week (score analysis)
- **Phase 5**: 1 week (NTK)
- **Phase 6**: 2 weeks (circuits)
- **Writing**: 2-3 weeks

**Total**: ~12-14 weeks for comprehensive study
