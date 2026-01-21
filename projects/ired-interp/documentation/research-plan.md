# IRED Energy Field Interpretability Research

## Research Question
How can we interpret and understand the energy dynamics of IRED's learned energy landscapes for matrix reasoning problems? What geometric structures do the energy fields capture?

## Background
IRED (Iterative Reasoning through Energy Diffusion) learns K=10 annealed energy functions E_θ^k(x,y) that assign low energy to correct solutions and high energy to incorrect ones. The model performs gradient-based optimization during inference: y^t = y^(t-1) - λ∇_y E(x, y^(t-1)).

## Research Phases

### Phase 1: Energy Landscape Characterization
**Objective**: Understand the geometry and topology of learned energy landscapes

#### 1.1 Hessian Eigenspectrum Analysis
- Compute Hessian H = ∇²_y E^k(x, y*) at correct solutions
- Use PyHessian with Lanczos iteration for top-k eigenvalues/eigenvectors
- Track eigenspectrum evolution across K=10 annealing levels
- **Hypothesis**: Small eigenvalues correspond to invariant directions (null space, tangent space of manifold)

**Metrics**:
- Eigenvalue distribution (bulk vs outliers)
- Spectral gaps
- Effective rank (participation ratio)
- Condition number evolution

#### 1.2 Loss Landscape Visualization
- Filter-normalized 2D slices (Li et al. 2018 method)
- Geodesic paths between solutions on Grassmannian
- Energy along geometrically meaningful directions:
  - Rank-increasing vs rank-preserving perturbations
  - Singular value scaling
  - Subspace rotations

#### 1.3 Mode Connectivity
- Find low-energy paths between different solutions
- Use geodesic interpolation on Grassmann manifold
- Compare energy barriers for in-distribution vs OOD problems

### Phase 2: Sparse Autoencoder Feature Decomposition
**Objective**: Discover monosemantic features in gradient field

#### 2.1 SAE Architecture for Energy Gradients
- Collect N×M gradient samples: ∇_y E^k(x,y) for N problems × M solutions
- Train sparse autoencoder with L1 regularization
- Objective: L_SAE = ||∇_y E - Decode(Encode(∇_y E))||² + λ||Encode(∇_y E)||_1

**Analysis**:
- Identify interpretable gradient features
- Test alignment with geometric structures:
  - Rank-correction features
  - Singular value adjustment features
  - Subspace rotation features

#### 2.2 Activation Atlas for Energy Function
- Sample (x, y) pairs, compute intermediate layer activations
- UMAP projection to 2D
- Overlay with energy values and geometric properties

### Phase 3: Geometric Manifold Analysis
**Objective**: Validate energy function respects matrix manifold structure

#### 3.1 Grassmannian Distance & Principal Angles
Using **Geomstats**:
- Represent solutions as subspaces (column spaces)
- Compute principal angles between subspaces
- Measure Riemannian distance on Grassmannian
- **Analysis**: Correlation between d_Grassmann(y, y*) and E(x,y)

#### 3.2 Tangent Space Decomposition
At correct solution y*, decompose perturbations:
- **Tangent space** T_y* M: rank-preserving (on-manifold)
- **Normal space** N_y* M: rank-increasing (off-manifold)

Compute:
- ∇_y E = ∇_T E + ∇_N E (tangent + normal components)
- Curvature ratio: ||∇_N E|| / ||∇_T E||
- **Hypothesis**: Energy steep in normal direction, flat in tangent direction

#### 3.3 Parallel Transport Analysis
- Track gradient direction changes along optimization trajectory
- Use Riemannian parallel transport to separate intrinsic vs extrinsic changes

### Phase 4: Score Function & Implicit Bias Analysis
**Objective**: Analyze score matching and implicit low-rank bias

#### 4.1 Score Matching Decomposition
IRED trained with: ∇_y E^k ≈ ε (denoising direction)

**Analysis**:
- SVD of gradient field: ∇_y E(x,y) = UΣV^T
- Check alignment: Do singular vectors align with problem structure?
  - Matrix completion: align with ground truth singular vectors?
  - Matrix inversion: align with eigenvectors?

#### 4.2 Implicit Bias Tracking
- Train IRED on matrix problems
- Track effective rank during training
- Compare to explicit low-rank regularization
- **Question**: Does energy landscape implicitly favor low-rank without explicit constraints?

### Phase 5: Neural Tangent Kernel Analysis
**Objective**: Understand function space dynamics

#### 5.1 Empirical NTK Computation
- Compute NTK using fast trace estimation (Hutch++)
- NTK reveals input feature sensitivity

**Analysis**:
- NTK alignment with geometric structures
- Feature learning evolution across annealing levels

### Phase 6: Circuit Discovery for Reasoning
**Objective**: Identify computational mechanisms

#### 6.1 Activation Patching
- Ablate attention heads, MLP layers
- Measure impact on energy at different σ_k
- **Causal tracing**: Which layers encode constraints?

#### 6.2 Interpretable Subgraphs
- Extract computational circuits
- Identify components for:
  - Constraint satisfaction
  - Denoising
  - Geometric structure handling

## Experimental Design

### Datasets
1. **Matrix Completion**: rank r ∈ {1, 2, 5, 10}, size n ∈ {20, 50}
2. **Matrix Inversion**: condition number κ ∈ {10, 100, 1000}
3. **Matrix Addition**: baseline (simplest constraint)

### Success Criteria
1. **Interpretable features**: SAE features align with known geometric structures
2. **Geometric consistency**: Energy respects manifold structure
3. **Annealing insight**: Clear coarse-to-fine progression through σ_k levels
4. **Mechanistic understanding**: Identify which network components enforce which constraints

## Key Innovation
This research combines state-of-the-art interpretability techniques (SAEs, circuit discovery) with rigorous Riemannian geometry to understand energy-based reasoning in a principled way.

## Tools & Libraries
- **PyHessian**: Hessian eigenvalue/eigenvector computation
- **Geomstats**: Riemannian geometry (Grassmann manifolds, geodesics, parallel transport)
- **torch.func (functorch)**: Efficient Jacobian, Hessian-vector products
- **Neural Tangents**: NTK computation
- Custom SAE implementation
