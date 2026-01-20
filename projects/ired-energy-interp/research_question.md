# Research Question: IRED Energy Dynamics and Interpretability

## Primary Research Question
**How can we understand and interpret the energy field learned by IRED on matrix inversion problems? Do interpretable directions in the energy landscape correspond to real matrix geometric properties?**

## Sub-Questions
1. **Energy Landscape Structure**: Is the learned energy landscape convex, multi-modal, or sparse? What is the geometry of minima and error basins?

2. **Interpretable Features**: What do the hidden representations in the EBM learn? Can we decompose them into human-interpretable features using sparse autoencoders?

3. **Gradient Semantics**: Do gradient fields point toward correcting specific matrix properties (conditioning, rank, eigenvalues)? Which input/output elements are most critical?

4. **Causal Directions**: Are discovered "interpretable directions" causally important, or merely correlated with performance?

5. **Geometric Alignment**: Does the energy function respect the underlying Riemannian geometry of the SPD manifold where matrices live?

6. **Generalization**: Do discovered interpretations transfer to out-of-distribution matrices (larger sizes, different conditioning)?

7. **Mining Connection**: Do hard negatives create more interpretable energy landscapes? Can interpretability insights improve negative mining strategies?

## Approach Overview

### Phase 1: Energy Landscape Characterization
- Visualize energy surface via 2D slicing and filter-normalized contours
- Analyze gradient field structure and alignment
- Compute Hessian properties (eigenvalues, conditioning, curvature)

### Phase 2: Latent Space Interpretability (Sparse Autoencoders)
- Train SAEs on hidden layer activations per layer
- Identify top interpretable features
- Correlate features with matrix properties (condition number, norm, rank, etc.)

### Phase 3: Gradient-Based Attribution & Concepts
- Apply Integrated Gradients for input-output attribution
- Discover interpretable concept vectors via CAV extension
- Find latent directions via PCA on gradient space
- Validate via intervention experiments

### Phase 4: Geometry-Aware Analysis
- Analyze on SPD manifold using Riemannian metrics
- Decompose energy by eigenvalue vs. eigenvector contributions
- Connect to optimal transport theory

### Phase 5: Layer-wise Composition
- Train linear probes to test property separability per layer
- Analyze gradient flow and energy composition
- Identify computational structure

### Phase 6: Robustness & Validation
- Intervention validation experiments
- Influence function analysis
- Adversarial sanity checks
- OOD generalization testing

### Phase 7: Integration with Mining
- Compare energy landscapes across baseline vs. mining strategies
- Evaluate interpretability under different training regimes
- Propose property-guided mining strategies

## Expected Outcomes
- Comprehensive visualization suite of energy landscapes
- Decomposition of EBM into interpretable feature basis
- Catalog of semantic interpretable directions
- Causal evidence linking directions to performance
- Geometric validation on SPD manifolds
- Best practices for mining strategy design

## Context
This project extends IRED (Iterative Reasoning through Energy Diffusion) from the original yilundu/ired_code_release. We apply state-of-art mechanistic interpretability techniques (sparse autoencoders, gradient attribution, causal interventions) developed in 2024-2025 to understand how IRED learns to solve matrix inversion via energy-based diffusion.
