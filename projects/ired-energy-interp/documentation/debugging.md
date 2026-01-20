# Debugging & Issues Log

## Setup Phase Issues

### (none yet)

---

## Known Limitations

### Data Constraints
- Matrix size: 20×20 (to match IRED training)
- Condition number range: 1-100 (typical dataset range)
- May need larger matrices for OOD testing (Phase 6)

### Computational Constraints
- Hessian computation is O(d²) where d = 400 (20×20 matrices)
- Influence function computation requires Hessian inverse (expensive)
- SAE training on full activations may require careful memory management

### Methodological Notes
- Sparse autoencoders are non-unique (different initializations learn different decompositions)
- Use multiple random seeds to estimate feature stability
- Linear probing results depend on probe quality; use validation to verify

### Gradient Attribution Limitations
- Integrated Gradients are path-dependent; use straight-line paths for simplicity
- Gradient-based methods can be noisy; use smoothing + validation
- Attribution aggregation (sum, mean) affects interpretation

---

## Troubleshooting Guide

### SLURM Job Submission Fails
- Check: git branch matches required format (claude/ired-energy-interp-SESSION_ID)
- Check: GIT_SHA environment variable is set correctly
- Check: sbatch script paths are absolute
- Retry with exponential backoff (2s, 4s, 8s, 16s)

### IRED Model Import Errors
- Check: PYTHONPATH includes project directory
- Check: diffusion_lib, models, dataset modules are importable
- Verify: PyTorch version compatibility (need 2.0+)

### GPU Memory Errors
- Reduce batch size in energy landscape computation
- Reduce number of sampled points for gradient/Hessian analysis
- Use gradient checkpointing for Hessian computation (trade compute for memory)

### NaN/Inf in Energy Values
- Check: input matrices are valid (invertible, well-conditioned enough)
- Check: numerical stability of energy computation (may need scaling)
- Verify: model checkpoint was trained properly

### Interpretation Divergence Across Runs
- Expected: some variation in discovered features (SAE non-uniqueness)
- Measure: how much do features change across 5+ different random seeds?
- Document: which features are stable vs. unstable

---

## Testing Checklist

Before each phase, verify:
- [ ] Baseline model loads without errors
- [ ] Can compute energy and gradients
- [ ] Analysis utilities run on small test batches
- [ ] Visualization generation succeeds
- [ ] Logging to wandb works
- [ ] Reproducibility with fixed seeds

---

## Common Pitfalls

1. **Forgetting to normalize metrics**: Energy landscapes can have very different scales; always use filter normalization or percentage changes

2. **Confusing directions**: Make sure when analyzing "interpretable directions" that you distinguish:
   - Gradient space directions (where gradients point)
   - Latent space directions (where discovered features lie)
   - Prediction space directions (where output moves)

3. **Property correlation vs. causation**: Discovering feature X correlates with condition number ≠ feature X causes conditioning behavior. Use interventions to validate causality.

4. **Generalization assumptions**: Results on 20×20 matrices may not transfer to 30×30 matrices or different matrix types. Test explicitly (Phase 6).

5. **SAE instability**: If SAE training is unstable, check:
   - Learning rate (try 1e-3 to 1e-4)
   - Sparsity penalty weight (balance reconstruction vs. sparsity)
   - Dead neuron fraction (should be < 5%)

---

## Expected Behavior

### Phase 1: Energy Landscapes
- Energy should form single well-defined basin near true inverse
- Gradients should point toward minimum
- Hessians should be positive-definite (or nearly so)
- Landscapes should be roughly convex for well-conditioned matrices

### Phase 2: Sparse Autoencoders
- Reconstruction loss should decrease with increasing autoencoder width
- Most features should have < 5% activation probability
- Top features should correlate with matrix properties
- Different seeds should discover overlapping (but not identical) features

### Phase 3: Gradient Attribution
- Important input elements should correspond to off-diagonal entries
- Important output elements should correspond to diagonal entries
- Interventions on interpreted directions should have predicted effects

### Phase 4: Riemannian Geometry
- Geodesic distances should differ from Euclidean (especially for ill-conditioned matrices)
- Energy landscape structure should be preserved under manifold operations
- Eigenvalue/eigenvector decomposition should reveal structured learning

### Phase 5: Layer-wise Analysis
- Property separability should increase with depth
- Early layers should have weak separability
- Late layers should achieve high R²
- Gradient flow should be continuous (no dead zones)

### Phase 6: Robustness
- Discovered features should be robust across multiple seeds (not all, but most)
- Intervention effects should be reproducible (within noise)
- Influence functions should identify meaningful training examples
- OOD performance should be > random but < in-distribution

### Phase 7: Mining Integration
- Adversarial mining should produce sharper energy minima
- Hard negatives should make landscape more interpretable
- Discovered directions should persist across mining strategies

---

## Reporting Issues

When an issue arises, document:
1. **What happened**: Specific error or unexpected behavior
2. **When it happened**: Which experiment, which phase
3. **Environment**: PyTorch version, GPU type, Python version
4. **Minimal reproduction**: Standalone script to reproduce
5. **Attempted solutions**: What did you try?
6. **Current status**: Blocker, workaround, resolved
