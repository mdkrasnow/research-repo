# Decision Summary

## Key Project Decisions

### 1. **7-Phase Research Structure**

**Decision**: Organize research into 7 sequential/parallel phases

**Rationale**:
- Phase 1 (Energy Landscape): Foundation for all other analyses
- Phases 2-5: Parallel deep-dives into different interpretability aspects
- Phase 6: Robustness validation (gates final claims)
- Phase 7: Connection back to original mining research question

**Alternatives Considered**:
- Single monolithic analysis → too large, unclear dependencies
- All phases in parallel → impossible without Phase 1 results
- Three broad phases → insufficient granularity for publication

**Trade-offs**:
- More phases = longer timeline but better modularity
- Can publish intermediate results (Phase 1-3) while later phases run

---

### 2. **Sparse Autoencoders as Core Interpretability Technique**

**Decision**: Use SAEs (ICLR 2025) as primary method for feature decomposition

**Rationale**:
- State-of-art mechanistic interpretability (proven on protein LMs, LLMs)
- Scaling laws well-understood (can push to larger dimensions)
- Non-unique but stable across seeds (measure robustness empirically)
- Directly outputs interpretable basis (unlike black-box dimensionality reduction)

**Alternatives Considered**:
- PCA/ICA → unsupervised, less interpretable
- Attention visualization → N/A (IRED uses dense EBM, no attention)
- Activation clustering → less principled than SAEs

**Trade-offs**:
- Higher training cost (but still feasible on 1 GPU)
- Feature non-uniqueness (must run multiple seeds)
- Hyperparameter selection (sparsity, latent dim)

---

### 3. **Integrated Gradients + CAVs for Attribution**

**Decision**: Combine Integrated Gradients (robust, principled) with CAVs (semantic)

**Rationale**:
- Integrated Gradients theoretically grounded, robust to perturbations
- CAVs give semantic meaning to discovered directions
- Visual-TCAV (2024) showed how to combine both effectively
- Recent NeurIPS 2024 work validates both techniques

**Alternatives Considered**:
- Vanilla gradients → noisy, not robust
- Layer-wise relevance propagation → harder to interpret for energy functions
- Attention rollout → N/A (no attention in EBM)
- Influence functions alone → slow, doesn't capture all attributions

**Trade-offs**:
- Path-dependent (Integrated Gradients depends on path choice)
- CAV training requires labeled concept datasets
- Multiple methods → more experimental work

---

### 4. **Riemannian Geometry on SPD Manifolds**

**Decision**: Respect underlying manifold structure when analyzing energy geometry

**Rationale**:
- Matrices (especially SPD) naturally live on Riemannian manifolds
- Recent work (NORM 2024, RNN-SPD 2024) shows manifold-aware methods improve interpretability
- Euclidean analysis may miss geometric structure
- Geodesic distance more meaningful than L2 distance for matrices

**Alternatives Considered**:
- Pure Euclidean analysis → simpler but potentially misleading
- Tangent space approximation → faster but less accurate
- Ignore geometry → miss important structural properties

**Trade-offs**:
- Adds computational complexity (SPD operations, geodesic distances)
- Requires careful numerical implementation (matrix exponentials)
- Harder to visualize (can't easily embed SPD space in 2D)
- Payoff: More principled interpretations, geometric validation

---

### 5. **Causal Intervention Validation (Phase 3 & 6)**

**Decision**: Explicitly test causality of discovered directions via interventions

**Rationale**:
- Correlation (feature X correlates with property) ≠ causation
- Interventions directly probe causal structure
- Recent work (Neural Causal Graphs 2024) shows feasibility
- Necessary for scientific credibility of interpretability claims

**Alternatives Considered**:
- Correlation + accounting → easier but lower confidence
- Ablation studies only → doesn't prove direction causality
- Assumption of linearity → may not hold

**Trade-offs**:
- Requires careful experiment design (avoid confounds)
- Time-consuming (multiple interventions × multiple seeds)
- Negative results possible (direction not truly causal)

**Risk Mitigation**:
- Use sanity checks (orthogonal direction should have no effect)
- Test on multiple matrix types (generalization)
- Report confidence intervals on effect sizes

---

### 6. **Robustness as Primary Validation Gate (Phase 6)**

**Decision**: Make Phase 6 robustness a hard gate for final claims

**Rationale**:
- Interpretability findings are often brittle (Explaining Explainability, 2024)
- Feature instability (SAE non-uniqueness) must be measured
- Need evidence that insights transfer to OOD data
- Robustness certification is increasingly expected in interpretability papers

**Alternatives Considered**:
- Skip robustness, rely on initial validation → risky
- Treat robustness as separate publication → delays main insights
- Light robustness check → insufficient validation

**Trade-offs**:
- Adds 2 weeks of experimentation
- May uncover failures (hard to recover if Phase 6 fails)
- Payoff: High confidence in final claims, publishability

---

### 7. **Mining Strategy Integration (Phase 7)**

**Decision**: Close the loop by connecting interpretability back to mining research question

**Rationale**:
- Original research context is adversarial mining on matrix inversion
- Test if interpretability insights inform better mining strategies
- Evaluate if hard negatives create more interpretable landscapes
- Actionable recommendations (not just analysis)

**Alternatives Considered**:
- Skip mining, publish pure interpretability → loses connection to motivation
- Do mining separately → missed opportunity for integration
- Only compare landscapes → insufficient analysis

**Trade-offs**:
- Requires retraining models (expensive)
- But provides closure and practical impact
- Enables property-guided mining innovation

---

### 8. **Technology Stack Selection**

**Decision**: Use PyTorch + modern interpretability libraries

**Rationale**:
- PyTorch: flexible, auto-differentiation, SLURM-friendly
- NumPy/SciPy: standard for matrix operations
- Scikit-learn: robust probing, classification, metrics
- Wandb: experiment tracking, visualization

**Alternatives Considered**:
- JAX → functional but steeper learning curve
- TensorFlow → less flexible for custom loss computation
- Custom CUDA → overkill, maintenance burden

**Trade-offs**:
- NumPy/SciPy slower than optimized BLAS (but sufficient)
- Wandb has free tier but requires internet connection
- Lock-in to PyTorch ecosystem

---

### 9. **Dataset Choices (20×20 matrices)**

**Decision**: Work with 20×20 matrices (as in original IRED training)

**Rationale**:
- Matches IRED baseline (reproducibility)
- Small enough for fast iteration (Hessian, SAE training)
- Large enough to be interesting (not trivial)
- Aligns with prior work

**Alternatives Considered**:
- Larger matrices (30×30, 40×40) → more realistic but slower
- Synthetic simple matrices → less representative
- Mixed sizes → hard to compare

**Trade-offs**:
- Limited realism for practical applications
- Risk: insights may not transfer to larger matrices (address in Phase 6)
- Decision: Plan OOD testing with larger matrices to assess transfer

---

### 10. **Publication & Presentation Plan**

**Decision**: Plan for 2-3 publications (incremental disclosure)

**Rationale**:
- Phase 1-3: "Interpretable Energy Landscapes in IRED" (interpretability focus)
- Phase 4-5: "Geometric Foundations of Energy-Based Reasoning" (geometry focus, optional)
- Phase 7: "Energy Interpretability for Improved Learning" (mining recommendations)

**Alternatives Considered**:
- Single massive paper → too long, harder to publish
- One paper per phase → too incremental
- Combine all → good fit for conference/journal special issue

**Trade-offs**:
- Multiple papers require coordination
- Each must be self-contained
- But allows focus and better acceptance rate

---

## Uncertain Decisions (To Be Validated in Phase 1)

1. **Are energy landscapes actually interpretable?**
   - Decision point: Phase 1 - if chaotic, reconsider approach

2. **Will SAEs find stable, reproducible features?**
   - Decision point: Phase 2 - if feature instability > 50%, simplify to PCA

3. **Do discovered directions have causal effects?**
   - Decision point: Phase 3 - if < 30% of directions pass intervention test, revise

4. **Will interpretations transfer to OOD?**
   - Decision point: Phase 6 - if transfer < 40%, add "limited to in-distribution" caveat

5. **Does mining strategy actually improve interpretability?**
   - Decision point: Phase 7 - if no difference, acknowledge negative result

---

## Go/No-Go Criteria Between Phases

| Phase Boundary | Success Criterion | If Failed |
|----------------|------------------|-----------|
| SETUP → P1 | Baseline model converges, MSE < 0.05 | Retrain or debug model |
| P1 → P2 | Energy landscape has clear structure | Analyze model for pathologies |
| P2 → P3 | SAE reconstruction loss < 10% | Increase latent dim or change sparsity |
| P3 → P4 | >50% of directions pass intervention test | Use simpler direction discovery |
| P4 → P5 | Geodesic analysis differs from Euclidean | Can skip to P5 (manifold optional) |
| P5 → P6 | Layer-wise probes achieve R² > 0.5 | May indicate trivial features |
| P6 → P7 | Robustness metrics > 60% | Add caveats to final claims |
| P7 → Publication | All phases complete, gate criteria met | Plan revision/resubmission |

---

## Flexibility & Adaptation Points

The plan is modular. If needed:

1. **Speed up**: Skip Phase 4 (manifold analysis), proceed to Phase 5
2. **Simplify**: Replace SAEs with PCA if training unstable
3. **Deepen**: Add extra phases (e.g., biological plausibility, domain comparison)
4. **Pivot**: If Phase 1 shows unexpected phenomena, adapt later phases
5. **Extend**: Phase 7 could lead to new mining algorithm design (future work)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Energy landscape is chaotic | Phase 1 serves as validation gate |
| Features are unstable | Multiple seed requirement ensures stability measure |
| Interpretations are brittle | Phase 6 robustness validation |
| Insights don't generalize | Phase 6 OOD testing |
| Mining shows no impact | Honestly report null result; still publishable |
| Compute time exceeds estimate | Can parallelize phases 2-5 |
| Model convergence issues | Early intervention in SETUP phase |
