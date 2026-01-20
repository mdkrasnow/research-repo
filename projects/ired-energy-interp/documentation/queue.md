# Experiment Queue

## Current Status: SETUP Phase

### Next Actions (Priority Order)

#### SETUP Phase Tasks
1. **Environment Setup** - Create Python env, install dependencies
   - Status: PENDING
   - Est. Duration: 15 minutes
   - Dependencies: None
   - Owner: automated

2. **Dataset Preparation** - Load matrix inversion data
   - Status: PENDING
   - Est. Duration: 30 minutes
   - Dependencies: Environment setup
   - Owner: automated

3. **Baseline Model Training** - Train IRED on matrix inversion
   - Status: PENDING
   - Est. Duration: 2 hours (GPU)
   - Dependencies: Dataset preparation
   - Owner: SLURM (gpu_test partition)
   - Resources: 1 GPU, 16GB RAM

4. **Analysis Infrastructure** - Set up interpretability tools
   - Status: PENDING
   - Est. Duration: 2 hours
   - Dependencies: Baseline model trained
   - Owner: automated

5. **Phase 1 Kickoff** - Energy landscape analysis
   - Status: PENDING
   - Est. Duration: 4 hours (GPU)
   - Dependencies: Analysis infrastructure ready
   - Owner: SLURM (gpu_test partition)
   - Resources: 1 GPU, 16GB RAM

---

## Phase 1 Planned Experiments

| Exp ID | Task | Est. Time | GPU | Status |
|--------|------|-----------|-----|--------|
| P1-01 | Energy landscape slicing (50 slices) | 2h | 1 | queued |
| P1-02 | Gradient field computation (1000 points) | 1h | 1 | queued |
| P1-03 | Hessian analysis (100 points) | 2h | 1 | queued |
| P1-04 | Quantitative metrics aggregation | 1h | 0 | queued |
| P1-05 | Visualization generation | 1h | 0 | queued |

**Phase 1 Total**: ~7 hours of compute (mostly GPU)

---

## Phase 2 Planned Experiments

| Exp ID | Task | Est. Time | GPU | Status |
|--------|------|-----------|-----|--------|
| P2-01 | Activation extraction | 1h | 1 | pending |
| P2-02 | SAE training (fc1 layer) | 3h | 1 | pending |
| P2-03 | SAE training (fc2 layer) | 3h | 1 | pending |
| P2-04 | SAE training (fc3 layer) | 2h | 1 | pending |
| P2-05 | Sparsity hyperparameter sweep | 4h | 1 | pending |
| P2-06 | Feature interpretation | 2h | 0 | pending |
| P2-07 | Property correlation analysis | 1h | 0 | pending |

**Phase 2 Total**: ~16 hours of compute

---

## Phase 3 Planned Experiments

| Exp ID | Task | Est. Time | GPU | Status |
|--------|------|-----------|-----|--------|
| P3-01 | Integrated gradients computation | 2h | 1 | pending |
| P3-02 | CAV training (matrix properties) | 2h | 1 | pending |
| P3-03 | Interpretable direction discovery (PCA) | 1h | 0 | pending |
| P3-04 | Intervention validation | 3h | 1 | pending |
| P3-05 | Causality analysis | 1h | 0 | pending |

**Phase 3 Total**: ~9 hours of compute

---

## Phase 4 Planned Experiments

| Exp ID | Task | Est. Time | GPU | Status |
|--------|------|-----------|-----|--------|
| P4-01 | SPD manifold setup & validation | 1h | 0 | pending |
| P4-02 | Geodesic landscape recomputation | 2h | 1 | pending |
| P4-03 | Eigenvalue-eigenvector decomposition | 2h | 1 | pending |
| P4-04 | Optimal transport analysis | 2h | 1 | pending |

**Phase 4 Total**: ~7 hours of compute

---

## Phase 5 Planned Experiments

| Exp ID | Task | Est. Time | GPU | Status |
|--------|------|-----------|-----|--------|
| P5-01 | Linear probing (per-layer) | 2h | 1 | pending |
| P5-02 | Gradient flow analysis | 1h | 0 | pending |
| P5-03 | Layer criticality ablation | 2h | 1 | pending |
| P5-04 | Compositionality analysis | 1h | 0 | pending |

**Phase 5 Total**: ~6 hours of compute

---

## Phase 6 Planned Experiments

| Exp ID | Task | Est. Time | GPU | Status |
|--------|------|-----------|-----|--------|
| P6-01 | Intervention robustness testing | 2h | 1 | pending |
| P6-02 | Influence function computation | 3h | 1 | pending |
| P6-03 | Sanity checks | 1h | 0 | pending |
| P6-04 | OOD generalization testing | 2h | 1 | pending |

**Phase 6 Total**: ~8 hours of compute

---

## Phase 7 Planned Experiments

| Exp ID | Task | Est. Time | GPU | Status |
|--------|------|-----------|-----|--------|
| P7-01 | Baseline model (no mining) | 2h | 1 | pending |
| P7-02 | Random mining model | 2h | 1 | pending |
| P7-03 | Adversarial mining model | 2.5h | 1 | pending |
| P7-04 | Mining comparison analysis | 2h | 1 | pending |
| P7-05 | Final synthesis | 2h | 0 | pending |

**Phase 7 Total**: ~10.5 hours of compute

---

## Summary

**Total Estimated Compute**: ~63.5 GPU-hours across all phases
**Optimal Schedule**: 2-3 months (with parallelization and focused iteration)
**Partition**: gpu_test for experiments < 4h, gpu for longer runs
**Early Polling**: Set to 60 seconds after each job submission (catch initialization errors)

## Notes

- Job dependencies are handled via pipeline.json phase tracking
- Experiments can be parallelized within a phase (e.g., multiple P2-0x runs simultaneously)
- Early phases (1-3) are critical path; later phases can begin once dependencies met
- Milestone: Phase 1 + Phase 2 complete = enough data for first draft insights
- Milestone: Phase 1-5 complete = sufficient for comprehensive interpretability report
- Milestone: All phases complete = publishable insights with mining recommendations
