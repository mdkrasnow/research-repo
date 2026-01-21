# Experiment Queue

## Status: RUNNING
Current priority: Monitor EXP-001 early poll at 12:56:27Z

## Running
- **EXP-001**: Hessian Eigenspectrum Analysis (RESUBMITTED)
  - Job ID: 56186252
  - Run ID: exp001_20260121_125527
  - Submitted: 2026-01-21T12:55:27Z
  - Config: Matrix inverse, rank 2, 20 samples, annealing levels [1,3,5,7,10]
  - Resources: gpu_test partition, 1 A100 GPU, 4h limit
  - Next poll: 2026-01-21T12:56:27Z (early poll to catch init errors)

## Ready to Run
None

## Waiting for Resources
None

## Blocked
None

## Completed

### Failed Runs
- **EXP-001** (exp001_20260121_074154): Job 56185132
  - Failed: 2026-01-21T12:41:58Z (4 seconds after submission)
  - Cause: SLURM log path resolution error (relative path in #SBATCH directive)
  - Exit code: 1:0
  - Fixed: Changed to absolute path 'projects/ired-interp/slurm/logs/...'
  - Resubmitted: exp001_20260121_125527 (Job 56186252)
  - See debugging.md Issue #1 for details

---

## Planned Experiments

### Phase 1: Energy Landscape Analysis

#### EXP-001: Hessian Eigenspectrum - Matrix Completion
**Status**: PLANNED
**Objective**: Compute Hessian eigenspectrum for matrix completion task across different ranks
**Config**:
- Task: Matrix completion
- Ranks: [1, 2, 5, 10]
- Matrix size: 20x20
- Samples: 100 problems
- Annealing levels: k=1,5,10
**Resources**: 1 A100 GPU, ~4 hours
**Output**: Eigenvalue distributions, eigenvector analysis

#### EXP-002: Energy Landscape Visualization
**Status**: PLANNED
**Objective**: Visualize 2D slices of energy landscape
**Config**:
- Task: Matrix completion (rank 2)
- Directions: rank-preserving, rank-increasing, random
- Grid resolution: 100x100
- Annealing levels: k=1,5,10
**Resources**: 1 A100 GPU, ~2 hours
**Output**: Energy contour plots, gradient field visualizations

#### EXP-003: Sparse Autoencoder Training
**Status**: PLANNED
**Objective**: Train SAE on gradient field to discover features
**Config**:
- Gradient samples: 10,000
- Hidden dimension: 512
- Sparsity coefficient: 0.01
- Training epochs: 100
**Resources**: 1 A100 GPU, ~8 hours
**Output**: Trained SAE model, feature visualizations

### Phase 2: Riemannian Geometry

#### EXP-004: Grassmannian Distance Correlation
**Status**: PLANNED
**Objective**: Measure correlation between Grassmannian distance and energy
**Config**:
- Task: Matrix completion (rank 2, 5)
- Perturbation levels: [0.1, 0.5, 1.0, 2.0]
- Samples: 500 problems
**Resources**: CPU, ~1 hour
**Output**: Correlation plots, statistical analysis

#### EXP-005: Tangent/Normal Decomposition
**Status**: PLANNED
**Objective**: Analyze energy gradient components (tangent vs normal to manifold)
**Config**:
- Task: Matrix completion (rank 2)
- Samples: 200 problems
- Annealing levels: all k=1..10
**Resources**: 1 A100 GPU, ~3 hours
**Output**: Curvature ratio analysis, geometric alignment metrics

### Phase 3: Score Function & Implicit Bias

#### EXP-006: Score Function SVD Analysis
**Status**: PLANNED
**Objective**: SVD of gradient field, check alignment with problem geometry
**Config**:
- Gradient samples: 5,000
- Tasks: completion, inversion
**Resources**: CPU, ~30 minutes
**Output**: Principal component analysis, alignment scores

#### EXP-007: Implicit Bias Tracking
**Status**: PLANNED
**Objective**: Track effective rank during IRED training
**Config**:
- Train from scratch on matrix completion
- Checkpoints: every 1000 steps
- Total steps: 50,000
**Resources**: 1 A100 GPU, ~12 hours
**Output**: Rank evolution plots, comparison to baseline

### Phase 4: NTK & Circuits

#### EXP-008: Neural Tangent Kernel Analysis
**Status**: PLANNED
**Objective**: Compute empirical NTK, analyze alignment
**Config**:
- Samples: 100 problems
- Use trace estimation (Hutch++)
**Resources**: 1 A100 GPU, ~6 hours
**Output**: NTK matrices, alignment analysis

#### EXP-009: Activation Patching - Circuit Discovery
**Status**: PLANNED
**Objective**: Identify critical computational components
**Config**:
- Ablate each layer systematically
- Tasks: completion, inversion
- Samples: 100 problems per ablation
**Resources**: 1 A100 GPU, ~10 hours
**Output**: Causal importance scores, circuit diagrams

### Comprehensive Analysis

#### EXP-010: Full Annealing Analysis
**Status**: PLANNED
**Objective**: Comprehensive analysis across all K=10 annealing levels
**Config**:
- All analysis methods (Hessian, landscape, SAE, geometry)
- Detailed tracking of coarse-to-fine emergence
**Resources**: 2 A100 GPUs, ~24 hours
**Output**: Full interpretability report

---

## Notes
- Experiments will be submitted to cluster via `scripts/cluster/remote_submit.sh`
- Each experiment generates run folder: `runs/<exp_id>_<timestamp>/`
- Results tracked in `results/<exp_id>/`
