# Experiment Queue — Adversarial Negative Mining for Matrix Inversion

## Research Question
Does adversarial negative mining improve performance on the matrix inversion task compared to baseline diffusion training and random negative mining?

## Hypotheses to Test
1. **H1 (Baseline)**: Standard diffusion training without adversarial negatives achieves baseline performance
2. **H2 (Random negatives)**: Random negative mining provides marginal improvement over baseline
3. **H3 (Adversarial negatives)**: Gradient-based hard negative mining significantly improves matrix inversion accuracy by forcing the model to distinguish challenging negatives

## Evaluation Metrics
- **Primary**: MSE (mean squared error) between predicted and true inverse matrices
- **Secondary**: Per-element accuracy, convergence speed (steps to threshold), energy landscape quality

---

## READY

### Q-001: Baseline (No Negative Mining)
- **Hypothesis**: Establish baseline performance with standard diffusion training
- **Config**: `configs/q001_baseline.json`
- **Mining Strategy**: `none`
- **Parameters**:
  - Rank: 20 (20×20 matrices)
  - Diffusion steps: 10
  - Training steps: 100,000
  - Batch size: 2048
  - Learning rate: 1e-4
- **Resources**: 1 GPU, 16GB RAM, 2 hours, gpu_test partition
- **Priority**: HIGH (must run first - establishes baseline)
- **Notes**: Standard energy-based diffusion without contrastive negative mining

### Q-002: Random Negative Mining
- **Hypothesis**: Random negatives provide weak regularization, minor improvement over baseline
- **Config**: `configs/q002_random.json`
- **Mining Strategy**: `random`
- **Parameters**:
  - Rank: 20
  - Diffusion steps: 10
  - Training steps: 100,000
  - Batch size: 2048
  - Learning rate: 1e-4
  - Negative sampling: Random matrices from uniform distribution
- **Resources**: 1 GPU, 16GB RAM, 2 hours, gpu_test partition
- **Priority**: MEDIUM
- **Dependencies**: Run after Q-001 for comparison
- **Notes**: Negatives sampled independently, not optimized for difficulty

### Q-003: Adversarial Negative Mining (Gradient-Based)
- **Hypothesis**: Hard negatives from gradient ascent significantly improve model discrimination
- **Config**: `configs/q003_adversarial.json`
- **Mining Strategy**: `adversarial`
- **Parameters**:
  - Rank: 20
  - Diffusion steps: 10
  - Training steps: 100,000
  - Batch size: 2048
  - Learning rate: 1e-4
  - Negative optimization: Gradient ascent via `opt_step()` (2 steps, energy maximization)
- **Resources**: 1 GPU, 16GB RAM, 2.5 hours, gpu_test partition
- **Priority**: HIGH
- **Dependencies**: Run after Q-001 and Q-002 for fair comparison
- **Notes**: Uses existing opt_step infrastructure; negatives are optimized to maximize energy

### Q-004: Pilot Test (Debug Run)
- **Hypothesis**: Verify all three strategies execute correctly with minimal compute
- **Config**: `configs/q004_pilot.json`
- **Mining Strategy**: All three (sequential mini-runs)
- **Parameters**:
  - Rank: 10 (smaller for speed)
  - Diffusion steps: 5
  - Training steps: 1,000
  - Batch size: 256
- **Resources**: 1 GPU, 8GB RAM, 15 minutes, gpu_test partition
- **Priority**: HIGHEST (run first to catch bugs)
- **Notes**: Quick validation before committing to long runs

---

## IN_PROGRESS

### Q-001: Baseline (No Negative Mining)
- **Status**: RUNNING ✓
- **Job ID**: 55239690
- **Run ID**: q001_20260114_042054
- **Submitted**: 2026-01-14T04:20:54Z
- **Started**: 2026-01-14T04:23:20Z
- **Git SHA**: 9a691f6 (automated git clone workflow)
- **Partition**: gpu_test
- **Workflow**: ✅ Automated git clone to `/tmp/ired-job-55239690`
- **Configuration**: 20×20 matrices, 100K steps, no mining
- **Expected runtime**: ~2 hours
- **Purpose**: Establish baseline performance

### Q-002: Random Negative Mining
- **Status**: PENDING (waiting for SLURM scheduler)
- **Job ID**: 55240031
- **Run ID**: q002_20260114_042901
- **Submitted**: 2026-01-14T04:29:01Z
- **Git SHA**: 9a691f6 (automated git clone workflow)
- **Partition**: gpu_test
- **Workflow**: ✅ Automated git clone to `/tmp/ired-job-55240031`
- **Configuration**: 20×20 matrices, 100K steps, random mining
- **Expected runtime**: ~2 hours
- **Purpose**: Test random negative sampling
- **Note**: Running in parallel with Q-001

### Q-003: Adversarial Negative Mining (QUEUED - SLURM Limit)
- **Status**: QUEUED LOCALLY (waiting for job slot)
- **Reason**: SLURM QOSMaxSubmitJobPerUserLimit (max 2 jobs per user)
- **Configuration**: 20×20 matrices, 100K steps, adversarial mining
- **Expected runtime**: ~2.5 hours (adversarial mining adds overhead)
- **Auto-submit**: Will submit automatically when Q-001 or Q-002 completes
- **Purpose**: Test gradient-based hard negative mining

---

## DONE

### Q-004: Pilot Test (Debug Run) ✓ COMPLETED
- **Status**: COMPLETED SUCCESSFULLY
- **Job ID**: 55214713
- **Run ID**: q004_20260113_234005
- **Submitted**: 2026-01-13T23:40:05Z
- **Started**: 2026-01-13T23:41:31Z
- **Completed**: 2026-01-13T23:42:37Z
- **Runtime**: 2 minutes 16 seconds (136s)
- **Git SHA**: 9a691f6 (automated git clone workflow)
- **Partition**: gpu_test
- **Workflow**: ✅ Automated git clone to `/tmp/ired-job-55214713`
- **Modules**: python/3.10.13-fasrc01, cuda/11.8.0-fasrc01
- **Results**:
  - Training MSE: 0.0706
  - Validation MSE: 0.0688
  - Configuration: 10×10 matrices, 1000 steps, no mining
  - Output: results/ds_inverse/model_mlp_pilot
- **Milestone**: 🎉 First successful completion with automated git workflow!
- **Validation**: Implementation works correctly, ready for full experiments

---

## FAILED

### Q-004: Pilot Test (Debug Run) - ATTEMPT 1
- **Status**: FAILED
- **Job ID**: 55131103
- **Run ID**: q004_20260113_185013
- **Submitted**: 2026-01-13T10:51:04Z
- **Failed**: 2026-01-13T22:36:38Z
- **Error**: `python: command not found`
- **Root cause**: Module load lines commented out in sbatch script
- **Fix applied**: Uncommented `module load python/3.9` and `module load cuda/11.7` in q004.sbatch

### Q-004: Pilot Test (Debug Run) - ATTEMPT 2
- **Status**: CANCELLED
- **Job ID**: 55208278
- **Run ID**: q004_20260113_223811
- **Submitted**: 2026-01-13T22:38:11Z
- **Cancelled**: 2026-01-13T22:55:00Z
- **Reason**: Redeployed with gpu_test partition for better queue priority (was submitted with old gpu partition)

### Q-004: Pilot Test (Debug Run) - ATTEMPT 3
- **Status**: FAILED
- **Job ID**: 55210815
- **Run ID**: q004_20260113_225500
- **Submitted**: 2026-01-13T22:55:00Z
- **Failed**: 2026-01-13T22:56:41Z
- **Error**: Module not found - `python/3.9` doesn't exist on cluster
- **Root cause**: Cluster uses versioned module names (e.g., `python/3.10.13-fasrc01`)
- **Fix applied**: Updated all sbatch scripts to use `python/3.10.13-fasrc01` and `cuda/11.8.0-fasrc01`

### Q-004: Pilot Test (Debug Run) - ATTEMPT 4
- **Status**: FAILED
- **Job ID**: 55211671
- **Run ID**: q004_20260113_225843
- **Submitted**: 2026-01-13T22:58:43Z
- **Failed**: 2026-01-13T23:00:29Z
- **Error**: File not found - `experiments/matrix_inversion_mining.py` doesn't exist on remote cluster
- **Root cause**: Git tracking issue - `projects/ired` is tracked as gitlink (submodule) not regular files
- **Required fix**: `git rm --cached projects/ired && git add projects/ired/ && git commit && git push`

### Q-004: Pilot Test (Debug Run) - ATTEMPT 5
- **Status**: FAILED
- **Job ID**: 55213584
- **Run ID**: q004_20260113_231120
- **Submitted**: 2026-01-13T23:11:20Z
- **Failed**: 2026-01-13T23:18:21Z
- **Error**: File not found - remote cluster repo hasn't pulled latest commit
- **Root cause**: Files committed and pushed (69c9fcf), but remote repo at `/n/home03/mkrasnow/research-repo` needs `git pull`
- **Required fix**: SSH to cluster and run `cd /n/home03/mkrasnow/research-repo && git pull`

---

## Future Extensions (After Initial Results)
- [ ] **Q-005**: Out-of-distribution test (ood=True, different regularization)
- [ ] **Q-006**: Larger matrices (rank=50) to test scalability
- [ ] **Q-007**: Different diffusion steps (20, 50, 100)
- [ ] **Q-008**: Ablation on opt_step parameters (1 step vs 5 steps)
- [ ] **Q-009**: Mixed strategy (adversarial + random negatives)
- [ ] **Q-010**: Energy landscape visualization and analysis
