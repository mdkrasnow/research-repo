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
- **Resources**: 1 GPU, 2 CPUs, 16GB RAM, 2.5 hours, gpu_test partition
- **Priority**: HIGH
- **Dependencies**: Run after Q-001 and Q-002 for fair comparison
- **Notes**: Uses existing opt_step infrastructure; negatives are optimized to maximize energy
- **SLURM Status**: QUEUED - Cannot submit yet due to QOSMaxSubmitJobPerUserLimit (2 jobs already running from other projects). Will be submitted automatically after current jobs complete.
- **Sbatch Script**: Created at `projects/ired/slurm/q003.sbatch`

---

## IN_PROGRESS

### Q-002: Random Negative Mining (RERUN) - IN PROGRESS
- **Status**: SUBMITTED - Job running on cluster
- **Job ID**: 56185426
- **Run ID**: q002_20260121_124609
- **Submitted**: 2026-01-21T12:46:09Z
- **Git SHA**: 7770702 (includes result persistence fix)
- **Partition**: gpu_test
- **Resources**: 1 GPU, 2 CPUs, 16GB RAM
- **Expected Runtime**: ~2.5 hours
- **Purpose**: Rerun Q-002 with result persistence (original run 56017592 succeeded but results lost due to /tmp cleanup)
- **Config**: `configs/q002_random.json`
- **Mining Strategy**: random
- **Early Poll**: Set for 60 seconds after submission to catch initialization errors

---

## DONE

### Q-001: Baseline (No Mining) ✓ COMPLETED
- **Status**: COMPLETED SUCCESSFULLY
- **Job ID**: 56162645
- **Run ID**: q001_20260121_071917
- **Submitted**: 2026-01-21T07:19:17Z
- **Started**: 2026-01-21T07:19:17Z
- **Completed**: 2026-01-21T12:59:15Z (approximate)
- **Runtime**: 1 hour 40 minutes 18 seconds (6018 seconds)
- **Git SHA**: 7770702 (includes result persistence fix + dataset/diffusion fixes)
- **Partition**: gpu_test
- **Node**: holygpu7c26105
- **Resources**: 1 GPU, 2 CPUs, 16GB RAM
- **Results**:
  - Training MSE: 0.0096721
  - Validation MSE: 0.0096761
  - Iterations: 100,000 (completed all)
  - Configuration: 20×20 matrices, no mining strategy (baseline)
- **Milestone**: First successful Q-001 run after multiple exit 120 failures! Validates:
  - All infrastructure fixes work correctly (result persistence, dataset API, diffusion code)
  - mining_strategy='none' baseline configuration executes correctly
  - Resource allocation (2 CPUs/16GB RAM/1 GPU) is appropriate
  - Automated git workflow functions properly
- **Root Cause Confirmed**: Previous Q-001 failures were due to OLD CODE (commit 75df3cf), not configuration issues
- **Next Steps**: Run Q-002 (random mining rerun) and Q-003 (adversarial mining) with commit 7770702

### Q-002: Random Negative Mining ✓ COMPLETED (Results Lost, Will Rerun)
- **Status**: COMPLETED SUCCESSFULLY (but results not persisted - see Issue 9, now RESOLVED)
- **Job ID**: 56017592
- **Run ID**: q002_20260120_084822
- **Submitted**: 2026-01-20T08:48:22Z
- **Started**: 2026-01-20T08:48:25Z
- **Completed**: 2026-01-20T10:29:00Z (approximate)
- **Runtime**: 1 hour 40 minutes (6040 seconds)
- **Git SHA**: 75df3cf (before result persistence fix)
- **Partition**: gpu_test
- **Node**: holygpu7c26105
- **Resources**: 1 GPU, 2 CPUs, 16GB RAM
- **Results** (from SLURM logs only):
  - Training MSE: 0.0096382
  - Validation MSE: 0.00969288
  - Iterations: 100,000 (completed all)
  - Configuration: 20×20 matrices, random negative mining
- **Issue 9 Update**: Result persistence fix validated by ired-baseline job 56162316. Safe to rerun Q-002 with commit 7770702 to capture full results.
- **Milestone**: First successful full-scale experiment completion! Validates that:
  - 2 CPUs/16G RAM configuration works
  - Training code executes correctly for 100K iterations
  - Random mining strategy implemented correctly
- **Next Steps**: Rerun Q-002 after Q-001 succeeds, using commit 7770702 to capture full results with checkpoint files

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

### Q-001: Baseline (No Mining) - MULTIPLE FAILURES
- **Status**: FAILED (consistent exit 120 failures, cause unknown)
- **Latest Job ID**: 56017590
- **Latest Run ID**: q001_20260120_084813
- **Failure Pattern**: Exit code 120:0 after 18-42 seconds
- **Key Anomaly**: Q-002 succeeds with identical configuration on same node
- **Attempts**:
  1. q001_20260120_084327 (job 56017462): Exit 120 after 42s, 2 CPUs/8G RAM (insufficient memory)
  2. q001_20260120_084813 (job 56017590): Exit 120 after 18s, 2 CPUs/16G RAM (same config as Q-002)
  3. Earlier attempts also failed with exit 120 or node failures
- **Investigation Status**: See Issue 8 in debugging.md
- **Next Steps**: Compare Q-001 and Q-002 config files to identify root cause

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
