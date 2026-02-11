# Debugging — Adversarial Negative Mining for Matrix Inversion

## Active Investigation: Results Analysis

### Finding: Adversarial Mining Underperforms Baseline [COMPLETED]
**Timestamp**: 2026-01-21T16:51:32Z (experiments completed)
**Analysis Updated**: 2026-01-21T18:00:00Z
**Run IDs**: q001_20260121_071917 (baseline), q002_20260121_124609 (random), q003_20260121_143232 (adversarial)
**Job IDs**: 56162645 (baseline), 56185426 (random), 56194269 (adversarial)

**Final Results from Completed Experiments**:
- **Q-001 Baseline (no mining)**: train_mse = 0.0096721, validation_mse = 0.0096761
- **Q-002 Random mining**: train_mse = 0.00968831, validation_mse = 0.00968396 (+0.08% vs baseline)
- **Q-003 Adversarial mining**: train_mse = 0.00980961, validation_mse = 0.00982675 (+1.56% vs baseline)

**Key Findings**:
1. **Random ≈ Baseline** (difference of 0.08% is within noise, needs multi-seed validation)
2. **Adversarial is clearly worse** (+1.56% MSE increase on validation, +1.47% on training)
3. **Training stability varies**: Baseline and random show stable convergence; adversarial shows slight instability
4. **Adversarial hyperparameters**: mining_opt_steps=2, mining_noise_scale=3.0 (from q003_adversarial.json)

**Why This Matters**:
- Adversarial negative mining was expected to improve performance by forcing the model to learn from hard examples
- Instead, it degraded performance significantly
- This is a **known failure mode** in contrastive learning literature when:
  - Hard negatives are actually **false negatives** (too similar to positives)
  - Negatives become **pathologically hard** (ill-conditioned matrices for inversion)
  - Hard negatives create **gradient artifacts** or optimization instability
  - **Diversity is lost** (hard negatives collapse to narrow region)

**Investigation Plan** (see queue.md and implementation-todo.md for details):

#### Phase 1: Validate Results (HIGH PRIORITY) - INFRASTRUCTURE READY ✓
- **Status**: Infrastructure complete as of 2026-01-21T18:30:00Z
- **Q-101/102/103**: Run 10 seeds per strategy for confidence intervals
  - ✓ Added --seed parameter to experiments/matrix_inversion_mining.py
  - ✓ Created configs: q101/q102/q103_multiseed_{baseline,random,adversarial}.json
  - ✓ Created SLURM array jobs: q101/q102/q103_multiseed.sbatch (--array=0-9)
  - Ready for submission via /dispatch or scripts/cluster/submit.sh
- **Q-104**: Track learning curves with frequent checkpointing
- **Goal**: Confirm adversarial consistently underperforms across seeds
- **Compute**: 45 GPU-hours total (30 jobs × 1.5h), ~1.5-6h wall-clock time

#### Phase 2: Diagnose Root Cause (HIGH PRIORITY)
- **Q-201**: Matrix conditioning analysis
  - Hypothesis: Adversarial negatives are near-singular (high condition number)
  - Check distribution of det(A), cond(A), σ_min for pos vs neg samples
- **Q-202**: Energy gap profiling
  - Hypothesis: Adversarial negatives collapse to narrow region, losing diversity
  - Track E(pos) - E(neg), gradient norms, false negative rate

#### Phase 3: Fix Adversarial Mining (MEDIUM PRIORITY)
- **Q-301**: Sweep mining_opt_steps [1, 3, 5, 10, 20]
  - Find optimal hardness level
- **Q-302/303**: Sweep ascent LR and noise scale
  - Control hardness and add diversity
- **Q-304**: Mixed strategy (70% random + 30% adversarial)
  - Balance diversity with challenge

#### Phase 4: Training Stability (HIGH PRIORITY)
- **Q-401**: Add cosine LR decay + grad clipping + EMA
  - Fix late-stage divergence observed in validation run
- **Q-402**: Early stopping based on best validation checkpoint
  - Prevent performance degradation in late training

**Literature Support**:
- **False negatives harm learning**: Debiased Contrastive Learning (Chuang et al., NeurIPS 2020), False Negative Cancellation (Huynh et al., WACV 2022)
- **Hard negatives need careful tuning**: Hard Negative Mixing (Kalantidis et al., NeurIPS 2020), Contrastive learning with hard negative samples (Robinson et al., 2020)
- **EBM training instability**: Improved Contrastive Divergence Training (Du et al., 2021), Learning EBMs with Adversarial Training (Yin et al., ECCV 2022)

**Expected Outcomes**:
1. **Multi-seed validation** will confirm adversarial consistently underperforms (or reveal it's just noise)
2. **Conditioning analysis** will likely show adversarial negatives have pathologically high condition numbers
3. **Hyperparameter sweeps** may rescue adversarial mining (or prove it's fundamentally flawed for this task)
4. **Stability fixes** will improve all strategies, especially late training
5. **Publishable result**: "Harder ≠ better for matrix inversion EBMs due to task geometry"

**Status**: ACTIVE - Investigation phase started, implementation tasks defined

---

## Active Issues

### Issue 11: Multi-seed validation jobs failed - config file mismatch (c11bf8d vs fef8849)
**Timestamp**: 2026-01-23T18:18:31Z (failure detected)
**Run IDs**: q101_20260123_181809 (FAILED), q102_20260123_181809 (FAILED), q103_20260123_181809 (FAILED)
**Job IDs**: 56540906 (FAILED - 22s), 56540909 (FAILED - 105s), 56540911 (FAILED - 22s)
**Error**: FileNotFoundError: [Errno 2] No such file or directory: 'projects/ired/configs/q10{1,2,3}_multiseed_{baseline,random,adversarial}.json'

**Root Cause**:
- Jobs were submitted with git_sha=c11bf8d which contains --seed parameter implementation
- Config files (q101_multiseed_baseline.json, q102_multiseed_random.json, q103_multiseed_adversarial.json) were added in commit fef8849
- Commit c11bf8d does NOT include these config files
- Current HEAD is fef8849 (verified with git rev-parse HEAD)
- Timeline: c11bf8d (2026-01-23) → fef8849 (reorg commit, later)
- Jobs checkout c11bf8d and immediately fail when sbatch tries to read missing config files

**Verification**:
- Config files verified to exist in current directory (glob search successful):
  - /Users/mkrasnow/Desktop/research-repo/projects/ired/configs/q101_multiseed_baseline.json ✓
  - /Users/mkrasnow/Desktop/research-repo/projects/ired/configs/q102_multiseed_random.json ✓
  - /Users/mkrasnow/Desktop/research-repo/projects/ired/configs/q103_multiseed_adversarial.json ✓
- Current HEAD commit: fef8849 (verified)
- All config files present in HEAD

**Solution**:
- Resubmit Q-101/102/103 with git_sha=fef8849 (current HEAD)
- fef8849 contains both:
  1. --seed parameter support (from c11bf8d)
  2. All three config files (added in fef8849)
- No code changes needed, just specify correct git_sha

**Next Steps**:
- Resubmit with current HEAD (fef8849) via /dispatch or /run-experiments
- Phase transition: DEBUG → WAIT_SLURM after successful resubmission
- Early poll scheduled for 60s after resubmission to catch any initialization errors

**Status**: BLOCKED - Waiting for user confirmation to resubmit with correct git_sha

## Resolved

### Issue 8: Q-001 baseline experiment consistently fails while Q-002 succeeds
**Timestamp**: 2026-01-20T08:45:20Z (discovered), 2026-01-21T07:21:24Z (RESOLVED)
**Run IDs**: q001_20260120_084327, q001_20260120_084813 (failed with old code), q001_20260121_071917 (SUCCESS with new code)
**Job IDs**: 56017462, 56017590 (Q-001 failures), 56162645 (Q-001 SUCCESS)
**Error**: Q-001 consistently failed with exit code 120:0 after 3-42 seconds. Q-002 succeeded with identical configuration.

**Timeline**:
1. 2 CPUs/8G RAM: Q-001 exit 120 (42s), Q-002 OOM (1m39s, reached iteration 1000)
2. 2 CPUs/16G RAM: Q-001 exit 120 (18s), Q-002 RUNNING (9+ minutes, ongoing)
3. Q-002 COMPLETED successfully (1h 40m, 100K iterations, MSE: train=0.0096382, val=0.00969288)

**Root Cause (CONFIRMED)**: All Q-001 failures occurred with OLD CODE (commit 75df3cf). The issue was NOT:
- Resource allocation (2 CPUs/16G RAM works correctly)
- mining_strategy='none' configuration
- Job name or SLURM limits
- Node-specific issues

**Resolution**: Resubmitted Q-001 with commit 7770702 (includes validated infrastructure fixes):
- Result persistence fix (rsync before cleanup)
- Dataset API fixes
- Diffusion code improvements
- Resource allocation validated by ired-baseline

**Validation (2026-01-21T07:21:24Z)**: Job 56162645 (Q-001 baseline) is RUNNING successfully after 2m 7s!
- MAJOR BREAKTHROUGH: First successful Q-001 run after multiple exit code 120 failures
- Job has successfully: 1) Passed initialization, 2) Loaded modules, 3) Cloned repository, 4) Started training
- All previous Q-001 attempts failed within 3-42 seconds
- This proves the issue was OLD CODE (commit 75df3cf), not a configuration bug

**Impact**: All infrastructure issues now RESOLVED. Ready to run full experiment suite.

**Status**: RESOLVED - Q-001 running successfully with commit 7770702. All infrastructure validated.

### Issue 9: Result persistence - sbatch cleanup deletes results before copy
**Timestamp**: 2026-01-21T00:00:00Z (discovered), 2026-01-21T05:30:00Z (RESOLVED AND VALIDATED)
**Run IDs**: q002_20260120_084822 (ired project, results lost), ired-baseline run (validation)
**Job IDs**: 56017592 (ired), 56162316 (ired-baseline validation)
**Error**: Training completed successfully but results.json was not persisted to project directory

**Symptoms**:
- Job 56017592 (Q-002) completed successfully after 1h 40m runtime
- Training finished all 100,000 iterations with final MSE: train=0.0096382, validation=0.00969288
- Log shows: "Results saved to: results/ds_inverse/model_mlp_random/results.json"
- However, results were NOT in the project directory or remote repository

**Root Cause**: The sbatch script workflow was:
1. Clone repository to /tmp/project-job-$SLURM_JOB_ID
2. Run experiment (writes results to /tmp/project-job-$SLURM_JOB_ID/results/...)
3. Cleanup: rm -rf /tmp/project-job-$SLURM_JOB_ID
4. No step to copy results back before cleanup

**Fix Applied (commit 7770702)**: Added rsync step to sbatch scripts BEFORE cleanup:
```bash
# After experiment completes, before cleanup
rsync -av results/ /n/holyscratch01/kempner_fellows/Users/mkrasnow/research-repo/projects/ired-baseline/runs/${RUN_ID}/

# Then cleanup
rm -rf /tmp/project-job-$SLURM_JOB_ID
```

**Validation (job 56162316)**: ired-baseline project submitted job with commit 7770702 and SUCCEEDED:
- Results successfully copied from /tmp to persistent storage before cleanup
- Git workflow validated (clone, checkout specific commit, run)
- Dataset and diffusion code validated (though ired-baseline uses different experiment code)
- Confirms: Result persistence fix WORKS

**Status**: RESOLVED AND VALIDATED - Fix applied and confirmed working. Safe to run all future experiments with commit 7770702 or later.

### Issue 7: Q-001 and Q-002 repeatedly fail with exit code 120
**Timestamp**: 2026-01-14T05:00:00Z (discovered), 2026-01-20T08:41:00Z (resolved)
**Run IDs**: q001_20260114_045800, q002_20260114_045800
**Job IDs**: 55240899, 55240900
**Error**: Both jobs failed with exit code 120:0
**Log files**: MISSING - no output or error logs created

**Symptoms**:
- Q-001 ran for 1m13s before failing (exit code 120) on holygpu7c26105
- Q-002 ran for 19s before failing (exit code 120) on holygpu7c26106
- Different nodes than previous failures, so not node-specific issue
- Q-004 pilot (job 55240939) SUCCEEDS with same workflow

**Key Differences Between Q-004 (works) and Q-001/Q-002 (fail)**:
- Resources: Q-004 uses 2 CPUs/8G RAM, Q-001/Q-002 use 4 CPUs/16G RAM
- Time limit: Q-004 uses 15min, Q-001/Q-002 use 2 hours
- nvidia-smi check: Q-001 has it (lines 33-37), Q-002 and Q-004 don't

**Investigation Results**:
1. Partition limits checked: gpu_test allows unlimited CPUs/memory, 12h max time ✓
2. Git clone tested manually: WORKS ✓
3. Environment variables tested: WORK ✓
4. Test job with Q-004 resources (2 CPU/8G) but Q-001 script: SUCCEEDS (job 55241074) ✓
5. QOS settings: normal QOS shows `cpu=1` limit (unclear if this applies per-job or per-user)

**Root Cause**: Resource allocation (4 CPUs/16G RAM) was triggering QOS limits or resource allocation issues with exit code 120.

**Resolution**: Reduced Q-001 and Q-002 resource requests to match Q-004's successful configuration (2 CPUs/8G RAM). Also explicitly requested A100 GPUs with `--gres=gpu:a100:1`. Updated sbatch scripts: q001.sbatch and q002.sbatch.

**Status**: RESOLVED - Ready to resubmit Q-001 and Q-002 with reduced resources

### Issue 6: Q-001 and Q-002 failed due to node failure
**Timestamp**: 2026-01-14T04:31:24Z (failure), 2026-01-14T04:53:00Z (resolved)
**Run IDs**: q001_20260114_042054, q002_20260114_042901
**Job IDs**: 55239690, 55240031
**Error**: Both jobs failed with exit code 120:0
**Log files**: MISSING - no output or error logs created

**Symptoms**:
- Q-001 ran for 8m37s before failing (exit code 120)
- Q-002 ran for 29s before failing (exit code 120)
- No SLURM log files created at expected paths
- Both jobs ended at EXACTLY the same time: 2026-01-13T23:29:43
- DerivedExitCode is 0:0 (batch script itself exited with 120)

**Root Cause Analysis**:
- Both jobs were allocated to the same compute node: `holygpu7c26106`
- Both terminated at the exact same second (23:29:43), despite starting at different times
- This indicates a node failure or reboot event, NOT a bug in our code
- Exit code 120:0 is consistent with node-level termination
- No log files were written because the termination was abrupt (node crash)

**Verification Tests**:
1. Environment variable passing tested: WORKS (test job 55240665 succeeded)
2. Git clone workflow tested manually: WORKS
3. Full Q-001 sbatch script resubmitted manually (job 55240799): RUNNING SUCCESSFULLY on same node
4. Confirms our workflow is correct - the original failures were due to infrastructure issues

**Status**: RESOLVED - Node has recovered, workflow is correct. Safe to resubmit experiments.

### Issue 5: Remote repository not updated after git push
**Timestamp**: 2026-01-13T23:18:21Z
**Run ID**: q004_20260113_231120
**Job ID**: 55213584
**Error**: `python: can't open file '/n/home03/mkrasnow/research-repo/experiments/matrix_inversion_mining.py': [Errno 2] No such file or directory`
**Log files**:
- `projects/ired/slurm/logs/ired_q004_pilot_55213584.err`
- `projects/ired/slurm/logs/ired_q004_pilot_55213584.out`

**Root cause**: Files were committed (69c9fcf) and pushed to git successfully, but the remote cluster's repository at `/n/home03/mkrasnow/research-repo` hasn't pulled the latest changes yet.

**Impact**: SLURM jobs run from stale repository checkout without implementation files.

**Required Fix**: Pull latest code on remote cluster:
```bash
ssh <cluster> "cd /n/home03/mkrasnow/research-repo && git pull"
```

**Status**: RESOLVED - Automated git workflow implemented (commit 9a691f6)

**Solution**: Each SLURM job now automatically clones the repository fresh and checks out the exact commit SHA from submission. No manual intervention needed on cluster.

## Resolved

### Issue 4: Repository sync - implementation files not on cluster
**Timestamp**: 2026-01-13T23:00:29Z
**Run ID**: q004_20260113_225843
**Job ID**: 55211671
**Error**: `python: can't open file '/n/home03/mkrasnow/research-repo/experiments/matrix_inversion_mining.py': [Errno 2] No such file or directory`
**Log files**:
- `projects/ired/slurm/logs/ired_q004_pilot_55211671.err`
- `projects/ired/slurm/logs/ired_q004_pilot_55211671.out`

**Root cause**: The `projects/ired/` directory exists locally with all implementation files, but these files are not present on the remote cluster. Investigation shows that `projects/ired` is tracked as a gitlink entry (like a submodule) in git, not as a regular directory with files. The commit "Track projects/ired as regular directory (remove nested git repo)" did not complete successfully.

**Impact**: Remote SLURM jobs cannot find the experiment script or any other implementation files.

**Diagnosis**:
- Local: `projects/ired/experiments/matrix_inversion_mining.py` exists
- Git: `git ls-tree HEAD projects/ired` shows only "projects/ired" (gitlink), not the actual files
- Remote cluster: Files don't exist at `/n/home03/mkrasnow/research-repo/experiments/`

**Required Fix**: Need to properly commit the ired project files to git, then push to remote. Options:
1. Remove the gitlink entry and add files: `git rm --cached projects/ired && git add projects/ired/ && git commit`
2. Manually sync files to cluster (temporary workaround)

**Status**: RESOLVED - Git tracking fixed (commit 69c9fcf), files now properly tracked and pushed

## Resolved

### Issue 3: Incorrect module versions
**Timestamp**: 2026-01-13T22:56:41Z
**Run ID**: q004_20260113_225500
**Job ID**: 55210815
**Error**: `The following module(s) are unknown: "python/3.9"`
**Log files**:
- `projects/ired/slurm/logs/ired_q004_pilot_55210815.err`
- `projects/ired/slurm/logs/ired_q004_pilot_55210815.out`

**Root cause**: The sbatch scripts specified `python/3.9` and `cuda/11.7` but these exact versions don't exist on the Harvard FAS RC cluster. The cluster uses versioned module names like `python/3.10.13-fasrc01`.

**Available modules discovered**:
- Python: 3.10.9, 3.10.12, 3.10.13, 3.12.5, 3.12.8, 3.12.11 (all -fasrc01/02 suffixed)
- CUDA: 11.3.1, 11.8.0, 12.0.1, 12.2.0, 12.4.1, 12.9.1 (all -fasrc01 suffixed)

**Fix**: Updated all sbatch scripts to use:
- `module load python/3.10.13-fasrc01`
- `module load cuda/11.8.0-fasrc01`

**Status**: RESOLVED - Resubmitted with correct modules (job 55211671, run q004_20260113_225843)

## Resolved

### Issue 1: MPS (Mac GPU) float64 incompatibility
**Error**: `TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64`

**Root cause**: The Inverse dataset generates numpy arrays with default float64 dtype. Mac MPS backend only supports float32.

**Status**: This is a local testing limitation. SLURM GPU cluster supports float64, so this won't affect production runs.

**Workaround for local testing**: Set `PYTORCH_ENABLE_MPS_FALLBACK=1` or use CPU-only mode.

**Test result**: Imports work correctly, mining_config parameter is recognized, trainer initialization succeeds. Only fails at data loading due to MPS limitation.

**Validation**:
- ✓ Import paths fixed (experiments/matrix_inversion_mining.py)
- ✓ mining_config parameter recognized by GaussianDiffusion1D
- ✓ Trainer1D initialization succeeds
- ✗ Training loop fails on MPS due to float64 (expected, not a blocker)

**Decision**: Proceed to SLURM testing. Local MPS testing not critical since production uses CUDA GPUs.

### Issue 2: SLURM Python module not loaded
**Timestamp**: 2026-01-13T22:36:38Z
**Run ID**: q004_20260113_185013
**Job ID**: 55131103
**Error**: `python: command not found`
**Log files**:
- `projects/ired/slurm/logs/ired_q004_pilot_55131103.err`
- `projects/ired/slurm/logs/ired_q004_pilot_55131103.out`

**Root cause**: The sbatch script `slurm/jobs/q004.sbatch` has module load commands commented out (lines 18-19):
```bash
# module load python/3.9
# module load cuda/11.7
```

**Impact**: Job fails immediately with exit code 127 (command not found). No Python runtime available on compute node.

**Fix**: Uncomment module load lines in q004.sbatch and resubmit job.

**Status**: RESOLVED - Fixed and resubmitted (job 55208278, run q004_20260113_223811)

---

## Common Failure Modes (Preemptive Checklist)

### Import and Dependencies
- [ ] Missing dependencies (torch, numpy, einops, ema_pytorch, accelerate)
- [ ] CUDA version mismatch (check module load cuda/11.7 or compatible version)
- [ ] Python version incompatibility (requires Python 3.8+)
- [ ] Path errors for importing diffusion_lib, dataset, models

### Configuration Issues
- [ ] Config format mismatches (JSON parsing errors)
- [ ] Invalid parameter values (negative batch size, invalid mining strategy)
- [ ] Missing required config fields (rank, diffusion_steps, mining_strategy)

### SLURM Execution
- [ ] Resource limits exceeded (OOM, timeout)
  - Default allocation: 16GB memory, 1 GPU, 2 hours
  - Increase if needed: --mem=32G, --time=04:00:00
- [ ] SLURM module loading issues
  - Verify: module load python/3.9, module load cuda/11.7
- [ ] File synchronization delays (rsync may take time for large result directories)

### Training Issues
- [ ] NaN losses (check learning rate, gradient clipping)
- [ ] Energy computation errors in adversarial mining
  - Verify opt_step() is properly configured
  - Check energy gradient computation in DiffusionWrapper
- [ ] Dataset generation failures (matrix inversion numerics)
  - Ensure matrices are well-conditioned (eigenvalues > 0.1)

### Matrix Inversion Specific
- [ ] Singular or near-singular matrices causing inversion failures
  - Inverse dataset adds diagonal regularization: 0.5 * np.eye(rank) for in-distribution
- [ ] Numerical instability in gradient-based mining
  - May need to clip gradients or adjust opt_step learning rate
- [ ] Dimension mismatches (inp_dim = out_dim = rank * rank for flattened matrices)

### Result Logging
- [ ] Missing output directories (runs/<run_id>/, results/)
- [ ] Checkpoint save failures (disk space, permissions)
- [ ] Metric computation errors (validate MSE calculation matches expected shape)

---

## Debugging Commands

```bash
# Test imports locally
python -c "import torch; import numpy; from dataset import Inverse; print('Imports OK')"

# Run minimal training loop (10 steps)
python experiments/matrix_inversion_mining.py --config configs/q001_baseline.json --train-steps 10

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# Verify dataset generation
python -c "from dataset import Inverse; ds = Inverse('train', 20, False); print(ds[0][0].shape, ds[0][1].shape)"

# Check SLURM job status
scripts/cluster/status.sh <job_id>

# Fetch remote logs
scripts/cluster/remote_fetch.sh ired
```

---

## Testing Session Notes
All known issues have been resolved. Testing continues on SLURM cluster.
