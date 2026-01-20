# Debugging — Adversarial Negative Mining for Matrix Inversion

## Active Issues

### Issue 7: Q-001 and Q-002 repeatedly fail with exit code 120 (INVESTIGATING)
**Timestamp**: 2026-01-14T05:00:00Z
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

**Hypothesis**: Resource allocation (4 CPUs or 16G RAM) may be triggering QOS limits or causing jobs to be terminated with exit code 120.

**Status**: INVESTIGATING - Need to test Q-001/Q-002 with reduced resources or different QOS

## Resolved

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
