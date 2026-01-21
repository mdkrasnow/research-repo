# Debugging — IRED Baseline (Official Release)

## Active Issues

### Issue #2: Missing Matplotlib Dependency
- **Job ID**: 56185047
- **Run ID**: q002_20260121_074043
- **Failed At**: 2026-01-21T12:42:39Z
- **Runtime**: 1 minute 56 seconds
- **Exit Code**: 1:0 (Python error)
- **Error Message**: `ModuleNotFoundError: No module named 'matplotlib'`
- **Root Cause**: dataset.py imports matplotlib.pyplot at line 1, but matplotlib was not included in the pip install command in sbatch scripts
- **Analysis**:
  - Environment setup succeeded: Python 3.10.13, CUDA 11.8.0, PyTorch 2.7.1+cu118, einops, numpy, scipy all installed successfully
  - Job failed during Python import phase when run_baseline.py tried to import dataset.py
  - matplotlib import appears to be unused (no plt. calls found in dataset.py - likely leftover from development)
- **Logs**:
  - `slurm/logs/q002_56185047.err`: Contains Python traceback with ModuleNotFoundError
  - `slurm/logs/q002_56185047.out`: Shows successful environment setup up to the point of failure
- **Resolution Plan**: Add matplotlib to pip install in all sbatch scripts (both q001_pilot.sbatch and q002_baseline.sbatch)
- **Status**: IN PROGRESS - awaiting fix implementation

## Resolved

### Issue #1: Git Push Workflow - Commit Not Available on Remote
- **Job ID**: 56162042
- **Run ID**: q001_20260121_160000
- **Failed At**: 2026-01-21T16:07:27Z
- **Runtime**: 3 seconds
- **Exit Code**: 128:0 (git error)
- **Error Message**: `fatal: reference is not a tree: 7770702133ed39020ae4a0424e6b600ec7a10c4b`
- **Root Cause**: Automated SLURM job clones repository and checks out commit by SHA, but commit 7770702 wasn't pushed to remote origin/main before job submission
- **Logs**:
  - `slurm/logs/q001_pilot_56162042.err`: Contains git checkout failure
  - `slurm/logs/q001_pilot_56162042.out`: Shows job initialization sequence
- **Resolution**: Commits a2edff8 (dataset API fix) and 7770702 (result persistence fix) pushed to origin/main. All required code now available on remote.
- **Prevention**: Ensure `git push` completes before submitting SLURM jobs when using automated git checkout workflow
- **Status**: FULLY RESOLVED AND VALIDATED
- **Validation**: Job 56162316 successfully completed (2026-01-21T16:12:40Z) with all commits properly available on remote. Git workflow confirmed operational.

---

## Common Failure Modes (Preemptive Checklist)

- [ ] **Import errors**: Missing pytorch, numpy, or diffusion_lib dependencies
  - Fix: Ensure Python 3.10.13-fasrc01 and required packages installed on cluster

- [ ] **Config format mismatches**: JSON parsing errors or missing required fields
  - Fix: Validate config schema before training

- [ ] **Path errors**: Relative paths not working in SLURM jobs
  - Fix: Use absolute paths or work from repo root

- [ ] **Resource limits**: Out of memory (OOM) or timeout
  - Fix: Adjust batch size, matrix rank, or time limit in sbatch

- [ ] **SLURM module loading**: Missing CUDA or python modules
  - Fix: Update module names to match cluster (e.g., python/3.10.13-fasrc01, cuda/11.8.0-fasrc01)

- [ ] **Data generation**: Random seed not set - non-reproducible results
  - Fix: Ensure seed parameter in dataset initialization

- [ ] **Energy computation**: NaN or Inf values during training
  - Fix: Check energy scale, gradient clipping, and model architecture

---

## Investigation Steps

1. Check SLURM log files: `slurm/logs/<job_name>_<job_id>.{out,err}`
2. Review pipeline.json event log for context
3. Check dataset generation (Inverse class in dataset.py)
4. Validate model energy computation (models.py EBM)
5. Confirm diffusion loop parameters (denoising_diffusion_pytorch_1d.py)
