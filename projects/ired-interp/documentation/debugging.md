# Debugging Log

## Active Issues

### Issue #1: Project Directory Not Committed to Git
**Date**: 2026-01-21T12:48:01Z (First discovered), 2026-01-21T12:57:05Z (Root cause identified)
**Job IDs**: 56185132 (first failure), 56186252 (second failure)
**Run IDs**: exp001_20260121_074154, exp001_20260121_125527
**Status**: CRITICAL - BLOCKING ALL EXPERIMENTS

**Error Progression**:
1. **First failure (56185132)**: Job failed during SLURM initialization with exit code 1:0 after 4 seconds. No log files generated.
2. **Second failure (56186252)**: Job failed at `cd projects/ired-interp` with exit code 1:0 after 2 seconds. Error: `cd: projects/ired-interp: No such file or directory`

**Root Cause**:
The entire `projects/ired-interp/` directory is **not committed to git**. It exists only in the local working directory as untracked files. When SLURM jobs clone the research repository and checkout commit `7770702`, the `projects/ired-interp/` directory does not exist in that commit, causing the `cd projects/ired-interp` command to fail.

**Evidence from logs** (`slurm/logs/exp001_56186252.err`):
```
Cloning into 'repo'...
Note: switching to '7770702133ed39020ae4a0424e6b600ec7a10c4b'.
...
HEAD is now at 7770702 Fix critical result persistence bug in SLURM jobs
/var/slurmd/spool/slurmd/job56186252/slurm_script: line 56: cd: projects/ired-interp: No such file or directory
```

**Git status verification**:
```bash
$ git status projects/ired-interp/
Untracked files:
  projects/ired-interp/
```

**Why first failure was misdiagnosed**:
The first failure appeared to be a SLURM log path issue because no logs were generated. However, this was actually because the job failed so early (before any output) that SLURM output redirection hadn't started yet. After fixing the log paths (making them relative to repo root), logs were successfully created, revealing the true issue: the project directory doesn't exist in the git repository.

**Fix**: Commit the entire `projects/ired-interp/` directory to git:
```bash
git add projects/ired-interp/
git commit -m "Add ired-interp project: interpretability analysis framework"
git push origin main
```

**Next Steps**:
1. Commit all ired-interp project files to git
2. Verify commit is pushed to remote
3. Resubmit EXP-001 (it will clone the updated repository)
4. Verify job succeeds in early poll

**Critical Lesson**: ALWAYS verify that project directories are committed to git before submitting SLURM jobs. The automated git workflow clones fresh on each job, so untracked files are invisible to cluster jobs.

---

---

## Resolved Issues
None yet

---

## Common Issues & Solutions

### Environment Setup

#### Issue: Geomstats installation fails
**Solution**: Install via conda instead of pip:
```bash
conda install -c conda-forge geomstats
```

#### Issue: PyHessian CUDA compatibility
**Solution**: Ensure PyTorch CUDA version matches cluster CUDA 11.8.0-fasrc01

### IRED Model Issues

#### Issue: Energy computation requires gradient
**Error**: `RuntimeError: grad can be implicitly created only for scalar outputs`
**Solution**: Ensure energy output is scalar (batch-wise sum if needed)

#### Issue: Hessian computation OOM
**Error**: `CUDA out of memory`
**Solution**:
- Use Lanczos iteration instead of full Hessian
- Reduce batch size
- Use CPU for largest matrices

### Geomstats Issues

#### Issue: Grassmannian point representation
**Solution**: Use projection matrix representation P where P² = P, P^T = P

#### Issue: Geodesic computation fails
**Solution**: Check that points are properly projected onto manifold before geodesic computation

### Data Pipeline Issues

#### Issue: Matrix rank mismatch
**Solution**: Verify SVD computation preserves specified rank exactly

---

## Performance Optimization Notes

### Hessian Computation
- Use `torch.func.vhp` for Hessian-vector products (faster than full Hessian)
- Lanczos iteration: k=20 eigenvalues sufficient for analysis
- Batch size: 32 optimal for A100 GPU

### Gradient Collection
- Pre-allocate tensors for gradient storage
- Use mixed precision (fp16) for memory efficiency
- Save gradients incrementally to avoid OOM

### Geomstats
- Numpy backend faster for small matrices (n < 100)
- PyTorch backend better for GPU acceleration
- Cache geodesic computations when possible

---

## Testing Checklist

Before submitting experiments:
- [ ] Verify model checkpoint loads correctly
- [ ] Test forward pass on small batch
- [ ] Check gradient computation works
- [ ] Validate output shapes
- [ ] Confirm GPU utilization > 80%
- [ ] Check for memory leaks in loop

---

## Useful Debugging Commands

### Check SLURM job status
```bash
scripts/cluster/status.sh <job_id>
```

### Fetch logs from cluster
```bash
scripts/cluster/remote_fetch.sh ired-interp
```

### Local gradient computation test
```python
# Test on CPU first
model = EBM(inp_dim=400, out_dim=400).cpu()
x = torch.randn(1, 400)
y = torch.randn(1, 400).requires_grad_(True)
energy = model(torch.cat([x, y], dim=-1), t=torch.tensor([5.0]))
grad = torch.autograd.grad(energy.sum(), y)[0]
print(f"Gradient shape: {grad.shape}")  # Should be [1, 400]
```

### Hessian eigenvalue test
```python
from pyhessian import hessian

# Create loss function wrapper
def energy_fn():
    return model(torch.cat([x, y], dim=-1), t).sum()

hessian_comp = hessian(model, energy_fn, cuda=True)
top_eigenvalues = hessian_comp.eigenvalues(top_n=10)
print(f"Top eigenvalues: {top_eigenvalues}")
```

---

## Error Patterns

### Pattern: NaN in energy
**Causes**:
- Exploding gradients
- Numerical instability in energy computation
**Debug**: Add gradient clipping, check input normalization

### Pattern: Slow convergence
**Causes**:
- Learning rate too low
- Poor initialization
**Debug**: Increase LR, use better init (Xavier, He)

### Pattern: Hessian computation hangs
**Causes**:
- Too many parameters
- Full Hessian instead of eigenvalues
**Debug**: Use Lanczos, reduce model size for testing
