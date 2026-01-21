# Quick Start Guide - IRED Interpretability

Get started analyzing IRED energy landscapes in 5 minutes.

## Setup

```bash
cd projects/ired-interp

# Install dependencies
./setup_env.sh

# Activate environment
source venv/bin/activate

# Verify installation
python test_modules.py
```

## Run Your First Analysis

### Option 1: Local Test (CPU, small scale)

```bash
# Quick test with dummy data (no checkpoint needed)
python -c "
from analysis import HessianAnalyzer
import torch

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(800, 1)
    def forward(self, x, t):
        return self.fc(x).pow(2).sum(dim=-1, keepdim=True)

model = DummyModel()
analyzer = HessianAnalyzer(model, device='cpu', n_eigenvalues=5, use_lanczos=False)

x = torch.randn(1, 400)
y = torch.randn(1, 400, requires_grad=True)
t = torch.tensor([5.0])

print('Computing Hessian eigenspectrum...')
result = analyzer.compute_hessian_eigenspectrum(x, y, t)
print(f'Top 5 eigenvalues: {result.eigenvalues}')
print(f'Condition number: {result.condition_number:.2f}')
print(f'Effective rank: {result.effective_rank:.2f}')
"
```

### Option 2: Full Experiment (requires checkpoint)

```bash
# Place pretrained IRED checkpoint in checkpoints/
# Then run:
python experiments/exp001_hessian_analysis.py \
    --checkpoint checkpoints/matrix_inverse.pt \
    --task inverse \
    --rank 2 \
    --num_samples 5 \
    --device cpu \
    --output_dir results/exp001_test
```

### Option 3: Cluster (GPU, full scale)

```bash
# From research-repo root
cd /Users/mkrasnow/Desktop/research-repo

# Ensure cluster SSH session active
scripts/cluster/ssh_bootstrap.sh

# Submit experiment
scripts/cluster/remote_submit.sh ired-interp slurm/exp001_hessian.sbatch

# Check status
scripts/cluster/status.sh <job_id>

# Fetch results
scripts/cluster/remote_fetch.sh ired-interp
```

## What Each Module Does

### Hessian Analysis
```python
from analysis import HessianAnalyzer

analyzer = HessianAnalyzer(energy_model, device="cuda")

# Compute eigenspectrum at one point
result = analyzer.compute_hessian_eigenspectrum(x, y, t)

# Analyze across annealing levels
results = analyzer.analyze_eigenspectrum_across_annealing(
    x, y,
    annealing_levels=[1, 3, 5, 7, 10]
)

# Visualize
analyzer.plot_eigenspectrum(results, save_path="hessian.png")
```

**Output**: Eigenvalues, condition number, effective rank

### Grassmannian Geometry
```python
from analysis import GrassmannianAnalyzer

analyzer = GrassmannianAnalyzer(n=20, k=2)  # 20x20 matrices, rank 2

# Riemannian distance
dist = analyzer.compute_grassmann_distance(matrix1, matrix2)

# Principal angles
angles = analyzer.compute_principal_angles(matrix1, matrix2)

# Tangent/normal decomposition
tangent, normal = analyzer.decompose_gradient_tangent_normal(gradient, matrix)
ratio = analyzer.compute_curvature_ratio(gradient, matrix)
```

**Output**: Distances, angles, curvature ratio (tests if energy respects manifold)

### Sparse Autoencoder
```python
from analysis import SAEConfig, SAETrainer, GradientDataset, collect_gradient_samples

# Collect gradients
gradients = collect_gradient_samples(
    model, x_samples, y_samples, t_samples, device="cuda"
)

# Train SAE
config = SAEConfig(input_dim=400, hidden_dim=512, sparsity_coef=0.01)
trainer = SAETrainer(config)
history = trainer.train(GradientDataset(gradients))

# Analyze features
from analysis import FeatureAnalyzer
analyzer = FeatureAnalyzer(trainer.model)
top_idx, top_vals = analyzer.find_top_activating_examples(gradients, feature_idx=0)
```

**Output**: Learned monosemantic features, feature visualizations

## Experiments Overview

| Exp | Name | Runtime | Output |
|-----|------|---------|--------|
| 001 | Hessian Eigenspectrum | ~4h | Eigenvalue plots, condition numbers |
| 002 | Landscape Viz | ~2h | 2D energy contours |
| 003 | SAE Training | ~8h | Feature visualizations |
| 004 | Grassmann Distance | ~1h | Distance vs energy correlation |
| 005 | Tangent/Normal | ~3h | Curvature ratios |

## Troubleshooting

### Import Errors
```bash
# Ensure you're in project directory
cd projects/ired-interp

# Activate environment
source venv/bin/activate

# Verify Python path includes current directory
python -c "import sys; print(sys.path)"
```

### Geomstats Not Found
```bash
pip install geomstats
# or
conda install -c conda-forge geomstats
```

### CUDA Out of Memory
```python
# Use smaller batch size or fewer eigenvalues
analyzer = HessianAnalyzer(model, n_eigenvalues=10)  # Instead of 20
```

## Understanding Results

### Good Eigenspectrum
- **Eigenvalues decay smoothly** (no huge gaps except at end)
- **Condition number increases** as k increases (landscape sharpens)
- **Effective rank is low** (~5-10 for 400-dim space)

### Energy Respects Manifold
- **Curvature ratio >> 1** (e.g., 10-100)
  - Energy steep in normal direction (off-manifold)
  - Energy flat in tangent direction (on-manifold)
- **Grassmannian distance correlates with energy** (r > 0.8)

### SAE Discovers Features
- **Sparsity ~5%** (each feature activates rarely)
- **Low reconstruction loss** (<0.01)
- **Features align with geometry** (singular vectors, rank, etc.)

## Next Steps

1. **Run EXP-001** to validate framework
2. **Examine results** in `results/exp001_*/`
3. **Iterate** on analysis based on findings
4. **Run EXP-002, 003, ...** for complete analysis
5. **Write paper** summarizing interpretability insights

## Documentation

- **Research Plan**: `documentation/research-plan.md`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Full README**: `README_INTERPRETABILITY.md`
- **Debugging**: `documentation/debugging.md`

## Questions?

Check `documentation/debugging.md` or examine module docstrings:
```python
from analysis import HessianAnalyzer
help(HessianAnalyzer)
```

---

**Ready to go!** Start with `python test_modules.py` to verify everything works.
