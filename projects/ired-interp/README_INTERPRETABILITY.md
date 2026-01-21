# IRED Energy Field Interpretability Research

This project investigates the interpretability of energy landscapes learned by IRED (Iterative Reasoning through Energy Diffusion) for matrix reasoning tasks.

## Research Overview

IRED learns K=10 annealed energy functions E_θ^k(x,y) that guide gradient-based optimization during inference. This research aims to understand:

1. **Energy Landscape Geometry**: What does the Hessian eigenspectrum reveal about energy curvature?
2. **Gradient Features**: Can we discover interpretable, monosemantic features in the gradient field using sparse autoencoders?
3. **Manifold Structure**: Does the energy function respect the geometric structure of matrix manifolds (e.g., Grassmannian)?
4. **Annealing Dynamics**: How do energy landscapes evolve from coarse (high σ) to fine (low σ)?

## Project Structure

```
ired-interp/
├── analysis/                      # Core analysis modules
│   ├── hessian_analysis.py       # Hessian eigenspectrum computation
│   ├── grassmann_geometry.py     # Riemannian geometry on Grassmannian
│   ├── sparse_autoencoder.py     # SAE for gradient features
│   └── landscape_viz.py          # Energy landscape visualization (TODO)
├── experiments/                   # Experiment scripts
│   ├── exp001_hessian_analysis.py
│   ├── exp002_landscape_viz.py (TODO)
│   └── exp003_sae_training.py (TODO)
├── slurm/                        # SLURM batch scripts
│   └── exp001_hessian.sbatch
├── documentation/
│   ├── research-plan.md          # Detailed research plan
│   ├── implementation-todo.md    # Implementation roadmap
│   ├── queue.md                  # Experiment queue
│   └── debugging.md              # Debugging log
├── results/                      # Experiment results (generated)
├── checkpoints/                  # Model checkpoints (to be added)
└── requirements.txt              # Python dependencies
```

## Installation

### Local Setup (CPU)

```bash
cd projects/ired-interp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Cluster Setup (GPU)

Dependencies are installed automatically in SLURM scripts. See `slurm/exp001_hessian.sbatch` for example.

Key modules:
- `torch` (with CUDA 11.8)
- `geomstats` (Riemannian geometry)
- `pyhessian` (Hessian eigenvalue computation)
- `umap-learn` (dimensionality reduction)
- `plotly` (interactive visualization)

## Running Experiments

### Experiment 001: Hessian Eigenspectrum Analysis

**Objective**: Compute Hessian eigenspectrum for matrix completion across different ranks and annealing levels.

**Local (CPU, small scale)**:
```bash
python experiments/exp001_hessian_analysis.py \
    --checkpoint checkpoints/matrix_inverse.pt \
    --task inverse \
    --rank 2 \
    --num_samples 5 \
    --device cpu \
    --output_dir results/exp001_local
```

**Cluster (GPU, full scale)**:
```bash
# From research-repo root
scripts/cluster/remote_submit.sh ired-interp slurm/exp001_hessian.sbatch
```

**Outputs**:
- `results/exp001_*/hessian_results.json`: Eigenvalues and metrics
- `results/exp001_*/hessian_sample_*.png`: Per-sample plots
- `results/exp001_*/aggregate_stats.png`: Aggregate statistics

### Experiment 002: Energy Landscape Visualization (TODO)

2D slices of energy landscape along geometrically meaningful directions.

### Experiment 003: Sparse Autoencoder Training (TODO)

Train SAE on gradient field to discover interpretable features.

## Analysis Modules

### Hessian Analysis

```python
from analysis.hessian_analysis import HessianAnalyzer

analyzer = HessianAnalyzer(model, device="cuda", n_eigenvalues=20)

# Analyze single point
result = analyzer.compute_hessian_eigenspectrum(x, y, t)
print(f"Eigenvalues: {result.eigenvalues}")
print(f"Condition number: {result.condition_number}")
print(f"Effective rank: {result.effective_rank}")

# Analyze across annealing levels
results = analyzer.analyze_eigenspectrum_across_annealing(x, y, [1,3,5,7,10])
analyzer.plot_eigenspectrum(results)
```

### Grassmannian Geometry

```python
from analysis.grassmann_geometry import GrassmannianAnalyzer

# For rank-2 matrices in R^{20x20}
analyzer = GrassmannianAnalyzer(n=20, k=2)

# Compute Riemannian distance
dist = analyzer.compute_grassmann_distance(matrix1, matrix2)

# Principal angles
angles = analyzer.compute_principal_angles(matrix1, matrix2)

# Tangent/normal decomposition
tangent, normal = analyzer.decompose_gradient_tangent_normal(grad, matrix)
curvature_ratio = analyzer.compute_curvature_ratio(grad, matrix)
```

### Sparse Autoencoder

```python
from analysis.sparse_autoencoder import SAEConfig, SAETrainer, GradientDataset

# Collect gradient samples
gradients = collect_gradient_samples(model, x_samples, y_samples, t_samples)

# Train SAE
config = SAEConfig(input_dim=400, hidden_dim=512, sparsity_coef=0.01)
trainer = SAETrainer(config)
history = trainer.train(GradientDataset(gradients))

# Analyze learned features
from analysis.sparse_autoencoder import FeatureAnalyzer
analyzer = FeatureAnalyzer(trainer.model)
activations = analyzer.get_feature_activations(gradients)
top_indices, top_values = analyzer.find_top_activating_examples(gradients, feature_idx=0)
```

## Expected Results

### Hessian Eigenspectrum
- **Hypothesis**: Eigenspectrum sharpens as k increases (σ decreases)
- **Metrics**: Condition number, effective rank, spectral gap
- **Interpretation**: Small eigenvalues → flat directions (invariant subspaces)

### Grassmannian Geometry
- **Hypothesis**: Energy correlates strongly with Grassmannian distance
- **Metrics**: Correlation coefficient, curvature ratio (normal/tangent)
- **Interpretation**: Energy respects manifold structure if ratio >> 1

### Sparse Autoencoder Features
- **Hypothesis**: SAE discovers monosemantic features aligned with geometry
- **Metrics**: Feature sparsity, reconstruction loss, alignment with singular vectors
- **Interpretation**: Features encode rank corrections, singular value adjustments

## Troubleshooting

### PyHessian CUDA Issues
If Hessian computation fails with CUDA errors:
```python
analyzer = HessianAnalyzer(model, use_lanczos=True, n_eigenvalues=10)
```
Reduce `n_eigenvalues` or use CPU.

### Geomstats Backend
To switch backends:
```python
import geomstats.backend as gs
gs.set_default_backend("pytorch")  # or "numpy", "tensorflow"
```

### Memory Issues
For large Hessian matrices:
- Use Lanczos iteration (default)
- Reduce batch size
- Compute eigenvalues sequentially

## Citation

If you use this interpretability framework, please cite:

```bibtex
@inproceedings{du2024ired,
  title={Learning Iterative Reasoning through Energy Diffusion},
  author={Du, Yilun and Mao, Jiayuan and Tenenbaum, Joshua B.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

## References

- [IRED Paper (arXiv)](https://arxiv.org/abs/2406.11179)
- [IRED Project Page](https://energy-based-model.github.io/ired/)
- [Geomstats Documentation](https://geomstats.github.io/)
- [PyHessian GitHub](https://github.com/amirgholami/PyHessian)
- [Anthropic - Scaling Monosemanticity (SAE)](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

## Contact

For questions about this interpretability research, please open an issue or contact the research team.
