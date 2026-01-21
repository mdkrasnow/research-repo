#!/bin/bash
# Setup script for IRED interpretability research environment

set -e

echo "=========================================="
echo "IRED Interpretability Environment Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for local, GPU will be installed on cluster)
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
echo "Installing core dependencies..."
pip install einops numpy scipy matplotlib seaborn plotly tqdm pandas scikit-learn

# Install interpretability libraries
echo "Installing interpretability libraries..."
pip install geomstats  # Riemannian geometry

# PyHessian (may need special handling)
echo "Installing PyHessian..."
pip install pyhessian || echo "Warning: PyHessian installation failed. Will install on cluster."

# Install UMAP for dimensionality reduction
echo "Installing UMAP..."
pip install umap-learn

# Install Jupyter for interactive analysis
echo "Installing Jupyter..."
pip install jupyter ipywidgets

# Install development tools
echo "Installing development tools..."
pip install pytest black flake8

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import einops; print('✓ einops')"
python -c "import numpy; print('✓ numpy')"
python -c "import matplotlib; print('✓ matplotlib')"
python -c "import plotly; print('✓ plotly')"

# Geomstats check
if python -c "import geomstats" 2>/dev/null; then
    echo "✓ geomstats"
else
    echo "✗ geomstats (will install on cluster)"
fi

# PyHessian check
if python -c "import pyhessian" 2>/dev/null; then
    echo "✓ pyhessian"
else
    echo "✗ pyhessian (will install on cluster)"
fi

python -c "import umap; print('✓ umap-learn')"
python -c "import jupyter; print('✓ jupyter')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run local tests:"
echo "  python experiments/exp001_hessian_analysis.py --device cpu --num_samples 2"
echo ""
echo "To submit to cluster:"
echo "  scripts/cluster/remote_submit.sh ired-interp slurm/exp001_hessian.sbatch"
echo ""
