#!/bin/bash
#SBATCH --job-name=ired_hessian
#SBATCH --partition=gpu_test
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=projects/ired-interp/slurm/logs/exp001_%j.out
#SBATCH --error=projects/ired-interp/slurm/logs/exp001_%j.err

# IRED Interpretability: Experiment 001 - Hessian Eigenspectrum Analysis
# Computes Hessian eigenvalues across annealing levels for matrix completion

set -e  # Exit on error

echo "=========================================="
echo "EXP-001: Hessian Eigenspectrum Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Git SHA: ${GIT_SHA:-unknown}"
echo "Start time: $(date)"
echo "=========================================="

# Module loads
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPU
echo "Checking GPU availability..."
nvidia-smi

# Clone research repository to /tmp for this job
WORK_DIR="/tmp/ired-interp-job-$SLURM_JOB_ID"
RESEARCH_REPO="https://github.com/mdkrasnow/research-repo.git"
PROJECT_DIR="projects/ired-interp"

echo "Setting up workspace: $WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone research repo
echo "Cloning research repository..."
git clone "$RESEARCH_REPO" repo

cd repo

# Checkout specific commit if provided
if [ -n "$GIT_SHA" ]; then
    echo "Checking out commit: $GIT_SHA"
    git checkout "$GIT_SHA"
fi

# Navigate to project directory
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# Create virtual environment
echo "Setting up Python environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install einops numpy scipy matplotlib seaborn plotly tqdm
pip install geomstats pyhessian

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import geomstats; print('Geomstats installed')"
python -c "import pyhessian; print('PyHessian installed')"

# Create results directory
RESULTS_DIR="results/exp001_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Run experiment (note: checkpoint may not exist yet, script will warn and use random init)
echo "Running Hessian analysis..."
python experiments/exp001_hessian_analysis.py \
    --checkpoint checkpoints/matrix_inverse.pt \
    --task inverse \
    --rank 2 \
    --num_samples 20 \
    --device cuda \
    --output_dir "$RESULTS_DIR"

# Get run_id from environment (set by submit.sh)
RUN_ID="${RUN_ID:-unknown}"
echo "Run ID: $RUN_ID"

# Persist results back to cluster home
REMOTE_RESULTS_DIR="$HOME/research-repo/$PROJECT_DIR/runs/$RUN_ID"
echo "Persisting results to: $REMOTE_RESULTS_DIR"
mkdir -p "$REMOTE_RESULTS_DIR"
rsync -av "$RESULTS_DIR/" "$REMOTE_RESULTS_DIR/"

echo "Results persisted successfully!"

# Cleanup
echo "Cleaning up workspace..."
cd /tmp
rm -rf "$WORK_DIR"

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
