#!/bin/bash
#SBATCH --job-name=ired_q002
#SBATCH --partition=gpu_test
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/q002_%j.out
#SBATCH --error=slurm/logs/q002_%j.err

# IRED Baseline: Q-002 Standard Configuration
# Full baseline experiment on matrix inversion (100k steps, rank=20)

set -e  # Exit on error

echo "=========================================="
echo "Q-002: IRED Baseline Standard Configuration"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Module loads
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPU
echo "Checking GPU availability..."
nvidia-smi

# Clone repository to /tmp for this job
WORK_DIR="/tmp/ired-baseline-job-$SLURM_JOB_ID"
echo "Setting up workspace: $WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repo and checkout specific commit
echo "Cloning repository..."
git clone https://github.com/mdkrasnow/research-repo.git .

# Checkout the commit passed via environment variable
GIT_SHA="${GIT_SHA:-7770702133ed39020ae4a0424e6b600ec7a10c4b}"
echo "Checking out commit: $GIT_SHA"
git checkout "$GIT_SHA"

# Verify commit
ACTUAL_SHA=$(git rev-parse HEAD)
echo "Current commit: $ACTUAL_SHA"

if [ "$ACTUAL_SHA" != "$GIT_SHA" ]; then
    echo "ERROR: Git checkout mismatch!"
    echo "  Expected: $GIT_SHA"
    echo "  Got: $ACTUAL_SHA"
    exit 1
fi

# Navigate to project
cd projects/ired-baseline

# Create virtual environment
echo "Setting up Python environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pandas einops tqdm tabulate ema-pytorch accelerate Pillow ipdb ipython matplotlib

# Verify installations
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import einops; print('einops installed')"

# Create run directory
RUN_ID="q002_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "Run ID: $RUN_ID"
echo "Run directory: $RUN_DIR"

# Save submission metadata
cat > "$RUN_DIR/submit.json" <<EOF
{
  "job_id": "$SLURM_JOB_ID",
  "run_id": "$RUN_ID",
  "experiment": "q002_baseline",
  "git_sha": "$GIT_SHA",
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "node": "$SLURM_NODELIST",
  "partition": "gpu_test",
  "config": "configs/q002_baseline.json"
}
EOF

# Run experiment
echo "Running Q-002 baseline experiment..."
echo "Configuration: 100k steps, rank=20, batch_size=2048"
python experiments/run_baseline.py \
    --config configs/q002_baseline.json \
    --output-dir "$RUN_DIR"

# Check if results were saved
if [ -f "$RUN_DIR/results.json" ]; then
    echo "Results saved successfully!"
    cat "$RUN_DIR/results.json"
else
    echo "WARNING: results.json not found"
fi

# Persist results back to shared storage BEFORE cleanup
echo "Persisting results to shared storage..."
REMOTE_RUN_DIR="/n/home03/mkrasnow/research-repo/projects/ired-baseline/runs/$RUN_ID"
mkdir -p "$REMOTE_RUN_DIR"
rsync -av "$RUN_DIR/" "$REMOTE_RUN_DIR/"
echo "Results persisted to: $REMOTE_RUN_DIR"

echo "Experiment complete!"

# Cleanup
echo "Cleaning up workspace..."
cd /tmp
rm -rf "$WORK_DIR"

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
