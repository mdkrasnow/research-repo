#!/bin/bash
#SBATCH --job-name=eqm-1k-fid
#SBATCH --output=slurm/logs/eqm-1k-fid_%A.out
#SBATCH --error=slurm/logs/eqm-1k-fid_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=6:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# FID evaluation for EqM-B/2 on ImageNet-1K
# Generates 50K samples with DDP, then computes FID against real ImageNet
#
# Required env vars: CHECKPOINT_PATH
# Optional: NUM_FID_SAMPLES, SAMPLE_BATCH_SIZE

set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/eqm-1k-fid-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== Cloning repo at SHA ${GIT_SHA} ===" >&2
git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -10 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || {
    echo "SHA not in last 50 commits, fetching full..." >&2
    git fetch --unshallow origin 2>&1 | tail -5
    git checkout "${GIT_SHA}" 2>&1 | tail -5 || exit 1
}

echo "=== Installing dependencies ===" >&2
pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -10

# Configuration
CHECKPOINT_PATH="${CHECKPOINT_PATH:?ERROR: CHECKPOINT_PATH must be set}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
IMAGENET_REF="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train"
SAMPLES_DIR="$WORK_DIR/samples"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"

mkdir -p "$SAMPLES_DIR"

# === STEP 1: Copy checkpoint locally ===
echo "=== Resolving checkpoint: $CHECKPOINT_PATH ===" >&2
if [ ! -f "$CHECKPOINT_PATH" ]; then
    FULL_CKPT="$REPO_ROOT/$CHECKPOINT_PATH"
    if [ -f "$FULL_CKPT" ]; then
        CHECKPOINT_PATH="$FULL_CKPT"
    else
        echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH or $FULL_CKPT" >&2
        exit 1
    fi
fi

LOCAL_CKPT="$WORK_DIR/checkpoint.pt"
cp "$CHECKPOINT_PATH" "$LOCAL_CKPT"
echo "=== Checkpoint copied ($(du -h "$LOCAL_CKPT" | cut -f1)) ===" >&2

# === STEP 2: Flatten reference images for pytorch-fid ===
echo "=== Preparing flat reference directory ===" >&2
FLAT_REF="$WORK_DIR/ref_flat"
mkdir -p "$FLAT_REF"

# Subsample reference images (50K to match generated count, speeds up FID)
find "$IMAGENET_REF" -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | shuf -n $NUM_FID_SAMPLES | while read f; do
    CLASSDIR=$(basename "$(dirname "$f")")
    FNAME=$(basename "$f")
    ln -s "$f" "$FLAT_REF/${CLASSDIR}_${FNAME}"
done

REF_COUNT=$(ls "$FLAT_REF" | wc -l)
echo "=== Flat reference: $REF_COUNT images ===" >&2

# === STEP 3: Generate samples with DDP ===
echo "=== Generating $NUM_FID_SAMPLES samples ===" >&2
echo "=== Start time: $(date) ===" >&2

MASTER_PORT=$((29500 + RANDOM % 1000))
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "=== Using $NUM_GPUS GPU(s) ===" >&2

python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    projects/diff-EqM/eqm-upstream/sample_gd.py \
    --model "${MODEL:-EqM-B/2}" \
    --ckpt "$LOCAL_CKPT" \
    --num-classes 1000 \
    --sampler gd \
    --stepsize 0.003 \
    --num-sampling-steps 250 \
    --cfg-scale 1.0 \
    --global-batch-size $SAMPLE_BATCH_SIZE \
    --num-fid-samples $NUM_FID_SAMPLES \
    --folder "$SAMPLES_DIR" \
    2>&1 | tee sample.log

SAMPLE_EXIT=${PIPESTATUS[0]}
echo "=== Sampling exit code: $SAMPLE_EXIT, time: $(date) ===" >&2

if [ $SAMPLE_EXIT -ne 0 ]; then
    cp sample.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet1k_fid_${SLURM_JOB_ID}_sample.log" 2>/dev/null || true
    exit $SAMPLE_EXIT
fi

SAMPLE_COUNT=$(find "$SAMPLES_DIR" -name "*.png" -type f | wc -l)
echo "=== Generated $SAMPLE_COUNT samples ===" >&2

# === STEP 4: Compute FID ===
echo "=== Computing FID ===" >&2
python -c "
import torch
import numpy as np
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torchvision import transforms
from PIL import Image
import pathlib

device = torch.device('cuda')
dims = 2048
batch_size = 64

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

def get_activations(image_dir, model, transform, batch_size, device):
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPEG'}
    path = pathlib.Path(image_dir)
    files = sorted([f for f in path.iterdir() if f.suffix in IMAGE_EXTS])
    print(f'  Found {len(files)} images in {image_dir}')
    all_acts = []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        imgs = []
        for f in batch_files:
            try:
                img = Image.open(f).convert('RGB')
                imgs.append(transform(img))
            except Exception as e:
                continue
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
        all_acts.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
        if (i // batch_size) % 100 == 0:
            print(f'  Processed {min(i+batch_size, len(files))}/{len(files)}')
    return np.concatenate(all_acts, axis=0)

print('Computing activations for generated samples...')
act1 = get_activations('$SAMPLES_DIR', model, transform, batch_size, device)
mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)

print('Computing activations for reference images...')
act2 = get_activations('$FLAT_REF', model, transform, batch_size, device)
mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
print(f'FID: {fid:.4f}')
" 2>&1 | tee fid.log

FID_EXIT=${PIPESTATUS[0]}
echo "=== FID RESULT ===" >&2
grep "^FID:" fid.log >&2 || echo "(no FID line)" >&2
echo "=== FID exit: $FID_EXIT, time: $(date) ===" >&2

# === Sync results ===
RESULTS_OUT="$REPO_ROOT/projects/diff-EqM/results/imagenet1k_fid_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_OUT"
cp fid.log sample.log "$RESULTS_OUT/" 2>/dev/null || true
cp fid.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet1k_fid_${SLURM_JOB_ID}.log" 2>/dev/null || true

rm -rf "$WORK_DIR"
echo "=== Done (exit=$FID_EXIT) ===" >&2
exit $FID_EXIT
