#!/bin/bash
#SBATCH --job-name=eqm-in100-pilot
#SBATCH --output=slurm/logs/eqm-in100-pilot_%A.out
#SBATCH --error=slurm/logs/eqm-in100-pilot_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=1:30:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Autoresearch pilot for DG-ANM on ImageNet-100 (4x GPU DDP)
# Trains for 2 epochs, samples 5K images, computes FID against precomputed ref stats.
# Outputs "imagenet100_fid: <value>" for autoresearch loop to parse.
#
# Required env vars:
#   CONFIG_PATH — training config JSON (in configs/)
# Optional:
#   SEED — random seed (default 0)
#   CKPT_EVERY — checkpoint frequency in steps (default 10000, usually not needed for 2ep pilots)

set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/eqm-in100-pilot-${SLURM_JOB_ID}"
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
pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -5

# --- Paths ---
REPO_ROOT="/n/home03/mkrasnow/research-repo"
IN100_DATA="/n/home03/mkrasnow/in100_subset"
REF_STATS="$REPO_ROOT/projects/diff-EqM/results/in100_reference_stats.npz"
CONFIG_PATH="${CONFIG_PATH:?ERROR: CONFIG_PATH must be set}"
SEED="${SEED:-0}"
RESULTS_DIR="projects/diff-EqM/results/pilot_${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR/$RESULTS_DIR"

if [ ! -d "$IN100_DATA" ]; then
    echo "ERROR: IN-100 subset not found at $IN100_DATA" >&2
    exit 1
fi
if [ ! -f "$REF_STATS" ]; then
    echo "ERROR: Reference FID stats not found at $REF_STATS" >&2
    echo "Run compute_in100_reference_stats.sbatch first." >&2
    exit 1
fi

# --- Parse candidate config JSON for training args ---
CONFIG_FULL="$WORK_DIR/$CONFIG_PATH"
python3 - "$CONFIG_FULL" > config_args.sh <<'PY'
import json, sys, shlex
cfg = json.load(open(sys.argv[1]))
def arg(k, v):
    if isinstance(v, bool):
        return f"--{k}" if v else ""
    return f"--{k} {shlex.quote(str(v))}"
parts = []
# Mining flags
if cfg.get("use_mining", False): parts.append("--use-mining")
for k in ["gamma","mining_epsilon","mining_steps","neg_margin","mine_every"]:
    if k in cfg: parts.append(arg(k.replace("_","-"), cfg[k]))
# Training flags
for k in ["lr","global_batch_size","epochs","image_size","global_seed"]:
    if k in cfg: parts.append(arg(k.replace("_","-"), cfg[k]))
print(" ".join(p for p in parts if p))
PY
EXTRA_ARGS=$(cat config_args.sh)

MASTER_PORT=$((29500 + RANDOM % 1000))

# --- STEP 1: Train ---
echo "=== STEP 1: Training 2 epochs on IN-100 ===" >&2
echo "=== Config: $CONFIG_PATH, extra args: $EXTRA_ARGS ===" >&2

TRAIN_START=$(date +%s)
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model EqM-B/2 \
    --data-path "$IN100_DATA" \
    --num-classes 100 \
    --image-size 256 \
    --results-dir "$RESULTS_DIR" \
    --ckpt-every 100000 \
    --global-seed "$SEED" \
    $EXTRA_ARGS \
    2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
TRAIN_END=$(date +%s)
echo "=== Training exit: $TRAIN_EXIT, duration: $((TRAIN_END-TRAIN_START))s ===" >&2

if [ $TRAIN_EXIT -ne 0 ]; then
    cp train.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/eqm-in100-pilot_${SLURM_JOB_ID}_train.log" 2>/dev/null || true
    exit $TRAIN_EXIT
fi

# --- STEP 2: Locate final checkpoint ---
FINAL_CKPT=$(find "$WORK_DIR/$RESULTS_DIR" -name 'final.pt' | head -1)
if [ -z "$FINAL_CKPT" ]; then
    echo "ERROR: No final.pt found" >&2
    exit 2
fi
echo "=== Final checkpoint: $FINAL_CKPT ===" >&2

# --- STEP 3: Sample 5K images ---
SAMPLES_DIR="$WORK_DIR/samples"
mkdir -p "$SAMPLES_DIR"
echo "=== STEP 2: Sampling 5K images ===" >&2
SAMPLE_START=$(date +%s)
MASTER_PORT=$((29500 + RANDOM % 1000))
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/eqm-upstream/sample_gd.py \
    --model EqM-B/2 \
    --ckpt "$FINAL_CKPT" \
    --num-classes 100 \
    --sampler gd \
    --stepsize 0.003 \
    --num-sampling-steps 250 \
    --cfg-scale 1.0 \
    --global-batch-size 64 \
    --num-fid-samples 5000 \
    --folder "$SAMPLES_DIR" \
    2>&1 | tee sample.log
SAMPLE_EXIT=${PIPESTATUS[0]}
SAMPLE_END=$(date +%s)
echo "=== Sample exit: $SAMPLE_EXIT, duration: $((SAMPLE_END-SAMPLE_START))s ===" >&2

if [ $SAMPLE_EXIT -ne 0 ]; then
    cp train.log sample.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/eqm-in100-pilot_${SLURM_JOB_ID}_fail.log" 2>/dev/null || true
    exit $SAMPLE_EXIT
fi

# --- STEP 4: Compute FID against precomputed reference stats ---
echo "=== STEP 3: Computing FID ===" >&2
python - "$SAMPLES_DIR" "$REF_STATS" <<'PY' 2>&1 | tee fid.log
import sys, numpy as np, torch, pathlib
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torchvision import transforms
from PIL import Image

samples_dir = sys.argv[1]
ref_stats = sys.argv[2]

device = torch.device('cuda')
block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block]).to(device).eval()
transform = transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor()])

def activations(d, bs=64):
    exts = {'.jpg','.jpeg','.png','.JPEG'}
    files = sorted(f for f in pathlib.Path(d).iterdir() if f.suffix in exts)
    print(f"Loading {len(files)} images from {d}", flush=True)
    acts = []
    for i in range(0, len(files), bs):
        imgs = []
        for f in files[i:i+bs]:
            try: imgs.append(transform(Image.open(f).convert('RGB')))
            except Exception: pass
        if not imgs: continue
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            p = model(batch)[0]
        if p.size(2) != 1: p = torch.nn.functional.adaptive_avg_pool2d(p, (1,1))
        acts.append(p.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(acts, 0)

act1 = activations(samples_dir)
mu1, sigma1 = np.mean(act1, 0), np.cov(act1, rowvar=False)
ref = np.load(ref_stats)
mu2, sigma2 = ref['mu'], ref['sigma']
fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
print(f"imagenet100_fid: {fid:.4f}")
PY

FID_EXIT=${PIPESTATUS[0]}
echo "=== FID exit: $FID_EXIT ===" >&2

# --- Sync logs back to persistent storage ---
cp train.log sample.log fid.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/eqm-in100-pilot_${SLURM_JOB_ID}.log" 2>/dev/null || \
    cat train.log sample.log fid.log > "$REPO_ROOT/projects/diff-EqM/slurm/logs/eqm-in100-pilot_${SLURM_JOB_ID}.log"

# --- Cleanup: only keep the FID log (checkpoints are huge and not needed) ---
rm -rf "$WORK_DIR"

echo "=== Done ===" >&2
exit $FID_EXIT
