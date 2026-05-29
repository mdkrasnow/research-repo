#!/bin/bash
#SBATCH --job-name=eqm-capabilities
#SBATCH --output=slurm/logs/eqm-capabilities_%A.out
#SBATCH --error=slurm/logs/eqm-capabilities_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1

# Capability eval: denoise / inpaint / compose on REAL val images.
# Runs ALL modes for BOTH vanilla + v10 checkpoints, single GPU.
# Output: PNG grids (qualitative) + metrics json, side-by-side merged.
#
# Required env: VANILLA_CKPT, V10_CKPT
# Optional: NUM_IMAGES (8), MODEL (EqM-B/2)

set -uo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/eqm-capabilities-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== Cloning repo at SHA ${GIT_SHA} ===" >&2
git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -5 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -3 || {
    git fetch --unshallow origin 2>&1 | tail -3
    git checkout "${GIT_SHA}" 2>&1 | tail -3 || exit 1
}

echo "=== Installing dependencies ===" >&2
pip install timm diffusers accelerate scipy torchvision lpips pillow 2>&1 | tail -5

REPO_ROOT="/n/home03/mkrasnow/research-repo"
MODEL="${MODEL:-EqM-B/2}"
NUM_IMAGES="${NUM_IMAGES:-8}"
VAL_PATH="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val"
VANILLA_CKPT="${VANILLA_CKPT:?ERROR: VANILLA_CKPT must be set}"
V10_CKPT="${V10_CKPT:?ERROR: V10_CKPT must be set}"

OUT_DIR="$REPO_ROOT/projects/diff-EqM/results/capabilities_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

run_one () {
    local mode="$1" ckpt="$2" tag="$3"
    echo "=== $mode / $tag ===" >&2
    python projects/diff-EqM/experiments/eval_capabilities.py \
        --mode "$mode" --ckpt "$ckpt" --tag "$tag" \
        --out-dir "$OUT_DIR" --model "$MODEL" \
        --num-images "$NUM_IMAGES" --val-path "$VAL_PATH" 2>&1 | tail -20
}

for mode in denoise inpaint compose; do
    run_one "$mode" "$VANILLA_CKPT" "vanilla"
    run_one "$mode" "$V10_CKPT" "v10"
done

# Side-by-side merge: stack vanilla grid above v10 grid per mode for easy compare.
echo "=== Merging vanilla|v10 grids ===" >&2
python - "$OUT_DIR" <<'PY' 2>&1 | tail -10
import sys, glob, os
from PIL import Image
out = sys.argv[1]
for mode_glob in ["denoise", "inpaint_box", "inpaint_half", "compose"]:
    va = sorted(glob.glob(os.path.join(out, f"{mode_glob}_vanilla.png")))
    v10 = sorted(glob.glob(os.path.join(out, f"{mode_glob}_v10.png")))
    if not va or not v10:
        continue
    a, b = Image.open(va[0]), Image.open(v10[0])
    W = max(a.width, b.width)
    canvas = Image.new("RGB", (W, a.height + b.height + 20), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (0, a.height + 20))
    p = os.path.join(out, f"COMPARE_{mode_glob}.png")
    canvas.save(p)
    print("wrote", p, "(top=vanilla, bottom=v10)")
PY

echo "=== Done. Grids + metrics in $OUT_DIR ===" >&2
ls -la "$OUT_DIR" >&2
rm -rf "$WORK_DIR"
