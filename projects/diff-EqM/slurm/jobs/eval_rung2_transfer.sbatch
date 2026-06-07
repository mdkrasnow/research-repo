#!/bin/bash
#SBATCH --job-name=eqm-rung2
#SBATCH --output=slurm/logs/eqm-rung2_%A.out
#SBATCH --error=slurm/logs/eqm-rung2_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1

# Rung-2 held-out corruption transfer: zero-shot restore of gray/lowres/blur/crop
# corruptions, vanilla vs v10. v10 mined ONLY on generation hard examples (never
# saw these corruptions) -> any v10>vanilla gap = general repair prior, not trick.
# Identical GD inference path both arms; only checkpoint differs.
#
# Required env: VANILLA_CKPT, V10_CKPT
# Optional: NUM_IMAGES (16), MODEL (EqM-B/2)

set -uo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/eqm-rung2-${SLURM_JOB_ID}"
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

echo "=== Checking dependencies ===" >&2
if ! python -c "import timm, diffusers, scipy, torchvision, lpips" 2>/dev/null; then
    pip install --no-input timm diffusers accelerate scipy torchvision lpips pillow 2>&1 | tail -5
fi

REPO_ROOT="/n/home03/mkrasnow/research-repo"
MODEL="${MODEL:-EqM-B/2}"
NUM_IMAGES="${NUM_IMAGES:-16}"
VAL_PATH="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val"
VANILLA_CKPT="${VANILLA_CKPT:?ERROR: VANILLA_CKPT must be set}"
V10_CKPT="${V10_CKPT:?ERROR: V10_CKPT must be set}"

OUT_DIR="$REPO_ROOT/projects/diff-EqM/results/rung2_transfer_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

run_one () {
    local corr="$1" ckpt="$2" tag="$3"
    echo "=== restore/$corr / $tag ===" >&2
    python -u projects/diff-EqM/experiments/eval_capabilities.py \
        --mode restore --corruption "$corr" --ckpt "$ckpt" --tag "$tag" \
        --out-dir "$OUT_DIR" --model "$MODEL" \
        --num-images "$NUM_IMAGES" --val-path "$VAL_PATH" 2>&1 | tail -20
}

for corr in gray lowres blur crop; do
    run_one "$corr" "$VANILLA_CKPT" "vanilla"
    run_one "$corr" "$V10_CKPT" "v10"
done

# Side-by-side compare per corruption (top=vanilla, bottom=v10).
python - "$OUT_DIR" <<'PY' 2>&1 | tail -10
import sys, glob, os
from PIL import Image
out = sys.argv[1]
for corr in ["gray", "lowres", "blur", "crop"]:
    va = sorted(glob.glob(os.path.join(out, f"restore_{corr}_vanilla.png")))
    v10 = sorted(glob.glob(os.path.join(out, f"restore_{corr}_v10.png")))
    if not va or not v10:
        continue
    a, b = Image.open(va[0]), Image.open(v10[0])
    W = max(a.width, b.width)
    canvas = Image.new("RGB", (W, a.height + b.height + 20), "white")
    canvas.paste(a, (0, 0)); canvas.paste(b, (0, a.height + 20))
    p = os.path.join(out, f"COMPARE_restore_{corr}.png")
    canvas.save(p); print("wrote", p, "(top=vanilla, bottom=v10)")
PY

echo "=== Done. Grids + metrics in $OUT_DIR ===" >&2
ls -la "$OUT_DIR" >&2
rm -rf "$WORK_DIR"
