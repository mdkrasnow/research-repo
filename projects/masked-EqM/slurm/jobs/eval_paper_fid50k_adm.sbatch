#!/bin/bash
#SBATCH --job-name=eqm-fid50k-adm
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fid50k-adm-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fid50k-adm-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320GB
#SBATCH --time=1-00:00:00
#SBATCH --partition=seas_gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:4
#SBATCH --exclusive
set -euo pipefail

# Paper-comparable EqM-B/2 FID protocol:
#   * EMA checkpoint, class-conditional ImageNet-256, CFG 1.0
#   * vanilla GD, eta=0.003, 250-step argument (249 update loop iterations)
#   * 50,000 generated samples
#   * OpenAI ADM TensorFlow evaluator and official ImageNet-256 reference batch

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
# The released EqM code enables TF32 for speed but records that it was disabled
# for the paper run. This driver follows the recorded paper setting.
export NVIDIA_TF32_OVERRIDE=0
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT="${CKPT:?set CKPT}"
EBM="${EBM:?set EBM to none or direct}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/paper_fid50k_adm}"
ADM_COMMIT="22e0df8183507e13a7813f8d38d51b072ca1e67c"
ADM_EVALUATOR_SHA256="c8f10ab1406c45d105d7f5b347e5be06c118bd2ddef1c54789c263ebae7f1ca7"
REFERENCE_URL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
REFERENCE_MD5="14c5913e1d24e8827c1ba61d8fc5aeb9"
WORK_DIR="/tmp/masked-eqm-paper-fid-${SLURM_JOB_ID}"
OUTPUT_DIR="$RESULTS_ROOT/$RUN_TAG"
ASSET_DIR="$RESULTS_ROOT/assets"
REFERENCE_BATCH="$ASSET_DIR/VIRTUAL_imagenet256_labeled.npz"
SAMPLE_DIR="$WORK_DIR/samples"
SAMPLE_NPZ="$SAMPLE_DIR.npz"

if [[ "$EBM" != "none" && "$EBM" != "direct" ]]; then
  echo "EBM must be none or direct, got: $EBM" >&2
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 3
fi

mkdir -p "$WORK_DIR" "$OUTPUT_DIR" "$ASSET_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM

python -m pip install --quiet \
  timm diffusers accelerate torchdiffeq torchvision wandb matplotlib scikit-learn \
  tensorflow==2.12.1 scipy==1.11.4 requests tqdm

curl --fail --location --retry 5 \
  "https://raw.githubusercontent.com/openai/guided-diffusion/${ADM_COMMIT}/evaluations/evaluator.py" \
  --output "$WORK_DIR/evaluator.py"
echo "$ADM_EVALUATOR_SHA256  $WORK_DIR/evaluator.py" | sha256sum --check

# The canonical reference batch is about 2 GB. Serialize its one-time download
# across jobs and validate the blob's published Content-MD5 before use.
(
  flock --exclusive 9
  if [[ ! -f "$REFERENCE_BATCH" ]] || ! echo "$REFERENCE_MD5  $REFERENCE_BATCH" | md5sum --check --status; then
    temporary_reference="$REFERENCE_BATCH.partial.${SLURM_JOB_ID}"
    rm -f "$temporary_reference"
    curl --fail --location --retry 5 "$REFERENCE_URL" --output "$temporary_reference"
    echo "$REFERENCE_MD5  $temporary_reference" | md5sum --check
    mv "$temporary_reference" "$REFERENCE_BATCH"
  fi
) 9>"$ASSET_DIR/reference-download.lock"

CHECKPOINT_SHA256="$(sha256sum "$CKPT" | awk '{print $1}')"
STARTED_AT="$(date --iso-8601=seconds)"

python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_port="$((29500 + SLURM_JOB_ID % 1000))" \
  sample_gd.py \
  --model EqM-B/2 \
  --image-size 256 \
  --num-classes 1000 \
  --global-batch-size 256 \
  --global-seed "$GLOBAL_SEED" \
  --vae ema \
  --cfg-scale 1.0 \
  --ckpt "$CKPT" \
  --stepsize 0.003 \
  --num-sampling-steps 250 \
  --folder "$SAMPLE_DIR" \
  --sampler gd \
  --num-fid-samples 50000 \
  --ebm "$EBM" \
  2>&1 | tee "$OUTPUT_DIR/sampling.log"

python - "$SAMPLE_NPZ" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
with np.load(path) as batch:
    samples = batch["arr_0"]
    assert samples.shape == (50_000, 256, 256, 3), samples.shape
    assert samples.dtype == np.uint8, samples.dtype
PY

mkdir -p "$OUTPUT_DIR/sample_probe"
cp "$SAMPLE_DIR"/0000{00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15}.png \
  "$OUTPUT_DIR/sample_probe/"
cp "$SAMPLE_NPZ" "$OUTPUT_DIR/samples_50000.npz"
SAMPLES_SHA256="$(sha256sum "$OUTPUT_DIR/samples_50000.npz" | awk '{print $1}')"

python "$WORK_DIR/evaluator.py" "$REFERENCE_BATCH" "$OUTPUT_DIR/samples_50000.npz" \
  2>&1 | tee "$OUTPUT_DIR/adm_evaluation.txt"

FID_VALUE="$(awk -F': ' '/^FID:/ {print $2}' "$OUTPUT_DIR/adm_evaluation.txt" | tail -1)"
SFID_VALUE="$(awk -F': ' '/^sFID:/ {print $2}' "$OUTPUT_DIR/adm_evaluation.txt" | tail -1)"
IS_VALUE="$(awk -F': ' '/^Inception Score:/ {print $2}' "$OUTPUT_DIR/adm_evaluation.txt" | tail -1)"
if [[ -z "$FID_VALUE" ]]; then
  echo "ADM evaluator completed without a parseable FID" >&2
  exit 4
fi

COMPLETED_AT="$(date --iso-8601=seconds)"
python - "$OUTPUT_DIR/result.json" <<PY
import json
import sys

result = {
    "status": "completed",
    "run_tag": ${RUN_TAG@Q},
    "checkpoint": ${CKPT@Q},
    "checkpoint_sha256": ${CHECKPOINT_SHA256@Q},
    "git_sha": ${GIT_SHA@Q},
    "model": "EqM-B/2",
    "ebm": ${EBM@Q},
    "checkpoint_weights": "ema",
    "image_size": 256,
    "num_classes": 1000,
    "class_conditional": True,
    "cfg_scale": 1.0,
    "sampler": "gd",
    "stepsize": 0.003,
    "num_sampling_steps_argument": 250,
    "function_evaluations_per_sample": 249,
    "num_samples": 50000,
    "global_seed": int(${GLOBAL_SEED@Q}),
    "vae": "stabilityai/sd-vae-ft-ema",
    "latent_scale": 0.18215,
    "nvidia_tf32_override": 0,
    "evaluator": "OpenAI ADM TensorFlow",
    "adm_commit": ${ADM_COMMIT@Q},
    "adm_evaluator_sha256": ${ADM_EVALUATOR_SHA256@Q},
    "reference_batch": ${REFERENCE_BATCH@Q},
    "reference_url": ${REFERENCE_URL@Q},
    "reference_md5": ${REFERENCE_MD5@Q},
    "samples_npz": ${OUTPUT_DIR@Q} + "/samples_50000.npz",
    "samples_sha256": ${SAMPLES_SHA256@Q},
    "fid": float(${FID_VALUE@Q}),
    "sfid": float(${SFID_VALUE@Q}) if ${SFID_VALUE@Q} else None,
    "inception_score": float(${IS_VALUE@Q}) if ${IS_VALUE@Q} else None,
    "started_at": ${STARTED_AT@Q},
    "completed_at": ${COMPLETED_AT@Q},
    "paper_reference_fid_b2_epoch80": 32.85,
    "claim_boundary": "Paper evaluation protocol; training recipe differs from the paper when ebm=direct or the checkpoint was trained with global batch 32.",
}
with open(sys.argv[1], "w") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "Paper-protocol FID complete: $OUTPUT_DIR/result.json"
