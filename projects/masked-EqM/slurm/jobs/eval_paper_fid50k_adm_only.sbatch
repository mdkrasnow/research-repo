#!/bin/bash
#SBATCH --job-name=eqm-fid50k-adm-only
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fid50k-adm-only-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fid50k-adm-only-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
GIT_SHA="${GIT_SHA:?set GIT_SHA}"
SAMPLES_NPZ="${SAMPLES_NPZ:?set SAMPLES_NPZ}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
EBM="${EBM:?set EBM}"
CKPT="${CKPT:?set CKPT}"
REFERENCE_BATCH="${REFERENCE_BATCH:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/paper_fid50k_adm/assets/VIRTUAL_imagenet256_labeled.npz}"
ADM_COMMIT="22e0df8183507e13a7813f8d38d51b072ca1e67c"
ADM_EVALUATOR_SHA256="c8f10ab1406c45d105d7f5b347e5be06c118bd2ddef1c54789c263ebae7f1ca7"
WORK_DIR="/tmp/masked-eqm-paper-fid-only-${SLURM_JOB_ID}"

[[ -f "$SAMPLES_NPZ" && -f "$REFERENCE_BATCH" ]] || { echo "missing samples or reference" >&2; exit 3; }
mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
python -m pip install --quiet tensorflow==2.12.1 scipy==1.11.4 requests tqdm
curl --fail --location --retry 5 "https://raw.githubusercontent.com/openai/guided-diffusion/${ADM_COMMIT}/evaluations/evaluator.py" --output "$WORK_DIR/evaluator.py"
echo "$ADM_EVALUATOR_SHA256  $WORK_DIR/evaluator.py" | sha256sum --check
python evaluate_adm_compat.py "$WORK_DIR/evaluator.py" "$REFERENCE_BATCH" "$SAMPLES_NPZ" 2>&1 | tee "$OUTPUT_DIR/adm_evaluation.txt"
FID_VALUE="$(awk -F': ' '/^FID:/ {print $2}' "$OUTPUT_DIR/adm_evaluation.txt" | tail -1)"
[[ -n "$FID_VALUE" ]] || { echo "no parseable FID" >&2; exit 4; }
python - "$OUTPUT_DIR/result.json" "$RUN_TAG" "$EBM" "$CKPT" "$SAMPLES_NPZ" "$FID_VALUE" <<'PY'
import json
import sys

path, run_tag, ebm, checkpoint, samples, fid = sys.argv[1:]
with open(path, "w") as handle:
    json.dump(
        {
            "status": "completed",
            "run_tag": run_tag,
            "ebm": ebm,
            "checkpoint": checkpoint,
            "checkpoint_weights": "ema",
            "samples_npz": samples,
            "num_samples": 50000,
            "sampler": "gd",
            "stepsize": 0.003,
            "num_sampling_steps_argument": 250,
            "function_evaluations_per_sample": 249,
            "cfg_scale": 1.0,
            "evaluator": "OpenAI ADM TensorFlow",
            "fid": float(fid),
            "compatibility_note": "The historical ADM evaluator's removed np.bool alias was restored only for optional precision/recall after its unchanged FID computation.",
        },
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
echo "ADM FID evaluation-only complete: FID=$FID_VALUE"
