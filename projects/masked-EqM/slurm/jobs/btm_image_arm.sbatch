#!/bin/bash
#SBATCH --job-name=btm-image
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-image-%x-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-image-%x-%A.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4
set -euo pipefail

# Experiment II: corrected-BTM arms on the real ImageNet B/2 EqM recipe.
#
# One arm per job.  Required env:
#   GIT_SHA     commit to run
#   BTM_MODE    btm_vector | btm_scalar_exact | btm_scalar_action_exact
#               | btm_scalar_fd_directional | btm_scalar_fd_action
#   EBM         none (vector arm) | direct (every scalar arm)
#   GLOBAL_SEED seed
#   RUN_TAG     human tag for the results dir
# Optional: EPOCHS SAVE_EPOCHS BTM_TC FD_EPS FD_K MAX_STEPS CKPT NPROC
#
# multi-GPU DDP => seas_gpu (full A100/H200).  gpu_test MIG slices cannot do
# multi-rank NCCL ("Duplicate GPU detected"); use NPROC=1 there.
#
# TELEMETRY (this file is the migration exemplar -- see slurm/lib/README.md):
#   --open-mode=append   a requeue must not truncate the previous attempt's log.
#   --signal=B:USR1@120  120s of warning before the wall clock kills us.  The
#       `B:` prefix is load-bearing: this script runs python directly in the
#       batch shell rather than under `srun`, and without `B:` SLURM signals the
#       job STEPS and explicitly not the batch shell -- the directive would be
#       present and inert.  The prelude's trap forwards it to the process group,
#       so telemetry.lifecycle's in-process handler still gets its chance first
#       and the shell-level sealer is only the fallback.
#   The log names now carry %x (the job name).  SLURM expands #SBATCH
#   directives before any shell variable exists, so RUN_TAG cannot appear
#   literally; the launcher must therefore submit with
#       sbatch --job-name="$RUN_TAG" ... btm_image_arm.sbatch
#   which is what makes %x equal RUN_TAG.  Without it every arm wrote to
#   btm-image-<jobid>.out and the arm was recoverable only by reading the file.
#
# The scientific command below is UNCHANGED from the pre-telemetry version:
# same model, same flags, same pytest gate, same rsync/cleanup, same exit code.

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
BTM_MODE="${BTM_MODE:?set BTM_MODE (use 'none' for the vanilla-EqM control path)}"
EBM="${EBM:?set EBM}"
GLOBAL_SEED="${GLOBAL_SEED:?set GLOBAL_SEED}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
EPOCHS="${EPOCHS:-80}"
SAVE_EPOCHS="${SAVE_EPOCHS:-1,2,5,10,20,40,80}"
BTM_TC="${BTM_TC:-0.8}"
BTM_INTERPOLANT="${BTM_INTERPOLANT:-self_stopping}"
FD_EPS="${FD_EPS:-1e-3}"
FD_K="${FD_K:-1}"
NPROC="${NPROC:-4}"
LOG_EVERY="${LOG_EVERY:-50}"
# ImageNet on holylfs06 measures ~27 cold files/s, ~3 MB/s per stream, so the
# dataloader -- not the GPU -- sets the pace. More workers, and FEW concurrent
# jobs: 5 simultaneous 4-GPU jobs (80 workers) starved each other badly enough
# that none reached step 50 in 20+ minutes.
NUM_WORKERS="${NUM_WORKERS:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
MAX_STEPS_ARG=""
if [ -n "${MAX_STEPS:-}" ]; then MAX_STEPS_ARG="--max-steps ${MAX_STEPS}"; fi
CKPT_ARG=""
if [ -n "${CKPT:-}" ]; then CKPT_ARG="--ckpt ${CKPT}"; fi

WORK_DIR="/tmp/btm-image-${SLURM_JOB_ID}"
export ENTITY="${ENTITY:-none}" PROJECT="${PROJECT:-none}"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
git fetch --depth=50 origin '+refs/heads/*:refs/remotes/origin/*'
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchdiffeq wandb torchvision pytest

# Telemetry prelude.  Sourced only AFTER the checkout, because it imports the
# telemetry package from the commit under test -- the launcher and the trainer
# must agree on the spec serializer, and the only way to guarantee that is for
# both to come from the same tree.
source "$PWD/slurm/lib/telemetry_env.sh"
eqm_forbid_positional "$@"
eqm_prelude "/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/btm"   # sets RESULTS_ROOT,
                                                           # IMAGENET_PATH, MASTER_PORT
RESULTS_DIR="$RESULTS_ROOT/${RUN_TAG}_job${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# Hard gate: the BTM mathematics must be validated on THIS commit before any
# GPU-hours are spent on it.
python -m pytest tests/test_btm_math.py tests/test_btm_image_conventions.py -q

# BTM_MODE=none runs the repository's ORIGINAL training path untouched -- the
# control that separates "our BTM branch hangs" from "4-GPU DDP hangs here".
if [ "$BTM_MODE" = "none" ]; then
  BTM_ARGS=""
else
  BTM_ARGS="--btm-mode $BTM_MODE --btm-interpolant $BTM_INTERPOLANT --btm-tc $BTM_TC --fd-eps $FD_EPS --fd-k $FD_K --btm-eval-every ${BTM_EVAL_EVERY:-500}"
fi

# Mint the run identity and arm the trap ladder.  Every knob that changes what is
# computed goes in as a --param so it is hashed into the run_uid; partition,
# worker count and paths deliberately do not (telemetry.ids.NON_IDENTIFYING_KEYS).
eqm_telemetry_begin \
  --campaign btm --phase II --arm "$BTM_MODE" \
  --seed "$GLOBAL_SEED" --git-sha "$GIT_SHA" \
  --param ebm="$EBM" \
  --param model=EqM-B/2 \
  --param image_size=256 \
  --param epochs="$EPOCHS" \
  --param global_batch="$GLOBAL_BATCH" \
  --param btm_interpolant="$BTM_INTERPOLANT" \
  --param btm_tc="$BTM_TC" \
  --param fd_eps="$FD_EPS" \
  --param fd_k="$FD_K" \
  --param max_steps="${MAX_STEPS:-}" \
  --param resume_ckpt="${CKPT:-}"

CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NPROC --master_port=$MASTER_PORT train.py \
  --model EqM-B/2 --data-path $IMAGENET_PATH --num-classes 1000 \
  --global-batch-size $GLOBAL_BATCH --epochs $EPOCHS --save-epochs $SAVE_EPOCHS \
  --image-size 256 --log-every $LOG_EVERY --ckpt-every 999999999 \
  --num-workers $NUM_WORKERS \
  --results-dir $RESULTS_DIR --ebm $EBM --global-seed $GLOBAL_SEED \
  --grad-log-every ${GRAD_LOG_EVERY:-50} $BTM_ARGS $MAX_STEPS_ARG $CKPT_ARG"
echo "$CMD"

# Backgrounded via eqm_run, not foreground: bash defers trap handlers for a
# foreground command until it returns, so a 24h job would process SLURM's
# 120-second warning 24 hours late.  The inner `exit ${PIPESTATUS[0]}` preserves
# the pre-existing guarantee that the trainer's status -- not tee's -- is what
# this script exits with.
EXIT_CODE=0
eqm_run bash -c "$CMD 2>&1 | tee train.log; exit \${PIPESTATUS[0]}" || EXIT_CODE=$?

# Seal BEFORE `rm -rf "$WORK_DIR"`: the EXIT trap runs the sealer out of the
# checked-out tree, which lives inside WORK_DIR.  Deleting it first would leave
# the trap with no interpreter for its own fallback.  Sealing twice is safe.
eqm_seal_now "$EXIT_CODE"

cp train.log "$REPO_ROOT/projects/masked-EqM/slurm/logs/${RUN_TAG}_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
