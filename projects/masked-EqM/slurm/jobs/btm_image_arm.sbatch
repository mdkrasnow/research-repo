#!/bin/bash
#SBATCH --job-name=btm-image
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-image-%x-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-image-%x-%A.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
# Required for the preflight's `scontrol requeue` to be permitted: without it
# SLURM refuses the requeue and the transient-fault path silently degrades into
# a plain exit, which is the behaviour it exists to replace. Pairs with
# --open-mode=append so a requeued attempt does not truncate the prior log.
#SBATCH --requeue
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

# ---------------------------------------------------------------------------
# AUTO-RESUME FROM THIS RUN'S OWN NEWEST CHECKPOINT.
#
# Mandatory on a preemptible partition, which gpu_requeue is. Without it a
# requeued job re-reads the seed `CKPT` env var and restarts from the phase's
# STARTING weights (II-B epoch 15) on every single preemption -- so under
# frequent preemption it makes no net progress and may never finish, while
# looking perfectly healthy in the log each time.
#
# SLURM reuses the job id across a requeue, so RESULTS_DIR is stable across
# attempts and the previous attempt's checkpoints are exactly where this looks.
# A fresh SUBMISSION gets a new id and therefore an empty dir, correctly falling
# back to the seed checkpoint.
#
# Newest-by-mtime, not by name: step checkpoints (0100000.pt) and epoch
# checkpoints (epoch20.pt) interleave in one directory and do not sort into a
# single consistent order, so a name sort can silently pick an OLDER state.
LATEST_OWN=$(find "$RESULTS_DIR" -name '*.pt' -type f -printf '%T@ %p\n' 2>/dev/null \
             | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_OWN" ]; then
  eqm_log "auto-resume: attempt ${SLURM_RESTART_COUNT:-0} resuming from this run's own
       newest checkpoint $LATEST_OWN (seed CKPT=${CKPT:-none} ignored -- it is only for
       the first attempt)"
  CKPT="$LATEST_OWN"
  CKPT_ARG="--ckpt $CKPT"
fi

# ---------------------------------------------------------------------------
# DATA-READINESS PREFLIGHT.
#
# Phase II-C burned ~38 A100-hours across four attempts at ZERO training steps
# because /n/holylfs06 developed an intermittent Lustre fault: metadata ops stay
# instant and `ls` of the dataset root returns immediately, but a subset of
# JPEG reads block forever in fstat. Every existing check passed -- the path
# exists, the class count is right, the dataset "contains 1,281,167 images" --
# and then the loader wedged.
#
# The invariant a path check cannot express is DATA IS ACTUALLY READABLE, so
# read some. Random fresh paths, never a fixed set: re-reading the same files
# hits cache and makes a sick filesystem look well (exactly what fooled the
# 2026-08-17 recovery probe). A hard per-file timeout converts an unbounded
# hang into a fast, attributable failure.
eqm_log "preflight: sampling ImageNet reads under a hard timeout"
timeout 180 python - "$IMAGENET_PATH" <<'PREFLIGHT' || {
import os, random, sys, time
root = sys.argv[1]
classes = os.listdir(root)
random.shuffle(classes)
n = 0
t0 = time.time()
for c in classes[:24]:
    d = os.path.join(root, c)
    files = os.listdir(d)
    if not files:
        raise SystemExit(f"preflight: class dir {d} is EMPTY")
    for f in random.sample(files, min(2, len(files))):
        with open(os.path.join(d, f), "rb") as fh:
            fh.read()
        n += 1
dt = time.time() - t0
print(f"preflight: {n} random reads in {dt:.1f}s ({n/dt:.1f}/s)", flush=True)
# Healthy is ~20-100 reads/s. Below 2/s the loader cannot feed 4 GPUs and the
# job would crawl rather than train -- also worth refusing.
if n / dt < 2.0:
    raise SystemExit(f"preflight: read throughput {n/dt:.2f}/s is too low")
PREFLIGHT
  # An unreadable filesystem is a TRANSIENT INFRASTRUCTURE FAULT, not a verdict
  # on the experiment, so the correct response is to give the allocation back
  # and return to the queue -- not to die (which wastes the ~6h queue wait and
  # requires a human to notice and resubmit) and not to wait in place (which
  # holds $NPROC GPUs idle, the exact failure this preflight exists to prevent).
  #
  # Bounded: PREFLIGHT_MAX_REQUEUE attempts, after which it is no longer
  # plausibly transient and a human should look. SLURM's own restart counter is
  # the state, so this survives the job being killed between attempts.
  ATTEMPT="${SLURM_RESTART_COUNT:-0}"
  MAXTRY="${PREFLIGHT_MAX_REQUEUE:-8}"
  if [ "$ATTEMPT" -lt "$MAXTRY" ]; then
    eqm_log "preflight FAILED (attempt $((ATTEMPT+1))/$MAXTRY): ImageNet at
       $IMAGENET_PATH is not readable at a usable rate, or the preflight itself
       timed out, which means reads are HANGING. Releasing the allocation and
       requeueing with a delay rather than holding $NPROC GPUs idle."
    # --begin spaces the retries out; without it the job is eligible instantly
    # and would spin through all its attempts against the same dead filesystem
    # in minutes, burning the budget before any plausible recovery window.
    scontrol requeue "$SLURM_JOB_ID" 2>/dev/null \
      && scontrol update JobId="$SLURM_JOB_ID" StartTime=now+45minutes 2>/dev/null
    exit 0
  fi
  eqm_die "ImageNet at $IMAGENET_PATH has been unreadable for $MAXTRY successive
       attempts. This is no longer plausibly transient -- it needs a human and
       probably an RC ticket. See memory/2026-08-16.md and 2026-08-17.md."
}

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
  --image-size 256 --log-every $LOG_EVERY --ckpt-every ${CKPT_EVERY:-999999999} \
  --num-workers $NUM_WORKERS \
  --results-dir $RESULTS_DIR --ebm $EBM --global-seed $GLOBAL_SEED \
  --grad-log-every ${GRAD_LOG_EVERY:-50} $BTM_ARGS $MAX_STEPS_ARG $CKPT_ARG"
echo "$CMD"

# Backgrounded via eqm_run, not foreground: bash defers trap handlers for a
# foreground command until it returns, so a 24h job would process SLURM's
# 120-second warning 24 hours late.  The inner `exit ${PIPESTATUS[0]}` preserves
# the pre-existing guarantee that the trainer's status -- not tee's -- is what
# this script exits with.
# PROGRESS WATCHDOG.
#
# The preflight above closes the one stall cause we have diagnosed. This closes
# the CLASS. The invariant is: a training job must make measurable progress or
# terminate. Anything that violates it -- a wedged filesystem, a NCCL deadlock,
# a dataloader worker race, a cause not yet seen -- is caught by the same check,
# because the check is on the observable (steps advancing) rather than on any
# particular mechanism.
#
# A stalled job is strictly worse than a failed one: it is indistinguishable
# from a slow one in `squeue`, it holds the allocation for the full wall clock,
# and it produces no evidence of why. Failing fast converts it into a dated log
# line and a freed allocation.
#
# STALL_TIMEOUT is generous by design (default 45 min). Resume from a checkpoint,
# the VAE warm-up and the first epoch's page-cache miss can all legitimately
# take many minutes, and a watchdog that fires on a slow-but-live job would be
# worse than none. It only ever fires on a job that has genuinely stopped.
STALL_TIMEOUT="${STALL_TIMEOUT:-2700}"
( # watchdog subshell -- exits on its own when train.log stops existing/growing
  last_step=""; last_change=$SECONDS
  while sleep 60; do
    # If the trainer is gone, so are we.
    kill -0 "$$" 2>/dev/null || exit 0
    [ -f train.log ] || continue
    cur=$(grep -o '(step=[0-9]*)' train.log 2>/dev/null | tail -1)
    if [ -n "$cur" ] && [ "$cur" != "$last_step" ]; then
      last_step="$cur"; last_change=$SECONDS
    fi
    if [ $((SECONDS - last_change)) -ge "$STALL_TIMEOUT" ]; then
      echo "[watchdog] FATAL: no step progress for ${STALL_TIMEOUT}s" \
           "(last seen: ${last_step:-NONE}). Killing the job rather than" \
           "holding $NPROC GPUs to the wall clock." >&2
      # Whole process group: torch.distributed.run's children must go too.
      #
      # CAVEAT, observed 2026-08-17 on probe job 39693987: a process blocked in
      # a Lustre read is in UNINTERRUPTIBLE sleep (D state) and cannot be killed
      # by any signal -- SLURM's own STEPD gave up on that job after 1215s and
      # logged "JOB NOT ENDING WITH SIGNALS". So this kill is best-effort. It
      # works for every stall whose process is still interruptible (NCCL
      # deadlock, dataloader race, livelock); for a D-state filesystem hang the
      # real defence is the PREFLIGHT above, which runs before any GPU work and
      # while the process can still be killed.
      kill -TERM -$$ 2>/dev/null || kill -TERM "$$" 2>/dev/null
      exit 1
    fi
  done
) &
WATCHDOG_PID=$!
# Deliberately NOT `trap ... EXIT` to reap this: an EXIT trap here would REPLACE
# the telemetry sealer's trap (bash keeps one handler per signal), silently
# removing the guarantee that every run emits a terminal record -- trading a
# stalled-job defect for a lost-telemetry defect. The explicit kill below covers
# the normal path; on an abnormal exit the subshell is in the job's process
# group and SLURM reaps it with everything else.
EXIT_CODE=0
eqm_run bash -c "$CMD 2>&1 | tee train.log; exit \${PIPESTATUS[0]}" || EXIT_CODE=$?
kill "$WATCHDOG_PID" 2>/dev/null || true

# Seal BEFORE `rm -rf "$WORK_DIR"`: the EXIT trap runs the sealer out of the
# checked-out tree, which lives inside WORK_DIR.  Deleting it first would leave
# the trap with no interpreter for its own fallback.  Sealing twice is safe.
eqm_seal_now "$EXIT_CODE"

cp train.log "$REPO_ROOT/projects/masked-EqM/slurm/logs/${RUN_TAG}_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
