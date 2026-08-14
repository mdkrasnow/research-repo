#!/bin/bash
# shellcheck shell=bash
# =============================================================================
# telemetry_env.sh -- the shell half of the masked-EqM telemetry contract.
#
# Source this near the top of an sbatch, after the module loads and after the
# repo checkout (it must be able to `python -m telemetry.*`):
#
#     source "$REPO_DIR/slurm/lib/telemetry_env.sh"
#     eqm_telemetry_begin --campaign btm --phase II --arm "$BTM_MODE" \
#         --seed "$GLOBAL_SEED" --git-sha "$GIT_SHA" \
#         --param fd_k="$FD_K" --param fd_eps="$FD_EPS"
#     eqm_run bash -c "$CMD 2>&1 | tee train.log; exit \${PIPESTATUS[0]}"
#
# Each block below exists to close ONE audited failure mode.  The mapping:
#
# (A) NO ADVANCE WARNING OF THE WALL CLOCK.  Zero of 92 sbatch files carried a
#     `#SBATCH --signal` directive, so a job that hit its time limit was killed
#     with no notice and left a stream that simply stops -- byte-identically
#     shaped to a run that was still going.  The prelude cannot add the directive
#     (that is per-file, see README), but it supplies the other half: signal
#     handlers that turn a warning into a terminal record.  See `eqm_run`.
#
# (B) EXIT STATUS LOST IN A PIPELINE.  `python ... | tee log` reports tee's
#     status unless `pipefail` is set.  Every current file does set it (see
#     README: the audit's claim here is a false positive), but the property is
#     one edit away from being lost, so `eqm_require_pipefail` asserts it at
#     source time rather than trusting it.  `eqm_run` additionally captures the
#     status explicitly so the sealer gets the trainer's status, not the shell's.
#
# (C) REQUEUE TRUNCATES STDOUT.  Without `#SBATCH --open-mode=append` SLURM
#     opens the log with O_TRUNC on every attempt, so a requeue destroys the
#     evidence of why the first attempt died.  Per-file directive; the prelude
#     warns loudly if it detects it is running an attempt > 0 without it.
#
# (D) SEVEN SPELLINGS OF THE RESULTS DIRECTORY.  RESULTS_ROOT / OUT_DIR /
#     OUT_ROOT / OUTPUT_DIR / OUTPUT_ROOT / OUT / RESULTS all named the same
#     concept, so a submitter who exported the wrong one silently got the
#     default.  `eqm_normalize_results_root` makes RESULTS_ROOT canonical and
#     accepts the legacy names with a deprecation warning, so no in-flight
#     submission script breaks while the corpus migrates.
#
# (E) TWO IMAGENET DEFAULTS.  33 files defaulted to the holylfs06 kempner copy,
#     8 to a stale holylabs copy.  Two arms defaulting differently is an
#     uncontrolled variable in an A/B comparison that nothing in the pipeline
#     would have flagged.  `eqm_resolve_imagenet` pins the correct one and
#     refuses (not warns) if pointed at the retired path without an explicit
#     override, because "trained on a different dataset copy" is not a warning.
#
# (F) CONFIG VIA POSITIONAL ARGS.  `sacct` records the script path but not its
#     argv, so a run launched as `sbatch job.sbatch 25 25` has an unrecoverable
#     epoch target.  `eqm_forbid_positional` makes that fail at submit time.
#     Everything identity-bearing must arrive as an env var, which is also what
#     lets it be hashed into the run_uid.
#
# (G) NONDETERMINISTIC MASTER_PORT.  `$((29500 + RANDOM % 1000))` is not
#     reproducible and not recoverable from any record, so a rerun cannot be made
#     byte-identical and a port collision cannot be diagnosed after the fact.
#     Derived from SLURM_JOB_ID instead: still collision-avoiding across
#     concurrent jobs (they have distinct ids), but a pure function of a value
#     sacct retains.
#
# Layer 5 of the terminal-record ladder.  Layers 1-4 live in
# telemetry/lifecycle.py and cover every death python can observe.  This file
# covers the ones it cannot: SIGKILL, node death, and a wall-clock kill that
# lands before RunRecorder.__enter__ installed its handlers.
# =============================================================================

# Guard against double-sourcing: the trap and the signal state are not
# idempotent, and a second `trap ... EXIT` would silently replace the first.
if [ -n "${_EQM_TELEMETRY_ENV_SOURCED:-}" ]; then
  return 0 2>/dev/null || exit 0
fi
_EQM_TELEMETRY_ENV_SOURCED=1

# ---------------------------------------------------------------------------
# Locating ourselves.  BASH_SOURCE, not $0: under `source` $0 is still sbatch's
# script, and the cluster runs from a fresh /tmp clone whose path is per-job, so
# nothing may be hardcoded.
# ---------------------------------------------------------------------------
# ${BASH_SOURCE[0]:-$0} not ${BASH_SOURCE[0]}: under `set -u` an unset
# BASH_SOURCE (this file sourced by a non-bash shell, or executed rather than
# sourced) would abort here with a message that names bash internals rather
# than the actual mistake.
_EQM_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# slurm/lib -> slurm -> projects/masked-EqM (the dir that contains telemetry/)
EQM_PROJECT_DIR="${EQM_PROJECT_DIR:-$(cd "$_EQM_LIB_DIR/../.." && pwd)}"
export EQM_PROJECT_DIR

eqm_log()  { echo "[telemetry_env] $*" >&2; }
eqm_warn() { echo "[telemetry_env] WARN: $*" >&2; }
# Exit 78 = EX_CONFIG (sysexits.h).  A distinct code so that "the launcher was
# misconfigured" is separable in sacct from "the science failed", which a bare
# `exit 1` would conflate.
eqm_die()  { echo "[telemetry_env] FATAL: $*" >&2; exit 78; }

# ---------------------------------------------------------------------------
# (B) pipefail assertion.
# ---------------------------------------------------------------------------
eqm_require_pipefail() {
  case "$(set -o | grep '^pipefail' | awk '{print $2}')" in
    on) : ;;
    *)  eqm_die "pipefail is not set. Without it, 'python ... | tee log' exits
       with tee's status and a crashed trainer reports COMPLETED 0:0 to sacct.
       Put 'set -euo pipefail' above this source line." ;;
  esac
}

# ---------------------------------------------------------------------------
# (F) positional-argument ban.
# ---------------------------------------------------------------------------
eqm_forbid_positional() {
  if [ "$#" -gt 0 ]; then
    eqm_die "this job was given positional arguments ($*). sacct does not record
       argv, so a run configured positionally has an unrecoverable configuration.
       Pass every knob as an environment variable instead."
  fi
}

# ---------------------------------------------------------------------------
# (D) results-root normalization.
# ---------------------------------------------------------------------------
# Legacy spellings, most-specific first.  RESULTS_DIR is intentionally NOT in
# this list: it names a per-job leaf directory, a genuinely different concept
# from the root that leaves live under, and folding the two would silently nest
# job directories inside each other.
_EQM_LEGACY_ROOT_VARS="OUT_ROOT OUTPUT_ROOT OUT_DIR OUTPUT_DIR RESULTS OUT"

eqm_normalize_results_root() {
  local default_root="${1:-}"
  local var value chosen="" chosen_var=""
  if [ -n "${RESULTS_ROOT:-}" ]; then
    chosen="$RESULTS_ROOT"; chosen_var="RESULTS_ROOT"
  else
    for var in $_EQM_LEGACY_ROOT_VARS; do
      # Bash indirect expansion, not `eval`: the value may be a path with
      # spaces, and an eval would re-parse it as shell syntax.
      value="${!var:-}"
      [ -n "$value" ] || continue
      eqm_warn "\$$var is a deprecated spelling of \$RESULTS_ROOT; using it for
       now, but export RESULTS_ROOT instead. Seven spellings of one concept is
       how a submitter silently gets the default results directory."
      chosen="$value"; chosen_var="$var"
      break
    done
  fi
  if [ -z "$chosen" ]; then
    [ -n "$default_root" ] || eqm_die "no RESULTS_ROOT and no default supplied"
    chosen="$default_root"; chosen_var="(default)"
  fi
  RESULTS_ROOT="$chosen"
  export RESULTS_ROOT
  # Deliberately NOT aliasing the other six names back to RESULTS_ROOT.  In
  # several files OUT_DIR names a per-job LEAF rather than a root, so blanket
  # aliasing would relocate outputs -- trading a naming defect for a data-loss
  # defect.  Migration is per-file and mechanical (see slurm/lib/README.md).
  eqm_log "RESULTS_ROOT=$RESULTS_ROOT (from $chosen_var)"
}

# ---------------------------------------------------------------------------
# (E) ImageNet path.
# ---------------------------------------------------------------------------
EQM_IMAGENET_CANONICAL="/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train"
EQM_IMAGENET_RETIRED="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train"

eqm_resolve_imagenet() {
  IMAGENET_PATH="${IMAGENET_PATH:-$EQM_IMAGENET_CANONICAL}"
  if [ "$IMAGENET_PATH" = "$EQM_IMAGENET_RETIRED" ] && \
     [ -z "${EQM_ALLOW_RETIRED_IMAGENET:-}" ]; then
    eqm_die "IMAGENET_PATH points at the retired copy
       $EQM_IMAGENET_RETIRED
       while the rest of the corpus defaults to
       $EQM_IMAGENET_CANONICAL
       Two arms training on two dataset copies is an uncontrolled variable that
       no downstream check would catch. Set EQM_ALLOW_RETIRED_IMAGENET=1 if this
       is deliberate (e.g. reproducing an old run) and say so in pipeline.json."
  fi
  export IMAGENET_PATH
  eqm_log "IMAGENET_PATH=$IMAGENET_PATH"
}

# ---------------------------------------------------------------------------
# (G) deterministic MASTER_PORT.
# ---------------------------------------------------------------------------
eqm_set_master_port() {
  local base="${1:-29500}"
  if [ -n "${MASTER_PORT:-}" ]; then
    eqm_log "MASTER_PORT=$MASTER_PORT (caller-supplied)"
    return 0
  fi
  local jid="${SLURM_JOB_ID:-0}"
  # Array tasks share SLURM_ARRAY_JOB_ID but land on possibly the same node, so
  # fold the task id in as well; SLURM_JOB_ID is already unique per task, this
  # is belt-and-braces for the case where a caller exported the array job id.
  local tid="${SLURM_ARRAY_TASK_ID:-0}"
  MASTER_PORT=$(( base + ( (jid + tid * 7919) % 1000 ) ))
  export MASTER_PORT
  eqm_log "MASTER_PORT=$MASTER_PORT (deterministic from job $jid task $tid)"
}

# ---------------------------------------------------------------------------
# (C) requeue / append-mode check.
# ---------------------------------------------------------------------------
eqm_check_open_mode() {
  local restarts="${SLURM_RESTART_COUNT:-0}"
  if [ "${restarts:-0}" -gt 0 ]; then
    eqm_warn "this is attempt #$restarts of job ${SLURM_JOB_ID:-?}. If this
       sbatch lacks '#SBATCH --open-mode=append', the previous attempt's stdout
       has just been TRUNCATED and the evidence of why it died is gone."
  fi
}

# ---------------------------------------------------------------------------
# Telemetry root + spec.
# ---------------------------------------------------------------------------
# Default under RESULTS_ROOT so telemetry travels with the results it describes,
# but as a sibling of the per-job leaves (not inside one), because a stream must
# outlive the job directory it was produced by: the whole point of a run_uid is
# that a requeued attempt appends to the SAME logical run's directory.
eqm_set_telemetry_root() {
  if [ -z "${EQM_TELEMETRY_ROOT:-}" ]; then
    [ -n "${RESULTS_ROOT:-}" ] || eqm_die "call eqm_normalize_results_root before eqm_set_telemetry_root"
    EQM_TELEMETRY_ROOT="$RESULTS_ROOT/_telemetry"
  fi
  export EQM_TELEMETRY_ROOT
  mkdir -p "$EQM_TELEMETRY_ROOT" || eqm_die "cannot create $EQM_TELEMETRY_ROOT"
  eqm_log "EQM_TELEMETRY_ROOT=$EQM_TELEMETRY_ROOT"
}

# eqm_telemetry_begin --campaign C --arm A --seed N --git-sha S
#                     [--phase P] [--planned-steps N] [--param k=v]...
#
# Mints EQM_RUN_SPEC by delegating to telemetry.mkspec.  The JSON is NEVER
# assembled here: RunSpec.from_env() content-hashes whatever it is handed, so
# bash quoting would become part of the experiment's identity.  One serializer,
# shared by launcher and trainer, is what keeps the two views identical.
eqm_telemetry_begin() {
  eqm_require_pipefail
  eqm_check_open_mode
  [ -n "${RESULTS_ROOT:-}" ] || eqm_die "call eqm_normalize_results_root first"
  eqm_set_telemetry_root

  local -a mkspec_args=()
  local have_campaign=0 have_arm=0 have_seed=0 have_sha=0
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --campaign)      have_campaign=1; mkspec_args+=(--campaign "$2"); shift 2 ;;
      --arm)           have_arm=1;      mkspec_args+=(--arm "$2");      shift 2 ;;
      --seed)          have_seed=1;     mkspec_args+=(--seed "$2");     shift 2 ;;
      --git-sha)       have_sha=1;      mkspec_args+=(--git-sha "$2");  shift 2 ;;
      --phase)         mkspec_args+=(--phase "$2");                     shift 2 ;;
      --planned-steps) mkspec_args+=(--planned-steps "$2");             shift 2 ;;
      --param)         mkspec_args+=(--param "$2");                     shift 2 ;;
      *) eqm_die "eqm_telemetry_begin: unknown argument $1" ;;
    esac
  done
  [ "$have_campaign" = 1 ] || eqm_die "eqm_telemetry_begin requires --campaign"
  [ "$have_arm" = 1 ]      || eqm_die "eqm_telemetry_begin requires --arm"
  [ "$have_seed" = 1 ]     || eqm_die "eqm_telemetry_begin requires --seed"
  [ "$have_sha" = 1 ]      || eqm_die "eqm_telemetry_begin requires --git-sha (an
       unpinned commit makes a result unreproducible and unattributable)"

  EQM_RUN_SPEC="$( cd "$EQM_PROJECT_DIR" && python -m telemetry.mkspec "${mkspec_args[@]}" )" \
    || eqm_die "telemetry.mkspec failed; refusing to run an experiment whose
       identity cannot be minted (an unidentified run is an unanalyzable run)"
  export EQM_RUN_SPEC
  EQM_RUN_UID="$( cd "$EQM_PROJECT_DIR" && python -m telemetry.mkspec "${mkspec_args[@]}" --print-uid )" || EQM_RUN_UID=""
  export EQM_RUN_UID
  eqm_log "EQM_RUN_UID=$EQM_RUN_UID"

  _eqm_install_traps
}

# ---------------------------------------------------------------------------
# The trap ladder.
# ---------------------------------------------------------------------------
# _EQM_SIGNAL records which signal the SHELL observed, which is better evidence
# than the child's wait status: in `python | tee`, a SIGUSR1 delivered to the
# batch shell never appears in any exit code at all.
_EQM_SIGNAL=""
_EQM_SIGNAL_SEEN=0

# Every descendant pid of $1, excluding $1 itself, computed from a SINGLE ps
# snapshot.  A transitive walk is required, not `pkill -P`: the process that must
# hear the signal is `python train.py`, which under
# `eqm_run bash -c "... | tee ..."` plus `torch.distributed.run` sits three or
# four levels below this shell.  Signalling only the direct child hits `bash` or
# `tee` and the trainer never learns it is about to die.
_eqm_descendants() {
  local root="$1"
  ps -Ao pid=,ppid= 2>/dev/null | awk -v root="$root" '
    { child[NR] = $1; parent[NR] = $2; n = NR }
    END {
      want[root] = 1
      # Repeat to transitive closure.  n passes always suffice, and n is a few
      # hundred, so this is cheaper than one extra fork.
      for (pass = 0; pass < n; pass++)
        for (i = 1; i <= n; i++)
          if (parent[i] in want) want[child[i]] = 1
      for (i = 1; i <= n; i++)
        if (child[i] in want && child[i] != root) print child[i]
    }'
}

_eqm_on_signal() {
  local sig="$1"
  # Re-entrancy guard.  The process-group fallback below includes this shell, so
  # without the guard the handler recurses and the seal never runs -- a hang at
  # exactly the moment the terminal record matters most.
  if [ "$_EQM_SIGNAL_SEEN" = "1" ]; then return 0; fi
  _EQM_SIGNAL_SEEN=1
  _EQM_SIGNAL="$sig"
  eqm_warn "received SIG$sig; forwarding to the job's descendants so the
       trainer's own handler (telemetry.lifecycle) can seal from inside, with
       the shell-level sealer as fallback."
  local pid
  for pid in $(_eqm_descendants "$$"); do
    kill -s "$sig" "$pid" 2>/dev/null || true
  done
  # Belt and braces: on SLURM the batch script is a session leader, so the
  # process group is exactly the job.  Off SLURM (a local dry run) $$ is usually
  # not a pgid and this is a harmless no-op -- which is why it is a FALLBACK and
  # not the primary mechanism.  The earlier version relied on it alone and
  # silently delivered nothing in a non-session-leader shell.
  kill -s "$sig" -- "-$$" 2>/dev/null || true
}

_eqm_on_exit() {
  local code=$?
  # Restore defaults first so a second signal during teardown kills us outright
  # rather than re-entering a handler.
  trap - EXIT TERM USR1 INT
  if [ -n "${EQM_RUN_SPEC:-}" ] && [ -n "${EQM_TELEMETRY_ROOT:-}" ]; then
    local -a seal_args=(
      --root "$EQM_TELEMETRY_ROOT" --run-spec "$EQM_RUN_SPEC"
      --job-id "${SLURM_JOB_ID:-local}" --exit-code "$code" --status auto
    )
    # if/then, not `[ ... ] && ...`: a false test as the last command of a
    # function under `set -e` would abort the trap before it reaches `exit`.
    if [ -n "$_EQM_SIGNAL" ]; then seal_args+=(--signal "$_EQM_SIGNAL"); fi
    # `|| true` is mandatory, not defensive sloppiness: this runs in an EXIT
    # trap, so a nonzero status here would REPLACE the job's real exit status and
    # the telemetry system would be manufacturing the failures it reports.
    ( cd "$EQM_PROJECT_DIR" && python -m telemetry.seal "${seal_args[@]}" ) || true
  fi
  exit "$code"
}

# eqm_seal_now <exit_code> -- run the sealer explicitly, ahead of the EXIT trap.
#
# Needed by any job that deletes its own working tree before exiting: the trap
# invokes `python -m telemetry.seal` out of $EQM_PROJECT_DIR, which for a
# clone-to-/tmp job is inside the directory `rm -rf` is about to remove.  Calling
# this before cleanup seals while the tooling still exists; the EXIT trap then
# finds an END already present and does nothing, which is exactly the idempotence
# telemetry/seal.py is built around.
eqm_seal_now() {
  local code="${1:-0}"
  if [ -z "${EQM_RUN_SPEC:-}" ] || [ -z "${EQM_TELEMETRY_ROOT:-}" ]; then
    return 0
  fi
  local -a seal_args=(
    --root "$EQM_TELEMETRY_ROOT" --run-spec "$EQM_RUN_SPEC"
    --job-id "${SLURM_JOB_ID:-local}" --exit-code "$code" --status auto
  )
  if [ -n "$_EQM_SIGNAL" ]; then seal_args+=(--signal "$_EQM_SIGNAL"); fi
  ( cd "$EQM_PROJECT_DIR" && python -m telemetry.seal "${seal_args[@]}" ) || true
}

_eqm_install_traps() {
  trap '_eqm_on_signal USR1' USR1
  trap '_eqm_on_signal TERM' TERM
  trap '_eqm_on_signal INT'  INT
  trap '_eqm_on_exit' EXIT
}

# ---------------------------------------------------------------------------
# eqm_run -- run the scientific command so that signals are actually deliverable.
# ---------------------------------------------------------------------------
# bash defers trap handlers for a FOREGROUND command until that command
# completes.  A 24-hour training job run in the foreground would therefore
# process SLURM's 120-second warning 24 hours late, i.e. never -- the directive
# would be present and inert, which is worse than absent because it looks
# handled.  Backgrounding and `wait`ing is what makes the trap fire promptly:
# `wait` is interruptible, returns >128 when a trapped signal arrives, and is
# then resumed.
#
# Returns the command's own status, which the EXIT trap turns into a status.
eqm_run() {
  "$@" &
  EQM_CHILD_PID=$!
  export EQM_CHILD_PID
  local status=0
  while :; do
    wait "$EQM_CHILD_PID"
    status=$?
    # >128 means `wait` itself was interrupted by a trapped signal.  If the child
    # is still alive it was the SHELL that got signalled (SLURM's B: warning), so
    # keep waiting: the child has been forwarded the signal and is now sealing
    # its own stream, and we want its real status, not 128+n.
    if [ "$status" -gt 128 ] && kill -0 "$EQM_CHILD_PID" 2>/dev/null; then
      continue
    fi
    break
  done
  return "$status"
}

# ---------------------------------------------------------------------------
# One-call convenience for the common case.
# ---------------------------------------------------------------------------
# `eqm_prelude <default_results_root>` does the four context-free fixes (D, E, G
# and the C check).  Identity still requires an explicit eqm_telemetry_begin,
# because campaign/arm/seed cannot be guessed and guessing them is precisely the
# path-decoding defect the telemetry package was written to eliminate.
eqm_prelude() {
  eqm_require_pipefail
  eqm_normalize_results_root "${1:-}"
  eqm_resolve_imagenet
  eqm_set_master_port
  eqm_check_open_mode
}
