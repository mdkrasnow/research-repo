#!/usr/bin/env bash
# Mutual exclusion for the Ralph loop, which mutates .state/pipeline.json.
# pipeline.json is the single source of truth for job tracking, so two
# simultaneous holders is a ledger-integrity failure, not an inconvenience.
#
# WHY THE PREVIOUS IMPLEMENTATION WAS NOT A MUTEX
# ------------------------------------------------
# It did:
#       mkdir "$lock_dir"           # (A) atomic test-and-set
#       echo $$ > "$lock_dir/pid"   # (B) a separate, strictly later syscall
# and decided staleness by reading that pid file. Between (A) and (B) the lock
# exists with NO owner record. A contender running the staleness check inside
# that window reads an empty pid, `is_pid_running ""` returns false, and it
# `rm -rf`s a perfectly live lock and acquires it. Two holders. The window is
# short but it is entered on EVERY acquisition, and contenders retried at 100ms,
# so hitting it is a matter of scheduling luck, not of rarity.
#
# Two further defects: `release` deleted the lock unconditionally, with no owner
# check, so a process that had already lost its lock destroyed the new owner's
# mutual exclusion; and staleness was PID-ONLY, which is unsound in three
# separate ways (see STALENESS below).
#
# THE FIX: THE OWNER RECORD *IS* THE LOCK
# ---------------------------------------
# Acquisition is a single `ln -s <owner-record> $lock_path`. POSIX symlink(2) is
# an atomic test-and-set: it fails with EEXIST if the path exists, and otherwise
# creates the link together with its target string indivisibly. We put the whole
# owner record (host | boot id | pid | process start time | epoch) INTO the
# symlink target string, so there is no separate write to race against.
#
# This is chosen over "stage a directory, then rename it into place" because
# `rename(2)` of a directory onto an existing directory is not portably a
# failure (and `mv -T` is GNU-only, while plain BSD `mv` would move the source
# *inside* the target -- a silent catastrophe here). `ln -s` needs no such
# platform branch.
#
# LINEARIZABILITY ARGUMENT
# ------------------------
# Abstract state: the lock path L is either absent or holds an owner record.
#   * ACQUIRE. The only operation that creates L is `ln -s`, which is atomic and
#     fails if L exists. Order all successful `ln -s` calls by the kernel's
#     serialization of symlink(2) on L's parent directory; at most one can
#     succeed against a given absence of L. That call is the linearization point
#     of a successful acquire; every other contender observes EEXIST and either
#     retries or fails. A failed acquire linearizes at its final failed `ln -s`.
#   * NO OWNERLESS STATE. The owner record is the link's target, created by the
#     same syscall. Therefore no reachable state has L present without a
#     complete owner record. The empty-pid window that let the old version
#     delete a live lock is not narrowed -- it is unreachable, which is the
#     property we actually need.
#   * RECLAIM. A reclaimer must first `mv` the stale L to a private, guaranteed-
#     absent name. That is rename(2) with a non-existent destination: atomic,
#     and only one of two concurrent reclaimers can succeed (the loser gets
#     ENOENT and re-polls). So a stale lock cannot be reclaimed into two
#     simultaneous acquisitions, and reclaim linearizes at that rename.
#   * RELEASE. Owner-checked: release compares the recorded (host, pid, start
#     time) against our own and refuses otherwise, so a process that already
#     lost L to a stale-reclaim cannot delete the new owner's lock. Release
#     linearizes at its `rm`.
#   * MUTUAL EXCLUSION. Between a successful acquire and its matching release,
#     L continuously holds this process's owner record: no other process can
#     create L (it exists), and no other process can remove it (reclaim requires
#     `owner_is_live` to be false, which it is not while we run; release
#     requires owner equality). Hence at most one process is in the critical
#     section at any time.
#
# STALENESS: WHY PID ALONE IS UNSOUND
# -----------------------------------
# `ps -p $pid` answers "is SOME process with this pid alive", which is not the
# question asked. It lies in three ways:
#   1. PID recycling -- the holder died and the OS reissued its pid to an
#      unrelated process. `ps` says live, the lock is never reclaimed:
#      permanent deadlock of the Ralph loop.
#   2. Another host -- this repo is rsync'd to the cluster, so a pid in a shared
#      tree may name a process on a machine whose process table we cannot see.
#      It either aliases an unrelated local pid (never reclaimed) or matches
#      nothing (a LIVE remote holder's lock gets stolen: two holders).
#   3. Reboot -- pids restart from 1, so any pre-reboot pid aliases trivially.
# The owner record therefore carries (host, boot id, pid, process start time).
# Live iff: same host AND same boot id AND pid exists AND its kernel-reported
# start time equals the recorded one. Start time is what defeats recycling: a
# recycled pid necessarily started later than the record.
# A lock owned by a DIFFERENT host is never declared dead by probing -- we
# cannot observe that host, and guessing is precisely how you get two holders.
# It is aged out by $LOCK_FOREIGN_TTL instead, the only sound option absent a
# distributed liveness protocol.
#
# Bias: every undecidable case resolves to "live". A false "dead" costs mutual
# exclusion (silent ledger corruption); a false "live" costs only waiting.

set -euo pipefail

cmd="${1:-}"
proj_dir="${2:-}"
lock_path="$proj_dir/.state/lock"

LOCK_FOREIGN_TTL="${LOCK_FOREIGN_TTL:-21600}"      # 6h; must exceed one Ralph iteration
LOCK_ACQUIRE_TRIES="${LOCK_ACQUIRE_TRIES:-50}"
LOCK_ACQUIRE_SLEEP="${LOCK_ACQUIRE_SLEEP:-0.1}"

# The owner is the process holding the critical section -- the CALLER -- not
# this script, which exits the instant `acquire` returns. The old code recorded
# `$$` (lock.sh's own pid), so every lock it took was already stale by the time
# the caller entered its critical section: any contender would find the pid dead
# and reclaim immediately. Recording $PPID is what makes the liveness signal
# refer to the entity whose progress the lock is actually protecting. Callers
# that fork (or invoke lock.sh from a subshell) can state the owner explicitly.
OWNER_PID="${LOCK_OWNER_PID:-$PPID}"

die() { echo "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# Identity primitives
# --------------------------------------------------------------------------

host_id() {
  # Prefer a stable machine id: a hostname can be reassigned or aliased.
  if [[ -r /etc/machine-id ]]; then
    tr -d '[:space:]|' < /etc/machine-id
  elif [[ -r /var/lib/dbus/machine-id ]]; then
    tr -d '[:space:]|' < /var/lib/dbus/machine-id
  else
    hostname | tr -d '[:space:]|'
  fi
}

boot_id() {
  # Distinguishes pid namespaces across reboots. macOS has no boot_id, but the
  # kernel boot time is an exact equivalent for this purpose.
  if [[ -r /proc/sys/kernel/random/boot_id ]]; then
    tr -d '[:space:]|' < /proc/sys/kernel/random/boot_id
  elif sysctl -n kern.boottime >/dev/null 2>&1; then
    sysctl -n kern.boottime | tr -cd '0-9,'
  else
    echo "unknown-boot"
  fi
}

# Kernel-reported start time of $1; empty if no such process. The value only has
# to compare equal for the same process on the same host, which both the /proc
# starttime jiffies field and the macOS `ps -o lstart=` string satisfy.
proc_start_time() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if [[ -r "/proc/$pid/stat" ]]; then
    # The comm field is parenthesized and may itself contain ')' and spaces, so
    # strip through the LAST ')' before splitting -- the standard fix for the
    # classic /proc/<pid>/stat parsing bug. After stripping, starttime is $20.
    local rest
    rest="$(sed -e 's/^.*) //' "/proc/$pid/stat" 2>/dev/null || true)"
    [[ -n "$rest" ]] || return 0
    awk '{print $20}' <<<"$rest"
  else
    ps -o lstart= -p "$pid" 2>/dev/null | tr -s ' ' | tr -d '|' || true
  fi
}

# --------------------------------------------------------------------------
# Owner record, carried entirely inside the symlink target string.
# Fields: host | boot | pid | start_time | epoch_seconds | iso8601
# --------------------------------------------------------------------------

own_record() {
  local pid="$OWNER_PID"
  printf '%s|%s|%s|%s|%s|%s' \
    "$(host_id)" "$(boot_id)" "$pid" "$(proc_start_time "$pid")" \
    "$(date -u +%s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# Populates O_HOST/O_BOOT/O_PID/O_START/O_EPOCH/O_ISO. Returns 1 if unreadable.
read_owner_record() {
  local path="$1" line=""
  if [[ -L "$path" ]]; then
    line="$(readlink "$path" 2>/dev/null || true)"
  elif [[ -d "$path" ]]; then
    # Legacy on-disk form (directory + pid file) written by the old, broken
    # implementation. Recognized so an in-flight upgrade does not deadlock.
    local pid; pid="$(cat "$path/pid" 2>/dev/null || true)"
    [[ -n "$pid" ]] || return 1
    line="legacy|legacy|$pid|||$(cat "$path/acquired_at" 2>/dev/null || true)"
  fi
  [[ -n "$line" ]] || return 1
  IFS='|' read -r O_HOST O_BOOT O_PID O_START O_EPOCH O_ISO <<<"$line"
  [[ -n "${O_PID:-}" ]] || return 1
  return 0
}

lock_exists() { [[ -L "$lock_path" || -e "$lock_path" ]]; }

# Exit 0 iff the recorded owner is (or may be) still running.
owner_is_live() {
  local path="$1"
  read_owner_record "$path" || return 0        # unreadable -> assume live

  if [[ "$O_HOST" == "legacy" ]]; then
    # Old-format lock: we have only a pid, which is exactly the unsound signal
    # this rewrite exists to eliminate. Probe it, but never trust "live"
    # indefinitely -- fall through to the pid check and let a stale one go.
    ps -p "$O_PID" >/dev/null 2>&1 && return 0
    return 1
  fi

  if [[ "$O_HOST" != "$(host_id)" ]]; then
    local now; now="$(date -u +%s)"
    [[ -n "${O_EPOCH:-}" ]] || return 0
    (( now - O_EPOCH < LOCK_FOREIGN_TTL )) && return 0
    return 1                                   # foreign and older than TTL
  fi

  [[ "$O_BOOT" == "$(boot_id)" ]] || return 1  # pre-reboot pid namespace

  ps -p "$O_PID" >/dev/null 2>&1 || return 1

  local now_start; now_start="$(proc_start_time "$O_PID")"
  if [[ -n "${O_START:-}" && -n "$now_start" && "$O_START" != "$now_start" ]]; then
    return 1                                   # pid recycled: different process
  fi
  return 0
}

# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

case "$cmd" in
  acquire)
    [[ -n "$proj_dir" ]] || die "Usage: lock.sh acquire|release|owner <project_dir>"
    mkdir -p "$proj_dir/.state"

    for _ in $(seq 1 "$LOCK_ACQUIRE_TRIES"); do
      if lock_exists && ! owner_is_live "$lock_path"; then
        # Reclaim via rename to a guaranteed-absent private name: atomic, and
        # only one of several concurrent reclaimers can win it.
        reclaim="$proj_dir/.state/.lock.stale.$$.${RANDOM}${RANDOM}"
        if mv "$lock_path" "$reclaim" 2>/dev/null; then
          rm -rf "$reclaim" || true
        fi
      fi

      # THE linearization point. Atomic test-and-set; the owner record is the
      # link target, so the lock cannot exist without it.
      if ln -s "$(own_record)" "$lock_path" 2>/dev/null; then
        exit 0
      fi
      sleep "$LOCK_ACQUIRE_SLEEP"
    done

    echo "Could not acquire lock for $proj_dir" >&2
    if read_owner_record "$lock_path" 2>/dev/null; then
      echo "  held by host=$O_HOST pid=$O_PID since=${O_ISO:-unknown}" >&2
    fi
    exit 1
    ;;

  release)
    [[ -n "$proj_dir" ]] || die "Usage: lock.sh acquire|release|owner <project_dir>"
    lock_exists || exit 0

    if read_owner_record "$lock_path"; then
      my_start="$(proc_start_time "$OWNER_PID")"
      if [[ "$O_HOST" != "$(host_id)" || "$O_PID" != "$OWNER_PID" ]] \
         || { [[ -n "${O_START:-}" && -n "$my_start" ]] && [[ "$O_START" != "$my_start" ]]; }; then
        echo "lock.sh release: refusing to release lock owned by host=$O_HOST pid=$O_PID" \
             "(we are host=$(host_id) pid=$OWNER_PID)" >&2
        exit 3
      fi
    fi
    rm -rf "$lock_path" || true
    ;;

  owner)
    # Introspection for operators and for the concurrency test.
    lock_exists || { echo "unlocked"; exit 1; }
    read_owner_record "$lock_path" || { echo "corrupt"; exit 2; }
    if owner_is_live "$lock_path"; then state=live; else state=stale; fi
    echo "$state host=$O_HOST boot=$O_BOOT pid=$O_PID start=${O_START:-} since=${O_ISO:-}"
    ;;

  *)
    echo "Usage: lock.sh acquire|release|owner <project_dir>" >&2
    exit 1
    ;;
esac
