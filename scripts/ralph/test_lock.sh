#!/usr/bin/env bash
# Tests for scripts/ralph/lock.sh.
#
# The headline test (T1) is a real concurrency test: N background subshells
# contend for the lock and each one, while holding it, does a
# read-modify-write on a shared counter with a deliberate delay between the
# read and the write. Under correct mutual exclusion the final counter equals
# N exactly; under any interleaving of two holders it is strictly less. That
# is a *falsifiable* assertion about the property we care about, as opposed to
# checking that acquire returns 0.
#
# T1 is also validated as a test: T0 runs the identical protocol with a
# deliberately broken (no-op) lock and asserts the counter DOES get corrupted.
# Without that negative control, a passing T1 could just mean the contention
# window was never entered and the test proves nothing.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$HERE/lock.sh"
PASS=0; FAIL=0

ok()   { echo "  PASS  $*"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }
check(){ if [[ "$2" == "$3" ]]; then ok "$1 ($2)"; else bad "$1: expected '$3' got '$2'"; fi; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

NWORKERS="${NWORKERS:-12}"

# --------------------------------------------------------------------------
# Shared critical section used by T0/T1.
#   $1 = project dir, $2 = counter file, $3 = "lock" | "nolock"
# --------------------------------------------------------------------------
contend() {
  local proj="$1" counter="$2" mode="$3"
  if [[ "$mode" == lock ]]; then
    LOCK_ACQUIRE_TRIES=400 LOCK_ACQUIRE_SLEEP=0.02 \
      LOCK_OWNER_PID=$$ bash "$LOCK" acquire "$proj" || return 1
  fi
  # Non-atomic read-modify-write: correct only under mutual exclusion.
  local v; v="$(cat "$counter")"
  sleep 0.05
  echo $((v + 1)) > "$counter"
  if [[ "$mode" == lock ]]; then
    LOCK_OWNER_PID=$$ bash "$LOCK" release "$proj" || return 1
  fi
}

echo "T0 negative control: without a lock the counter must be corrupted"
proj="$TMP/t0"; mkdir -p "$proj/.state"; echo 0 > "$proj/counter"
for _ in $(seq 1 "$NWORKERS"); do contend "$proj" "$proj/counter" nolock & done
wait
got="$(cat "$proj/counter")"
if (( got < NWORKERS )); then
  ok "unlocked counter lost updates ($got < $NWORKERS) -- the race is reachable"
else
  bad "unlocked counter reached $NWORKERS: the test never entered the race, so T1 proves nothing"
fi

echo "T1 mutual exclusion under $NWORKERS concurrent contenders"
proj="$TMP/t1"; mkdir -p "$proj/.state"; echo 0 > "$proj/counter"
for _ in $(seq 1 "$NWORKERS"); do contend "$proj" "$proj/counter" lock & done
wait
check "every increment survived" "$(cat "$proj/counter")" "$NWORKERS"
if [[ -e "$proj/.state/lock" || -L "$proj/.state/lock" ]]; then
  bad "lock still held after all contenders released"
else
  ok "lock released"
fi

echo "T2 the lock never exists without an owner record"
# The old bug: mkdir then a later `echo \$\$ > pid`. Poll hard during a live
# acquire; every observation of the lock path must also yield a readable owner.
proj="$TMP/t2"; mkdir -p "$proj/.state"
violations=0
(
  for _ in $(seq 1 60); do
    LOCK_OWNER_PID=$$ bash "$LOCK" acquire "$proj" >/dev/null 2>&1
    LOCK_OWNER_PID=$$ bash "$LOCK" release "$proj" >/dev/null 2>&1
  done
) &
churn=$!
observations=0
while kill -0 "$churn" 2>/dev/null; do
  # ONE syscall per observation. A test/read pair would be racy on its own
  # account (the lock can be released between them) and would report test
  # artifacts as lock defects. readlink returns 0 iff the link exists, and its
  # output is the owner record, so "exists but ownerless" is exactly
  # "rc==0 && output empty".
  target="$(readlink "$proj/.state/lock" 2>/dev/null)"; rc=$?
  if (( rc == 0 )); then
    observations=$((observations+1))
    [[ -n "$target" ]] || violations=$((violations+1))
  elif [[ -d "$proj/.state/lock" ]]; then
    observations=$((observations+1))
    violations=$((violations+1))   # directory form => a staged/partial lock
  fi
done
wait "$churn" 2>/dev/null
check "ownerless-lock observations" "$violations" "0"
if (( observations > 0 )); then
  ok "held-lock observations: $observations (the window was actually sampled)"
else
  bad "never observed the lock held: this test proves nothing"
fi

echo "T3 release is owner-checked"
proj="$TMP/t3"; mkdir -p "$proj/.state"
# Owner is a live sleeper; a different process must not be able to release it.
sleep 30 & holder=$!
LOCK_OWNER_PID=$holder bash "$LOCK" acquire "$proj" >/dev/null
out="$(bash "$LOCK" release "$proj" 2>&1)"; rc=$?
check "non-owner release rejected (rc)" "$rc" "3"
if [[ -L "$proj/.state/lock" ]]; then ok "lock survived the foreign release"
else bad "lock was destroyed by a non-owner"; fi
LOCK_OWNER_PID=$holder bash "$LOCK" release "$proj" >/dev/null
if [[ -L "$proj/.state/lock" ]]; then bad "owner release did not remove the lock"
else ok "owner release removed the lock"; fi
kill "$holder" 2>/dev/null; wait "$holder" 2>/dev/null

echo "T4 a dead owner's lock is reclaimed"
proj="$TMP/t4"; mkdir -p "$proj/.state"
sleep 30 & victim=$!
LOCK_OWNER_PID=$victim bash "$LOCK" acquire "$proj" >/dev/null
kill -9 "$victim" 2>/dev/null; wait "$victim" 2>/dev/null
LOCK_ACQUIRE_TRIES=20 LOCK_OWNER_PID=$$ bash "$LOCK" acquire "$proj" >/dev/null; rc=$?
check "stale lock reclaimed (rc)" "$rc" "0"
LOCK_OWNER_PID=$$ bash "$LOCK" release "$proj" >/dev/null

echo "T5 a LIVE owner's lock is not stolen"
proj="$TMP/t5"; mkdir -p "$proj/.state"
sleep 30 & holder=$!
LOCK_OWNER_PID=$holder bash "$LOCK" acquire "$proj" >/dev/null
LOCK_ACQUIRE_TRIES=3 LOCK_ACQUIRE_SLEEP=0.01 LOCK_OWNER_PID=$$ \
  bash "$LOCK" acquire "$proj" >/dev/null 2>&1; rc=$?
check "contender blocked by live owner (rc)" "$rc" "1"
kill "$holder" 2>/dev/null; wait "$holder" 2>/dev/null
LOCK_OWNER_PID=$holder bash "$LOCK" release "$proj" >/dev/null 2>&1

echo "T6 pid recycling: same pid, different process => stale"
# Forge an owner record with a live pid but a WRONG start time. Under the old
# PID-only rule this reads as live forever (permanent deadlock); the start-time
# field must make it stale.
proj="$TMP/t6"; mkdir -p "$proj/.state"
sleep 30 & live=$!
host="$(cat /etc/machine-id 2>/dev/null || hostname)"; host="${host//[[:space:]|]/}"
if [[ -r /proc/sys/kernel/random/boot_id ]]; then boot="$(tr -d '[:space:]|' < /proc/sys/kernel/random/boot_id)"
elif sysctl -n kern.boottime >/dev/null 2>&1; then boot="$(sysctl -n kern.boottime | tr -cd '0-9,')"
else boot="unknown-boot"; fi
ln -s "$host|$boot|$live|0000000000|$(date -u +%s)|forged" "$proj/.state/lock"
state="$(bash "$LOCK" owner "$proj" | awk '{print $1}')"
check "recycled-pid owner reported" "$state" "stale"
LOCK_ACQUIRE_TRIES=20 LOCK_OWNER_PID=$$ bash "$LOCK" acquire "$proj" >/dev/null; rc=$?
check "recycled-pid lock reclaimed (rc)" "$rc" "0"
LOCK_OWNER_PID=$$ bash "$LOCK" release "$proj" >/dev/null
kill "$live" 2>/dev/null; wait "$live" 2>/dev/null

echo "T7 foreign-host lock: never probed, aged out on TTL only"
proj="$TMP/t7"; mkdir -p "$proj/.state"
ln -s "some-other-host|some-boot|1|1|$(date -u +%s)|now" "$proj/.state/lock"
LOCK_ACQUIRE_TRIES=2 LOCK_ACQUIRE_SLEEP=0.01 LOCK_OWNER_PID=$$ \
  bash "$LOCK" acquire "$proj" >/dev/null 2>&1; rc=$?
check "fresh foreign lock respected (rc)" "$rc" "1"
rm -f "$proj/.state/lock"
ln -s "some-other-host|some-boot|1|1|$(( $(date -u +%s) - 999999 ))|old" "$proj/.state/lock"
LOCK_ACQUIRE_TRIES=5 LOCK_OWNER_PID=$$ bash "$LOCK" acquire "$proj" >/dev/null 2>&1; rc=$?
check "expired foreign lock reclaimed (rc)" "$rc" "0"
LOCK_OWNER_PID=$$ bash "$LOCK" release "$proj" >/dev/null

echo "T8 legacy directory-form lock is recognized, not deadlocked on"
proj="$TMP/t8"; mkdir -p "$proj/.state/lock"
echo 999999 > "$proj/.state/lock/pid"          # a pid that does not exist
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$proj/.state/lock/acquired_at"
LOCK_ACQUIRE_TRIES=20 LOCK_OWNER_PID=$$ bash "$LOCK" acquire "$proj" >/dev/null 2>&1; rc=$?
check "legacy stale lock reclaimed (rc)" "$rc" "0"
LOCK_OWNER_PID=$$ bash "$LOCK" release "$proj" >/dev/null

echo
echo "passed=$PASS failed=$FAIL"
[[ "$FAIL" -eq 0 ]]
