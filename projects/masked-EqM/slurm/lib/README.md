# `slurm/lib/` — the shell half of the telemetry contract

`telemetry_env.sh` is the sourceable prelude every sbatch in this project should
adopt. It is **layer 5** of the five-layer terminal-record ladder described in
`telemetry/lifecycle.py`: layers 1–4 live inside python and cover every death the
process can observe (normal return, exception, signal, `atexit`); layer 5 covers
the ones it cannot — `SIGKILL`, OOM kill, node death, and a wall-clock kill that
lands before `RunRecorder.__enter__` installed its handlers.

By definition that layer must be implemented by a surviving observer, and the
batch shell is the nearest one.

---

## The contract in one screen

```bash
#!/bin/bash
#SBATCH --job-name=my-arm            # launcher overrides with --job-name="$RUN_TAG"
#SBATCH --output=.../my-%x-%A.out
#SBATCH --error=.../my-%x-%A.err
#SBATCH --open-mode=append           # (C) a requeue must not truncate the last attempt
#SBATCH --signal=B:USR1@120          # (A) 120s of warning before the wall clock
#SBATCH --time=24:00:00
set -euo pipefail                    # (B) pipefail is asserted by the prelude

module load python/3.10.13-fasrc01 2>/dev/null || true
# ... git checkout of $GIT_SHA, pip install ...
cd projects/masked-EqM

source "$PWD/slurm/lib/telemetry_env.sh"
eqm_forbid_positional "$@"                       # (F)
eqm_prelude "/n/netscratch/.../default_root"     # (D)(E)(G) + the (C) check

eqm_telemetry_begin \
  --campaign btm --phase II --arm "$BTM_MODE" \
  --seed "$GLOBAL_SEED" --git-sha "$GIT_SHA" \
  --planned-steps "${MAX_STEPS:-}" \
  --param ebm="$EBM" --param fd_k="$FD_K" --param fd_eps="$FD_EPS"

EXIT_CODE=0
eqm_run bash -c "$CMD 2>&1 | tee train.log; exit \${PIPESTATUS[0]}" || EXIT_CODE=$?

eqm_seal_now "$EXIT_CODE"     # only if the job deletes its own working tree
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
```

The worked example is `slurm/jobs/btm_image_arm.sbatch`. It is the migration
exemplar: its scientific behavior (model, flags, pytest gate, rsync, cleanup,
exit code) is byte-identical to the pre-telemetry version.

---

## What the prelude exports

| Variable | Meaning |
| --- | --- |
| `RESULTS_ROOT` | canonical results root; legacy spellings accepted with a warning |
| `IMAGENET_PATH` | pinned to the holylfs06 kempner copy |
| `MASTER_PORT` | deterministic function of `SLURM_JOB_ID` (+ array task id) |
| `EQM_TELEMETRY_ROOT` | `$RESULTS_ROOT/_telemetry` unless preset |
| `EQM_RUN_SPEC` | canonical spec JSON — the blob `RunSpec.from_env()` reads |
| `EQM_RUN_UID` | the content-addressed run id, for log lines and dir names |
| `EQM_PROJECT_DIR` | the tree containing `telemetry/` (derived from `BASH_SOURCE`) |

## Functions

| Function | Purpose |
| --- | --- |
| `eqm_prelude [default_root]` | the four context-free fixes: D, E, G, and the C check |
| `eqm_telemetry_begin …` | mints `EQM_RUN_SPEC` and arms the trap ladder |
| `eqm_run CMD…` | runs the scientific command so signals are actually deliverable |
| `eqm_seal_now [code]` | seals explicitly, ahead of the EXIT trap |
| `eqm_forbid_positional "$@"` | fails a job configured via argv |
| `eqm_normalize_results_root` / `eqm_resolve_imagenet` / `eqm_set_master_port` | individually callable |

---

## Three details that are load-bearing, not stylistic

### 1. `--signal=B:USR1@120`, with the `B:`

Without `B:`, SLURM signals the job *steps* and **explicitly not the batch
shell**. Every job in this corpus runs python directly in the batch shell rather
than under `srun`, so a bare `--signal=USR1@120` is present-and-inert: the
directive appears in the file, nothing is ever delivered, and the absence of a
terminal record looks like a telemetry bug rather than a missing prefix.

Jobs that *do* launch their work under `srun` should use the unprefixed form (or
both) so the step itself is signalled directly.

### 2. `eqm_run` backgrounds the command, and that is the point

Bash defers trap handlers for a **foreground** command until that command
returns. A 24-hour training job run in the foreground would therefore process
SLURM's 120-second warning 24 hours late — i.e. never. `eqm_run` backgrounds the
command and `wait`s, because `wait` is interruptible: it returns >128 when a
trapped signal arrives, the handler runs promptly, and the wait is resumed so the
child's real status is still what gets returned.

### 3. Signal forwarding walks the whole descendant tree

The process that must hear `SIGUSR1` is `python train.py`, which under
`eqm_run bash -c "… | tee …"` plus `torch.distributed.run` sits three or four
levels below the batch shell. `pkill -P $$` would hit `bash` or `tee`; a process
group kill depends on the batch script being a session leader (true on SLURM,
false in a local dry run, and silently a no-op when false). The prelude therefore
computes the transitive descendant closure from a single `ps` snapshot and
signals each pid, with the process-group kill kept only as a fallback.

The forwarding exists so the **in-process** handler gets first refusal: python
seals its own stream with a real `END` (`inferred: false`, with `wall_seconds`,
peak memory, and the true `last_step`). The shell sealer is the fallback that
runs when python could not, and its records are always `inferred: true`.

---

## Migration checklist for the remaining sbatch files

`--open-mode=append` has already been added to all 92 files. Everything below is
per-file work.

For each file, in this order:

- [ ] **1. Add `#SBATCH --signal`.** `B:USR1@120` for jobs that run python
      directly in the batch shell; plain `USR1@120` for jobs that use `srun`.
      Skip for jobs whose `--time` is far above their real runtime *and* which
      write no telemetry (pure `sacct`-only utilities such as
      `prune_active_ckpts.sbatch`).
- [ ] **2. Confirm `set -euo pipefail` is above the `source` line.** The prelude
      hard-fails without `pipefail`. `prune_active_ckpts.sbatch` currently uses
      `set -uo pipefail` (no `-e`) deliberately — it must keep polling across
      transient failures — so it needs `eqm_require_pipefail` but not `-e`.
- [ ] **3. Fix the log filenames** to include `%x` (job name) and require the
      launcher to pass `--job-name="$RUN_TAG"`. `#SBATCH` directives are expanded
      by SLURM before any shell variable exists, so `RUN_TAG` cannot appear
      literally. Files whose output path is *relative*
      (e.g. `projects/masked-EqM/slurm/logs/...`) additionally depend on the
      submitting cwd and should be made absolute.
- [ ] **4. Rename the results variable to `RESULTS_ROOT`** and call
      `eqm_prelude "<the file's existing default>"`. Distinguish carefully:
      `RESULTS_ROOT` is a **root** that per-job leaves live under; many files'
      `OUT_DIR` is a **leaf**. The prelude does not alias the legacy names back
      to `RESULTS_ROOT` precisely so that a leaf/root confusion cannot silently
      relocate outputs — decide per file which one you have.
- [ ] **5. Delete the file's own `IMAGENET_PATH` default** and let
      `eqm_resolve_imagenet` supply it. The 8 files still defaulting to
      `/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train` will now hard-fail unless
      `EQM_ALLOW_RETIRED_IMAGENET=1` is set, which is intended: a rerun that was
      silently reading a different dataset copy should stop and be looked at.
- [ ] **6. Replace `MASTER_PORT=$((29500 + RANDOM % 1000))`** (12 occurrences)
      with nothing — `eqm_prelude` sets it deterministically. Keep a caller-set
      `MASTER_PORT` working; the prelude already defers to one.
- [ ] **7. Convert positional args to env vars.**
      `direct_energy_longer_train.sbatch` (`EPOCHS="${1:-25}"`,
      `SAVE_EPOCHS="${2:-25}"`) is the remaining offender.
      `direct_energy_ddp_continue.sbatch` already prefers `TARGET_EPOCHS` /
      `TARGET_SAVE_EPOCHS` and only needs the positional fallback deleted. Then
      add `eqm_forbid_positional "$@"`.
- [ ] **8. Add `eqm_telemetry_begin`** with the arm's real campaign/phase/arm/
      seed/git-sha and every identity-bearing knob as a `--param`. Rule of thumb:
      if changing it changes what is computed, it is a `--param`; if it changes
      only where or how fast, it is not (see
      `telemetry.ids.NON_IDENTIFYING_KEYS`, which already excludes partition,
      worker count, results paths and log cadences).
- [ ] **9. Wrap the scientific command in `eqm_run`** and capture the status with
      `|| EXIT_CODE=$?`.
- [ ] **10. If the job `rm -rf`s its own checkout**, call `eqm_seal_now
      "$EXIT_CODE"` *before* the cleanup. The EXIT trap runs the sealer out of
      `$EQM_PROJECT_DIR`; deleting that first leaves the fallback with no
      interpreter. Sealing twice is safe and expected.

### Verification after each file

```bash
bash -n slurm/jobs/<file>.sbatch            # syntax
grep -c 'open-mode=append' slurm/jobs/<file>.sbatch
```

and, for a file you have fully migrated, a local dry run with `SLURM_JOB_ID` set
and the trainer replaced by a `sleep`, checking that a `SIGUSR1` to the batch
shell produces an `END` record under `$EQM_TELEMETRY_ROOT`.

---

## Correction to the original audit (defect B)

The audit named 14 files as piping python to `tee` without capturing
`${PIPESTATUS[0]}`, and concluded they exit with tee's status and report
`COMPLETED 0:0` to sacct when python crashes.

**All 14 set `set -euo pipefail`.** With `pipefail`, a pipeline's status is that
of the rightmost command to fail, so a crashed python already propagates, and
`set -e` already aborts the script. None of the 14 has the reported defect; the
audit appears to have grepped for the absence of `PIPESTATUS` without checking
for `pipefail`. Across all 92 files there is no file that both uses `tee` and
lacks `pipefail`, so the class is empty, not merely small.

The residual real problem in those files is different and is what the prelude
addresses: because `set -e` aborts at the failing pipeline, the exit status is
never *captured into a variable*, so there is nothing to hand a sealer and no
cleanup runs. `eqm_run … || EXIT_CODE=$?` plus the EXIT trap fixes that without
restructuring the pipeline.

`eqm_require_pipefail` exists to keep the property from silently regressing: it
is currently true of every file, but nothing enforced it.
