---
name: cluster-partition-policy
description: PI-mandated SLURM partition policy — gpu_requeue only, seas_gpu and kempner_h100 banned, and what a preemptible MIG-mixed partition requires of every job
status: active
---

# Cluster partition policy (PI-mandated, 2026-08-17)

## The rule

Two Slack messages from Yilun Du on 2026-08-17:

1. *"can you run jobs on kempner_requeue? Don't run jobs on kempner_h100"*
2. *"you are running jobs on seas_gpu now / Please do not do that / you can run jobs on
   gpu_requeue"*

| partition | status |
|---|---|
| `gpu_requeue` | **USE THIS.** The designated partition. |
| `gpu_test` | Allowed — never named in the ban. Short single-GPU work only. |
| `seas_gpu` | **BANNED** (message 2). Was our default for ≥12h / multi-GPU DDP. |
| `kempner_h100` | **BANNED** (message 1). |
| `kempner_requeue` | Named, but **NOT USABLE** — see below. |
| `gpu` | Chronic backlog; outside the policy. |

**`kempner_requeue` cannot currently be used.** `sacctmgr -n show assoc user=$USER` returns
only `ydu_lab`, so submissions fail with *"Invalid account or account/partition combination
specified"*. Someone must add the account association first. Worth raising with Yilun —
13 nodes were sitting idle there when we tried.

**Do not re-derive the partition from wait times.** The prior standing rule was "pick by
ACTUAL wait time, not habit," and following it is precisely what put us on `seas_gpu` and
produced the complaint. Allocation policy now outranks queue optimization.

## What `gpu_requeue` demands of a job

This is not a partition rename. Two properties change how jobs must be written.

### 1. MIG roulette — pin the full card

The pool mixes full cards (`nvidia_a100-sxm4-80gb:4`, `nvidia_h100_80gb_hbm3:4`) with MIG
slices (`nvidia_a100_3g.20gb:8`). MIG **cannot do multi-rank NCCL** — it fails ~3 minutes
in with `Duplicate GPU detected: rank 1 and rank 0 both on CUDA device`.

For any multi-GPU DDP job, pin the type rather than switching partitions:

    --gres=gpu:nvidia_a100-sxm4-80gb:4

A bare `--gres=gpu:4` is a coin flip. Single-GPU jobs are unaffected.

### 2. Preemption is the norm — design for it

A job **will** be killed and requeued. Every submission needs all four of:

- `#SBATCH --requeue` — without it SLURM refuses `scontrol requeue` and any
  requeue-on-fault path silently degrades to a plain exit.
- `#SBATCH --open-mode=append` — otherwise a requeue truncates the previous attempt's log
  and destroys the evidence of why it died.
- Dense `CKPT_EVERY`.
- **Auto-resume from the run's own newest checkpoint.**

Auto-resume is the non-obvious one and it is mandatory, not a nicety. A requeued job that
re-reads its seed `CKPT` env var restarts from the phase's *starting* weights on every
preemption — under frequent preemption it makes **no net progress**, while every attempt's
log looks perfectly healthy. SLURM reuses the job id across a requeue, so the results dir
is stable and the previous attempt's checkpoints are findable; a fresh submission gets a
new id and correctly falls back to the seed.

## Two checkpoint traps, both learned the expensive way

**`CKPT_EVERY` counts from step 0, not from the resume point.** A run resuming at step
75,000 with `CKPT_EVERY=25000` has its first checkpoint 25,000 steps (~1.7h) away. On
2026-08-17 four II-C arms died at step ~89,450 to an NCCL collective timeout and **not one
had written a checkpoint** — ~2h x 16 GPUs lost. Set the interval against the *exposure
window* you can afford, remembering the offset.

**Pick the resume checkpoint by mtime, not by name.** Step checkpoints (`0100000.pt`) and
epoch checkpoints (`epoch20.pt`) share one directory and do not sort into a consistent
order — lexically `'0100000.pt' < 'epoch20.pt'`, so a name sort silently resumes from the
OLDER state.

## Related

- [[btm-fd-scalar-campaign]] — the campaign whose II-C runs this policy now governs.
- Storage: the same day, Yilun also required `/n/holylabs/LABS/ydu_lab` be cut down
  (176G -> 19G done). Both are shared-resource complaints; treat lab-resource asks as
  higher priority than experiment throughput.
