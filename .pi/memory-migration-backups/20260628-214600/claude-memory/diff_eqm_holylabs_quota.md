---
name: diff_eqm_holylabs_quota
description: "IN-1K trains MUST write checkpoints to holylabs scratch, not home03 (95G) — home-quota deadlock silently wedges RUNNING jobs"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4f4c5428-98a7-4292-b0a5-e56722645048
---

`/n/home03` is a **95G hard cap**. diff-EqM IN-1K trains write 2GB checkpoints every 5000 steps;
defaulting them to home fills it and **deadlocks the train** (ENOSPC → tee/stdout pipe blocks →
process halts but stays `RUNNING` in squeue, burning 4×A100). Freeing space does NOT revive an
already-deadlocked job — it must be killed and resumed.

**Why:** 2026-06-04 the v10 3-seed gate (l03-s1/s2) + vanilla-s2 + lambda10-scale were all wedged
~12h this way — submitted without the storage override, ckpts went to home, home hit 100% ~10am.

**How to apply:**
1. Any `imagenet1k_80ep_*.sbatch` submit MUST set
   `PERSISTENT_RESULTS=/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/<dir>` (holylabs = 1.5PB). The sbatch
   supports it (lines ~66-69) but defaults to the home repo if unset.
2. Resume after a wedge: `--export=...,RESUME_CKPT=<dir>/000-*/checkpoints/<latest>.pt,PERSISTENT_RESULTS=holylabs...`
   plus the original `SEED`, `GAMMA` (=lambda; lambda03→0.3), `MINING` (v10 or none for vanilla),
   `RESULTS_DIR`.
3. Liveness check: a job is alive only if its **stdout mtime / latest-ckpt mtime is recent** —
   "RUNNING" in squeue is not proof. `scontrol show job <id> | grep StdOut` then `stat`.
4. Standing pruner for home: `prune_aggressive.sbatch` (keep {65000 anchor}+latest2 on *80ep*,
   prune bridge dirs, 180s). CIFAR bridge variant ckpts are small and can stay on home.

See [[imagenet_blocker]]. Incident logged in `projects/diff-EqM/documentation/debugging.md` (2026-06-04).
