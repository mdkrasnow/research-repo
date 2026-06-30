---
name: diff_eqm_v10_in1k_3seed
description: v10 IN-1K B/2 80ep 3-seed FID result — first multi-seed paper-scale win vs vanilla
metadata: 
  node_type: memory
  type: project
  originSessionId: 67b10278-e867-4729-a35f-6ebd2688269f
---

v10 PGD hard-example mining, IN-1K-256 class-cond EqM-B/2 80ep, FID 50K (gd eta=0.003/250steps/cfg=1.0, identical harness to trusted vanilla baseline 31.41):

- seed0 **27.2086** (job 19680996), seed1 **27.9230** (19681008, ep79 ckpt), seed2 **27.6020** (19681015)
- mean **27.58 ± 0.36** → **−3.83 FID** vs trusted vanilla seed0 31.41. All 3 seeds disjoint-below vanilla.
- Phase 1 gate (seed0 ≤ 30.41) **PASS**. Consistent w/ Exp3 single-seed ANM 26.88.

**Blocked**: Phase 2 Welch t-test needs vanilla 3-seed; only vanilla seed0 trusted. vanilla seed1 (19051147 ep59) + seed2 (19399165 ep57) trains FAILED final-sync (home quota) → need resume to ep80. PI update drafted 2026-06-07, needs_user_input flagged to authorize.

FID-eval gotcha: `imagenet1k_fid_eval.sbatch` default `GIT_SHA=HEAD` fails (`git checkout HEAD` invalid on fresh `git init` clone) — always pass an explicit commit SHA. Submit from `projects/diff-EqM` so logs land in convention dir.

Related: [[diff_eqm_exp3_fidelity_diversity]], [[diff_eqm_baseline_verified]], [[diff_eqm_holylabs_quota]]
