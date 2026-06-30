---
name: diff_eqm_exp3_fidelity_diversity
description: "Exp 3 result — ANM's IN-1K FID gain has NO diversity tax (strong_success); plus reusable eval-infra gotchas"
metadata: 
  node_type: memory
  type: project
  originSessionId: 65d8faa9-7841-4beb-b938-d01e60090a68
---

Exp 3 (fidelity-diversity & mode coverage) DONE 2026-06-05. Verdict **STRONG_SUCCESS — no diversity tax**. vanilla vs ANM v10 λ=0.3, EqM-B/2 80ep IN-1K-256, 49996 identical ids, parity-controlled (gd/250/cfg1.0/EMA, shared schedule hash 83a8ede763e1b318, fixed seeded ref, pytorch_fid inception, vendored PRDC). Job 19120911 exit 0.

Numbers: FID 26.88 vs 31.27 (−4.38, disjoint 95% CIs 27.27–27.85 vs 31.64–32.28). **recall FLAT 0.7185→0.7193 (diversity axis preserved)**, coverage +0.072 (0.443→0.515), density +0.044, precision +0.023, KID −0.0057. Weak-class bottom-quartile FID −5.61 (62.80→57.19, weak gain MORE than mean). classifier TV→requested 0.181→0.162, conditional top-1 +0.050, 91% classes improve.

Meaning: ANM's FID win is quality, not mode-dropping — closes the "you just sharpened samples / lost variety" reviewer attack. Pairs with Exp 1 (sampler robustness) + Exp 2 (off-traj field robustness). Caveat: SINGLE SEED; Phase 2 3-seed Welch t (p<0.05, ≥1 FID gain) still required for paper-final claim.

Full writeup `documentation/exp3-fidelity-diversity-results.md`; data `results/exp3_metrics_out/`; gen PNGs on holylabs `mkrasnow_eqm/exp3/`. Related: [[diff_eqm_exp2_field_robustness]] [[diff_eqm_baseline_verified]] [[diff_eqm_phase_0_v10_pass]].

Reusable eval-infra gotchas hit (full chain in `documentation/debugging.md` 2026-06-05):
- home03 is 95G HARD cap → exit-53 quota deadlock (SLURM stdout pipe blocks). Move big outputs to holylabs.
- holylabs has a per-lab (ydu_lab) GROUP block quota INVISIBLE to `df`/`df -i` — EDQUOT fires with 1.5P "free" + 1% inodes. "df has space" ≠ writable. Split read-fs (holylabs, reads ignore quota) from write-fs (home) when read side is huge.
- gen job FAILED but full PNG count = log-pipe ENOSPC on home, NOT data loss. Verify count+provenance before regenerating.
- rsync moves can truncate a few files (4 zero-byte vanilla PNGs here). Feature extractors that skip-on-error MUST return which items succeeded; never pair extractor output with the pre-extraction file list by position (caused 49996-vs-50000 IndexError). Make eval score on the cross-arm COMMON readable id set so a few corrupt PNGs don't torpedo a 50K run. Fixed commit abe9dbf.
