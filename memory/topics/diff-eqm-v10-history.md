---
name: diff-eqm-v10-history
description: v10 hard-example mining variant on diff-EqM — passed Phase 1/2 IN-1K gates before project was archived
status: dormant
---

Summary of the diff-EqM project's strongest result before it was archived 2026-07-02
in favor of masked-EqM. Kept for reference only — do not resume without explicit
user redirect (see `AGENTS.md` hard scope rule).

- Baseline: IN-1K-256 EqM-B/2 80ep vanilla, FID 31.41 (trusted).
- v10 (hard-example mining, auxiliary loss = EqM base loss on a perturbed/mined input):
  CIFAR 150ep sanity FID 13.40 vs vanilla 14.17 PASS; IN-1K 3-seed FID 27.58±0.36
  vs 31.41 (Δ−3.83) — Phase 1 gate PASS across all 3 seeds.
- Field-robustness (Exp 2) and fidelity/diversity (Exp 3, FID 26.88 vs 31.27, no
  diversity tax) both confirmed at IN-1K B/2 scale.
- Capability ladder v2 (A–F): gain is real and behavioral (quality + hard-class +
  sample-efficiency positive; image-repair transfer null) — mining sharpens the
  field near the data manifold only, not a general-purpose capability boost.
- CAFM-on-EqM port (Branch B-Both) FAILED Phase 1 (FID 341 vs 31.41) — mechanism
  bug, not retunable; postmortem written, branch pivoted to v10-only.

Full detail lives in Claude Code's private per-session memory
(`~/.claude/projects/-Users-mkrasnow-Desktop-research-repo/memory/`) under entries
like `diff_eqm_v10_in1k_3seed.md`, `diff_eqm_exp2_field_robustness.md`,
`diff_eqm_exp3_fidelity_diversity.md`, `diff_eqm_capability_ladder_v2.md`. This file
is a condensed, git-tracked copy so the finding survives even if that private memory
is ever cleared.
