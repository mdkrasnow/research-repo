---
name: diff-eqm-phase-0-v10-pass
description: "v10 PGD hard-example mining on EqM CIFAR-10 PASSED Phase 0.3 sanity gate; FID 13.40 vs vanilla 14.17. Mining non-saturating (ratio 1.047-1.049 stable across 150 epochs), the central differentiation from v02 which saturated on EqM-B/2."
metadata: 
  node_type: memory
  type: project
  originSessionId: f3229205-c8f9-42de-862b-1e264830d515
---

## Phase 0.3 result (2026-05-20)

Job 13868114 (variant-pilot, seas_gpu, 6h23m). v10_hard_example variant, 150ep CIFAR-10 variant harness, seed 0.

**Headline**: v10 final FID@5K = **13.40** vs vanilla v00 R4 baseline 14.17 → beats by **0.77 FID** on identical harness.

**Mining diagnostics** (all 150 epochs):
- L_base: 2.31 → 2.13 (descending)
- L_hard: 2.41 → 2.23 (descending in parallel)
- ratio L_hard/L_base: 1.047 → 1.049 stable (slowly increasing)
- ||δ|| mean: 0.300 (at L2 boundary every epoch)

FID trajectory @1K samples: 324.8 (e30) → 98.6 (e60) → 48.7 (e90) → 39.7 (e120) → 37.0 (e150).

**Why**: confirms mining non-saturating, unlike v02 cosine which saturated to ``cos=0.999`` within 9 epochs on EqM-B/2 IN-1K (job 10198798 logs). v10's L2-regression objective has unbounded gradient even at perfect alignment, so PGA always finds non-trivial hard examples.

**How to apply**:
- Cite this as Phase 0.3 PASS in any subsequent diff-EqM session/paper draft.
- Unblocks Phase 1a CAFM-EqM 10-ep post-training (gated on smoke 13997995).
- Detailed result at `projects/diff-EqM/results/phase_0_3_v10_cifar_sanity_result.md`.
- Per CLAUDE.md CIFAR rule: stability check only, not IN-1K publishable. Phase 1b/2 still mandatory.

Related: [[diff-eqm-framing-branch-b-both]] for paper positioning, [[diff-eqm-variant-findings]] for prior variant history.
