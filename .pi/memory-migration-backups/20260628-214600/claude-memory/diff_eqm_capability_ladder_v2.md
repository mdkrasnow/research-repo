---
name: diff_eqm_capability_ladder_v2
description: ANM/v10 capability ladder v2 (A-F) — FID gain IS real+behavioral but bounded to near-manifold quality
metadata: 
  node_type: memory
  type: project
  originSessionId: 74757461-ae3e-41e5-9da5-d3db38cca78e
---

Capability ladder v2 (2026-06-07/08) re-opened the ANM question after v1 zero-shot
inpainting NULL. Frozen ckpts, vanilla-s0 vs v10-s1 λ0.3. SUPERSEDES the pessimism of
[[diff_eqm_capability_ladder_null]] — v1 only falsified naive zero-shot restoration.

**Verdict: v10 FID gain (-3.83) IS a real behavioral change, BOUNDED to "near-manifold quality".**

POSITIVE:
- A (gain loc, from Exp3): quality+class-adherence+coverage, recall FLAT, 91% classes improve, cond-top1 +0.050.
- B (hard-class): HARD classes improve 1.34x feat-dist / ~1.9x class-adherence vs easy. Mining mechanism confirmed.
- D (sampler sweep, job 19948994, 96 cells): NO collapse (0 nan/div); converged regime nfe>=100 v10 -2.5..-3.2 FID; ~2.5x sample-efficient (v10@nfe100 <= vanilla@nfe250 both gd/ngd); overshoot-tolerant. BUT starved nfe<=25 v10 ~= or slightly worse — NOT a brittleness fix.

NULL:
- C (rescue): v10 conf +0.0018 on bad mid-states, no rescue (c(γ) decay → trajectory committed early).
- E (swap): all vanilla/v10 splices ~= pure arms; contribution not localizable early/late.
- F (edit): label-switch inert BOTH arms (base EqM-B/2 too weakly class-conditional, cond-top1 0.43) — uninformative, not true negative.

Mechanism: PGD hard-example mining sharpens the velocity field NEAR the data manifold in weak (hard-class)
regions. Buys final quality + sample-efficiency. Installs NO far-from-manifold behavior (repair/steer/restructure).

Paper framing: claim quality+hard-class+sample-efficiency; do NOT claim editing/repair/low-NFE-robustness.
needs_user_input set: accept framing or 3-seed-confirm A/B/D first.

**BENCHMARK ROLE (2026-06-08):** this A-F ladder is the SHARED capability benchmark for
**ANM vs SYMMETRY-DISCOVERY** (see [[symmetry_v17_morphismgym]] / [[diff_eqm_symmetry_ladder]]).
v10/ANM numbers above = the BASELINE BAR. Run the symmetry-discovery ckpt through the SAME
Rungs A-F (same frozen-ckpt protocol, metrics, seeds, sampler; swap ANM arm -> symmetry arm,
keep vanilla-s0 control; reuse eval_capabilities.py + eval_trajectory.py + exp1 sweep).
Compare: (i) does symmetry beat ANM on A/B/D (quality/hard-class/sample-eff)? (ii) does it
light up the ANM-NULL rungs C/E/F (rescue/splice-localization/counterfactual steering)? A
symmetry method that installs FAR-FROM-MANIFOLD structure could win exactly where ANM is null.

Infra built: experiments/eval_trajectory.py (rescue/swap/edit scheduled-GD splice harness),
eval_capabilities.py restore mode, exp1 sweep (Rung D). Doc: documentation/capability-ladder-v2-2026-06-07.md.
CRITICAL OPS GOTCHAS this run: (1) sbatch --export comma-split truncates comma-lists (NFE_LIST=5,10,..->just 5);
bake lists into script defaults instead. (2) home03 95G HARD cap -> exit-53 in 5s when full (results=87G);
exec>>$SYNC/job.log fails. Free regenerable caches/dup-ckpts(holylabs has them)/rsync-temps. (3) Run a pruner
(prune_all_active.sbatch, shared partition) when >=3 trains active — auto-trims *80ep* dirs keep anchors+latest2.
(4) v10 λ0.3 seed0 ckpt was PRUNED — used seed1 step-matched 0400000 for v2. See [[diff_eqm_v10_in1k_3seed]] [[diff_eqm_exp3_fidelity_diversity]].
