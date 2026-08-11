# Postmortem: Stage A top-k head-subspace hypothesis FALSIFIED (2026-08-11)

Job 38338496 (`topk_subspace_diagnostic.py`, K=64/48/32/59 per tail level, k=8 block
subspace, energy_head-only). Full results: `results/topk_subspace_diagnostic_38338496.json`.

## What was tested

Does direct scalar EqM's catastrophic gradient events come from the residual falling into a
small, high-gain SUBSPACE of the energy-head Jacobian — direct-specific, and worsening with
tail severity? Prerequisite gate before any head-only Gauss-Newton work.

## Gate result: FAIL

| criterion | result |
|---|---|
| residual_ratio ~= 1 | PASS (0.99/0.99/0.99/1.00 across tail levels) |
| A_head substantially higher on spikes | PASS (2.06x-2.36x) but A_backbone MORE so (2.50x-4.45x) |
| Q_8(spike) > Q_8(control) | weak PASS -- real but tiny (0.91->0.96-0.97), Q_8 already near-ceiling for BOTH |
| stronger in direct than none | **FAIL** -- I_Q interaction flips sign across tail levels (-0.15,-0.17,+0.17,-0.10); none's raw Q8 spike/control ratio (1.22,1.26,0.90,1.16) is not smaller than direct's (1.05,1.07,1.07,1.05) |
| strengthens with tail severity | **FAIL** -- Q_8 ratio flat/non-monotonic (1.052->1.066->1.069->1.049) |
| head remains major contributor | **FAIL** -- C_head SHRINKS on spikes vs control at every level (e.g. 17.3%->9.6% at top2.5%, 17.8%->4.3% at top0.5%), reversing the earlier ~85-88% claim from a much smaller K |

## The redirect finding

`A_backbone`'s spike/control ratio scales cleanly and monotonically with tail severity:
**2.50x (top2.5%) -> 3.56x (top1%) -> 4.45x (top0.5%)**. This is the single cleanest
severity-scaling trend in the entire session's diagnostics -- and it points at the BACKBONE,
not the energy head. The head-Jacobian-centric framing that motivated this whole
Stage A/B design is not where the mechanism lives, per this data.

## Why the earlier single-vector alignment result (job 38150502) was misleading

sigma_1/sigma_2 on spike batches (1.68-1.74) is *smaller* than on control (1.85-1.93) --
the top singular direction is LESS isolated on spikes, confirming the near-degeneracy
concern that motivated building the block subspace estimator in the first place. The
earlier alignment_ratio=6.62 finding was very likely basis rotation inside a near-degenerate
cluster, not a real single-direction effect -- exactly the failure mode the spec warned
about, and exactly why Stage A was required before touching Gauss-Newton.

## Decision (per AGENTS.md stop conditions + explicit spec instruction)

Per "if Q_k spike ~= Q_k control, or top-k modes explain little of A_head... STOP. Do NOT
implement the Gauss-Newton intervention. Instead report that the top-k alignment hypothesis
was falsified or insufficient" -- Stage B is NOT built. No head-only Gauss-Newton work
follows from this thread.

## Recommended next direction (not yet started, awaiting user call)

The backbone-severity-scaling finding (A_backbone ratio climbing 2.5x->4.45x with tail
severity) is a real, clean, monotonic signal that was never the target of this
investigation. If this thread continues, the theoretically-justified next step is a
BACKBONE-focused version of the same top-k subspace diagnostic (attention QKV/proj blocks
instead of energy_head), not a generic backbone regularizer chosen because it's common.
This has NOT been scoped, proposed in detail, or approved.

## Engineering notes (for future diagnostics on this codebase)

- `build_probe_bank`'s `ImageFolder()` call and `build_pool`'s VAE-encode loop had zero
  progress logging -- caused 3 of 5 total submission attempts to be diagnosed (correctly,
  in hindsight) as bad-node stalls purely from silence, when in fact node/filesystem
  variance was real but unverifiable in real time. Fixed (commit 5a6af4f): progress logging
  added to both.
- Neither `python` invocation in `topk_subspace_diagnostic.sbatch` used `-u` (unbuffered
  stdout) -- when piped through `tee`, Python defaults to full block buffering, so the new
  progress logging above did NOT stream in real time on the run that ultimately succeeded
  (38338496) -- output appeared in large delayed bursts. Fixed for future resubmissions
  (commit 23aa714, `python -u`). Mid-run verification instead used `sstat`'s `AveCPU` and
  (more reliably) `srun --jobid=<id> --overlap nvidia-smi` to directly confirm the process
  was actively computing on the GPU, independent of log buffering.
- `energy_head.linear.bias` has an exactly-zero field-Jacobian column (summed BEFORE the
  token dimension -> additive constant in z) -- `field_jvp_direct`/`_none` needed
  `allow_unused=True` + zero-substitution to handle any head-only (or generally, any
  parameter-subset) params list. Fixed commit c8f6491, regression test added.
- 5 total submission attempts for this one job: wrong env-var mechanism, checkpoint path
  documented wrong (real root is `/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/`,
  not `/n/holylabs/.../masked-EqM/results/` as both audit docs incorrectly stated -- corrected),
  zero-Jacobian bug, and 3 of 5 attempts hit prolonged stalls at the identical pool-build
  step on different nodes (node/filesystem variance, not reproducible on-demand -- the
  successful run also took an unusually long ~40min at that step before speeding up).
