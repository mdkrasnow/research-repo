# Update for Yilun — EqM trajectory-metacognition

**TL;DR (updated 2026-06-14 — both ran).** Your two questions: (1) are gains
consistent — **YES**: 50k × 3 seeds, mean Δ1.87±0.11 FID, 95% CI ±0.12 excludes 0,
probe<vanilla every seed. (2) capabilities — the EqM-native **online metacognition
sampler works at 15k** (probe-restart 28.51 < random-restart 29.76 at equal NFE,
vanilla sanity OK), and the **maze-planning** analog is positive at equal compute.
Energy-as-quality is dead; descent-*dynamics* is the live signal. Headline result is
paper-ready. Open: image-domain capability rungs (inpainting/translation) — your
priority call below.

### The numbers
- **Consistency (50k, 3 seeds):** vanilla→probe = 28.20→26.21, 27.78→25.95,
  27.83→26.04. Mean Δ1.87±0.11 FID, CI ±0.12. CONSISTENT.
- **Online sampler (15k, equal-NFE):** vanilla 29.55 / random-restart 29.76 /
  probe-restart 28.51 / oracle 23.32. Δ1.24 over random at identical compute. WORKS.
- **Maze planning — now on a REAL trained EqM (not just the toy):** a small
  conditional EqM trained to solve grid mazes (layout→path) hits 0.99 valid in-dist,
  0.88–0.93 OOD (4× larger). Trajectory-metacognition on it (exact BFS labels, equal
  NFE): probe-restart > random on 2 seeds × 2 OOD tiers, Δ+0.10 to +0.22 valid-rate,
  46–59% of oracle; harder tier → higher probe AUROC (0.76) + bigger rescue. This is
  the EqM-native planning result you suggested. Detail: `experiments/maze_eqm/`.

### Priority question — UPDATE
Maze planning is now done on a real EqM. The remaining open capability rung is image-
domain **inpainting / repair / translation** (designed, unrun). Worth building next,
or is the planning + image-FID story enough for the first paper?

---
*(Original pre-run draft below, kept for the proven/not-proven framing.)*

**TL;DR (original).** Your two questions were (1) are the gains consistent, and (2) does this
unlock capabilities like inpainting / translation / maze planning. Status: the
*detection* result is solid and the *maze-planning* capability now has a positive,
equal-compute result. The decisive at-scale consistency number (50k, multiple
seeds) and the EqM-native online sampler are coded, smoke-tested, and one GPU block
from running — I'm holding on your priority call before spending it.

## What is proven

- **Vanilla baseline reproduced.** Our sampler pipeline gives FID 29.53 at 15k
  (≈ the trusted 31.41 B/2 baseline) — the FID path is trustworthy, so deltas are real.
- **Endpoint energy fails as a quality signal.** The EqM energy scalar (−⟨f,x⟩) and
  the path-integral reach only ~0.61 de-confounded AUROC for good-vs-garbage — a
  dumb latent-NN baseline (0.627) beats them. EqM energy is a *density* skeleton,
  not a *quality* axis. (This kills the original "energy marks the bad minimum" idea.)
- **Descent *dynamics* predict failure.** A small learned probe over the descent-
  trajectory *shape* (oscillation, log-decay slope, normalized norm/dot curves) hits
  **0.82 held-out, de-confounded** from gradient norm — stable across 5 seeds, and
  just as strong read at **step 100/249** as at the end (so it can act early).
- **Acting on it improves controlled FID.** Probe-guided best-of-R=3 restart at 15k:
  vanilla 29.53 → **probe 27.84** → oracle 17.75. Probe beats the random-keep floor,
  inside the neg/pos control band, recovering 14% of the oracle gain.
- **The mechanism transfers to maze planning (your suggestion).** In an energy-descent
  path planner with real spurious minima, a dynamics-probe that branches the flagged
  candidates reaches **0.928 valid-path-rate vs 0.794 for random branching at equal
  compute** (hardest mazes +0.228), recovering 89% of the oracle gain. The gap grows
  with difficulty.

## What is NOT yet proven (and the exact next test)

- **Consistency at the real metric.** Today's FID gains are 15k, single sampler seed.
  `consistency.sbatch` runs vanilla/probe/oracle at **50k across 3 seeds** → a
  mean±CI gain. *This is the number that turns "promising" into "consistent."* Coded
  and smoke-ready; needs a 4×A100 block.
- **Online equal-NFE sampler at scale.** The 15k result is post-hoc best-of-R. The
  *true* metacognition sampler (restart only the high-risk slots mid-flight, equal
  NFE vs random) is built and **mock-validated** (probe 0.212 vs random 0.385 at
  identical NFE), but not yet run on EqM. `online_adaptive.sbatch`.
- **Capabilities beyond maze.** Inpainting/repair and translation rungs are designed
  and scoped but not run.

## One decision I need from you

The next GPU block can go to one of:
1. **50k × 3-seed consistency** — locks the headline FID claim (lowest-risk, highest
   paper value).
2. **Maze planning, deeper** — scale from the toy to a harder planner / real maze
   benchmark, since the toy already shows transfer and you flagged planning.
3. **EqM inpainting/repair rung** — the first "unlocks a capability" result on images.

My recommendation: **(1) first** (it's the claim everything else rests on), then **(2)
maze depth** over image repair, since you suggested planning and the toy already
gives a clean positive to build on. **Should maze planning be prioritized over image
repair/translation for the capability story?**

*(Backing detail: `RUN_INDEX.md`, `FINDINGS.md`, `CAPABILITY_LADDER_RESULTS.md`,
`results/online_adaptive/ONLINE_ADAPTIVE_SUMMARY.md`, `PAPER_CLAIM_STATUS.md`.)*
