# Maze-EqM Step 3 — trajectory-metacognition on a REAL trained EqM ✅

The metacognition mechanism, previously shown on the image EqM and a hand-built toy
planner, now demonstrated on a **real trained EqM solving a planning task**, with
**exact BFS validity labels** (no inception-knn noise) and equal-NFE controls.

Setup: the c5-trained EqM (Step 2) sampled on OOD mazes. For each maze, R=4 independent
EqM-GD draws, logging per-step descent (norm/dot); each draw labelled valid/invalid by
BFS. (1) a trajectory-shape probe (the SAME image-probe feature builder) predicts
invalid from descent dynamics; (2) best-of-R restart picks by probe vs random vs oracle
at identical NFE (R×steps).

| tier | grid | steps | invalid | probe AUROC (de-conf) | vanilla | random | **probe** | oracle | Δ(probe−rand) | % oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| c7  | 15 | 20 | 0.775 | **0.762** | 0.214 | 0.220 | **0.436** | 0.586 | **+0.216** | 59% |
| c10 (seed0) | 21 | 25 | 0.396 | 0.666 | 0.618 | 0.576 | **0.708** | 0.830 | **+0.132** | 52% |
| c10 (seed1) | 21 | 25 | 0.405 | 0.684 | 0.594 | 0.616 | **0.718** | 0.836 | **+0.102** | 46% |

**Verdict: PROBE > RANDOM at equal NFE on all 3 runs (2 seeds × 2 OOD tiers).** The
descent-trajectory shape predicts maze-solve failure (de-confounded AUROC 0.67–0.76),
and probe-guided restart rescues 46–59% of the oracle gain over random restart.

## Difficulty-scaling story (the headline)
Harder regime → stronger signal AND bigger rescue:
- c7 at 77% invalid: AUROC **0.762**, Δ **+0.216** (59% of oracle).
- c10 at ~40% invalid: AUROC ~0.67, Δ ~+0.12 (~50% of oracle).
More failures = more dynamics signal = more for the probe to fix — exactly the
adaptive-compute regime metacognition is meant for (the IRED "harder→more steps" axis).

## Why this is the result Yilun asked for
- A **genuine trained EqM** (not the toy, not the image checkpoint) **plans** (solves
  grid mazes by gradient descent on its learned field) and **generalizes** to 4× larger.
- Trajectory-metacognition transfers to it under strict controls: exact labels (BFS),
  positive (oracle) + negative (random) brackets, identical NFE.
- Recovers MORE of the oracle (46–59%) than the image best-of-R (14% at 50k) — exact
  labels + a more dynamics-detectable failure mode.

## Honest caveats
- New small EqM trained on mazes (653K params), not the IN-1K B/2 checkpoint. Claim =
  "metacognition works on a real trained EqM planner", not "the image EqM plans".
- Probe AUROC 0.67–0.76 < image 0.82 (short 25-step trajectories, OOD, exact labels =
  harder). Single checkpoint; 2 seeds × 2 tiers. Operating points chosen for failure
  headroom (low step budget), reported transparently.
- `maze_metacog.py` (reuses the separability_diagnostic probe builder + numpy logistic).
