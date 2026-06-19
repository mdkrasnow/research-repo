# Sudoku-EqM — constraint reasoning as a third task type (2026-06-18)

A conditional EqM maps clues -> a full solution grid; invalid = constraint violation (exact
checker = oracle). Distinct from image generation (pixels) and maze planning (paths): pure CSP.

## Can an EqM do constraint reasoning?
- **9x9: NO (at this scale).** Loss fell cleanly 2.40 -> 0.99 over 80 epochs, but exact
  solve-rate = **0.000**. Solving requires ALL 81 cells jointly correct + constraint-consistent;
  a small conv GD-EqM never nails a full grid. Not a metacognition failure — the task is too hard
  for this architecture/budget (IRED-style CSPs need many refinement steps / constraint propagation).
- **4x4: YES.** Same model, 2x2 boxes. Train gate **PASS** ([GATE PASS] solve-rate=0.950  random-floor~0). The EqM solves 4x4 sudoku
  (~0.88-0.95 depending on clue count) >> random floor 0.000 — constraint reasoning is learnable.

## Metacognition on 4x4 (R=4, equal NFE, exact oracle, de-conf AUROC)
| test clues | invalid | probe AUROC (de-conf) | vanilla | random | probe | oracle | probe−random | oracle−random |
|---|---|---|---|---|---|---|---|---|
| 4 (hard) | 0.28 | **0.861** | 0.721 | 0.721 | 0.728 | 0.735 | +0.006 | +0.014 |
| 5 | 0.13 | **0.840** | 0.875 | 0.876 | 0.880 | 0.884 | +0.004 | +0.008 |
| 6 (easy) | 0.04 | **0.903** | 0.953 | 0.955 | 0.955 | 0.959 | +0.000 | +0.004 |

## Reading — DETECTION transfers, restart-ACTION does not (and why)
- **Detection works: de-confounded AUROC 0.84–0.90.** The descent-shape probe reads
  constraint-violation from the dynamics — the metacognition *signal* transfers cleanly to a
  third, non-spatial-planning task (CSP). A constraint-violating fill IS an instability in the
  relaxation, exactly as the mechanism predicts.
- **But best-of-R restart barely helps (gap ~0).** The tell is the ORACLE: oracle−random is also
  tiny (+0.004…+0.014). The R=4 restarts are highly CORRELATED — a puzzle the model fails, it
  fails on *every* restart. Sudoku failures are **puzzle-intrinsic/deterministic**, not stochastic,
  so there is no "lucky restart" for best-of-R to select. Even a perfect detector can't rescue
  what restarting can't change.

## Verdict
Sudoku separates two things the other tasks conflated: **detection** (the probe reads failure from
dynamics — transfers to CSP, AUROC ~0.86) vs **actionability-via-restart** (needs *stochastic*
failure diversity — present in image gen & maze, ABSENT in deterministic CSP). The metacognition
signal is real for constraint reasoning; best-of-R is just the wrong intervention there. A
different action (extra compute / constraint-guided refinement on flagged puzzles) is the open
direction. Honest caveat: 9x9 out of reach for this small EqM; result is on 4x4.
