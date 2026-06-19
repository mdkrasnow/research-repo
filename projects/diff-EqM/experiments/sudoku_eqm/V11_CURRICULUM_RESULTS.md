# v11 Metacognitive Curriculum on real Sudoku — RESULT (2026-06-19, job 23508890)

Tests the user's training-method idea: periodically run the model's own GD sampling on seen
data, flag the failures, and upweight them in continued training (hard-example mining) — with
the metacognition probe as the selector. From the SATNet v2 base (~0.80 cell-acc peak), 8
mining rounds × 5 weighted epochs, pool 3000, λ=1. Four arms, SAME base + SAME compute. Metric
= held-out CELL-ACCURACY (loss decouples from sampling — see SATNet v3). Board-acc stays 0 for
all (pure EqM-GD can't fully solve; 0.76^47 ≈ 0).

## Result
| arm | cell-acc r0 → r7 | final | Δ vs random |
|---|---|---|---|
| uniform (floor) | 0.711 → 0.695 | 0.695 | — |
| random (floor) | 0.718 → 0.680 | 0.680 | — |
| **oracle** (mine true-hard) | 0.714 → **0.760** | **0.760** | **+0.080** |
| **probe** (mine probe-flagged) | 0.709 → 0.703 | 0.703 | +0.023 |

- **The mechanism WORKS (oracle):** mining the model's own true failures is the **only arm that
  IMPROVES** over its start (+0.046 trajectory; +0.080 vs random, +0.065 vs uniform). Uniform and
  random both *degrade* (the over-training pathology). So **training on your failures breaks the
  EqM plateau** — and it does what inference-restart could NOT (the night's deterministic-failure
  gap): the right action for deterministic failures is training, not restart. **Confirmed.**
- **The probe as a label-free selector is WEAK here:** probe beats random (+0.023) and uniform
  (+0.008), so the descent shape carries *some* hard-example signal — but it is far below oracle
  (0.703 vs 0.760, −0.057). On Sudoku the trajectory dynamics don't encode "which board is harder"
  nearly as well as the true cell-error. The "metacognition enables LABEL-FREE hard mining" claim
  is NOT strongly supported on Sudoku.

## Interpretation
Two separable findings:
1. **Hard-example-mining-as-training-action is real** (oracle +0.08) — and it's the correct lever
   for the deterministic CSP failures that inference-restart leaves untouched. This is the strong
   half of the v11 idea, and it ties off the night's "detection≠actionability": the missing action
   is *training*, demonstrated here.
2. **The metacognition probe is a poor hard-selector on Sudoku.** Note the contrast with the night's
   4×4 *detection* AUROC 0.84–0.90: that predicted binary solved/unsolved; here the probe must rank
   *degrees* of cell-error among all-failing boards, which the descent shape does weakly. Selector ≠
   detector.

## Recommendation / next
- The probe's strong turf is **image generation** (descent-shape AUROC 0.82 there vs weak on Sudoku).
  The label-free payoff of v11 (mine-without-labels) should be tested where the probe is strong:
  v11-probe on EqM-B/2 image gen, vs oracle (Inception-NN-dist) and random. That is the natural home
  for the no-label claim.
- On Sudoku, oracle-mining (with labels) is the demonstrated win; probe-mining is weak — report both.
- 1 retune available if pursued: continuous cell-error target for the probe + out-of-sample fit
  (current is in-sample median split, optimistic yet still weak — a real ceiling, not a fit artifact).

## Honest scope
- Single SATNet base, in-sample probe fit, board-acc 0 throughout (cell-acc is the metric).
- The positive (oracle) shows the *training-as-action* mechanism; the probe arm shows the *selector*
  is task-dependent (strong on images, weak on Sudoku). Don't claim label-free mining from this run.
