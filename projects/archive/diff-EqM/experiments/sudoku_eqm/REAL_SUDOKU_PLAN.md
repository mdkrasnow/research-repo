# Plan — EqM solves REAL Sudoku (IRED-parity), then metacognition. 2026-06-19

**Goal (user):** show trajectory-metacognition WORKS on Sudoku using a REAL dataset the way
IRED does. First make an **EqM actually solve real Sudoku at scale on GPU** (a new task for
EqM), THEN layer metacognition. Build up carefully — gated stages, positive controls.

## Why the night's 9×9 failed (diagnosis, drives the plan)
Our toy 9×9 EqM had a conv-UNet that **downsampled 9→3** (destroys the grid) at width 96.
IRED's Sudoku model (`projects/archive/ired/models.py:SudokuEBM/SudokuDenoise`) runs
**full-res 9×9, h=384, ResBlocks + self-attention** (attention carries the global row/col/box
constraints). The architecture, not EqM, was the limiter. Fix = adopt that backbone.

## IRED parity (so "like they use it" is literal)
- **Data:** SATNet (`data/download-satnet.sh` → `powei.tw/sudoku.zip`; ~10k, fixed givens,
  EASY) and RRN-hard (`data/download-rrn.sh` → dropbox `sudoku-hard.zip`; 17–34 givens, HARD,
  IRED's scaling/OOD set). Both need internet → fetch on login node, rsync to data dir.
- **Encoding (sat_dataset.py):** board (9,9,9) one-hot → flat 729 → rescale `(x-0.5)*2` ∈{−1,+1};
  givens = features (empty cells all-zero→−1); `cond_entry` = given-cell mask; label = solution.
- **Eval:** **board accuracy** (entire grid correct) — IRED's metric. Cell-accuracy as a
  secondary diagnostic.

## EqM keeps its identity
We do NOT switch to IRED's energy-diffusion. We keep the **faithful EqM target**
(`c(t)=min(1,5(1-t))·4`, GD sampler t=0) — only the *backbone* upgrades to the
attention/full-res Sudoku architecture. The paper claim stays "EqM + trajectory-metacognition".

## Staged build-up (each stage gated; do not advance on fail)

### Stage 0 — Data + harness parity (no training; CPU/login)
- Download SATNet + RRN on login node; place in `sudoku_eqm/data_real/{sudoku,sudoku-rrn}`.
- Port IRED encoding + **board-accuracy** eval into our harness (replace synthetic gen path;
  keep synthetic 4×4 for fast CI). Verify shapes/round-trip on 50 boards.
- GATE: loader returns correct (B,9,9,9){−1,+1} tensors + given-mask; board-acc on GT labels = 1.0.

### Stage 1 — EqM-Sudoku architecture that SOLVES (the crux; GPU)
- New `SudokuEqM` backbone = IRED `SudokuDenoise` style: input `[xt(9) ⊕ givens(9)]` (18ch) at
  full 9×9, h=384, 3 ResBlocks + Attention, conv→9ch field output. Faithful EqM target + GD.
- Train on **SATNet (easy) first**. POSITIVE CONTROL gate: **board-acc ≥ 0.80** (IRED ~98% on
  SATNet; EqM should clear 0.8 if the backbone works). If fail → debug arch/sampler steps/eta
  (1 retune), not a new idea.
- Smoke first (tiny subset, 5 epochs, board-acc finite) before full train.

### Stage 2 — Scale to HARD RRN + GPU (the "scale up, works on GPU" deliverable)
- Train SudokuEqM on RRN-hard. GPU (full A100). Match IRED test-split board-acc.
- GATE: board-acc well above 0 and ideally a workable band (0.3–0.9) so metacognition has
  failures to act on. Report vs IRED's published RRN number for context (not parity-required).
- Adaptive GD steps at inference (IRED's idea): more steps → higher acc on harder boards.

### Stage 3 — Metacognition on REAL Sudoku (the actual goal)
- Descent-shape probe over the EqM GD trajectory (reuse `feature_groups`); detection AUROC
  (predict unsolved board), de-confounded.
- Two actions, equal-NFE: (a) best-of-R restart vs random; (b) **extra-GD-on-flagged** (spend
  the same extra steps on probe-flagged boards vs random boards). The night showed CSP failures
  are deterministic → restart may not help, but (b) (adaptive compute, IRED-native) should.
- GATE: detection AUROC > chance (expected ~0.8 from the 4×4 result) AND at least one action
  beats its equal-compute control.

### Stage 4 — Write-up + integrate
- `REAL_SUDOKU_RESULTS.md`; update SYNTHESIS link (#8 upgraded from 4×4-toy to real-dataset).
- Honest scope: EqM-on-real-Sudoku is a NEW capability result; metacognition = detection +
  (which) action works.

## Risks / mitigations
- **No node internet** → download on login node (Stage 0).
- **EqM may not solve hard RRN even with good arch** (IRED needed annealed energy + inner-loop;
  plain EqM-GD may plateau). Mitigation: SATNet-first gate isolates "EqM-can-solve-Sudoku" from
  "EqM-matches-IRED-on-hard". Even SATNet-solved + metacog is a real result.
- **Restart-action null on deterministic CSP** (night finding) → Stage 3 also tests adaptive-compute action.
- Keep faithful EqM target — don't drift into IRED's training tricks (would muddy the claim).

## Independent: image RePaint job (23442566) continues in background; unrelated to this.
