# EqM maze planning — plan (grid-maze → path-grid)

Goal: a REAL trained-EqM planning task (not the CPU toy), so the trajectory-
metacognition mechanism is demonstrated on a genuine EqM, answering Yilun's maze
suggestion. Framing A (chosen): conditional EqM maps a maze layout to its solution
path, both as H×W grids.

## Task
- **Input (conditioning):** 3 channels — `wall` (1=wall), `start` (1-hot), `goal` (1-hot).
- **Output (data x1):** 1 channel — `path` (1 on the BFS shortest-path cells incl S,G), in {−1,+1}.
- **Model:** EqM field `f(x_γ, γ, cond)` — small conv net. Input = noisy path channel
  concat the 3 cond channels (4ch in), predicts the EqM target on the path channel (1ch out).
- **EqM target (faithful, replicated from `eqm-upstream/transport/transport.py`):**
  x0=noise, x1=path; linear plan xt=(1−t)x0+t·x1, ut=x1−x0; target = ut·c(t),
  `c(t)=min(1, 5(1−t))·4` (=4 for t≤0.8, →0 at t=1). Energy-compatible, c→0 at data.
- **Inference:** EqM GD from noise conditioned on the maze: `xt += f(xt,t,cond)·η`.
  Decode: threshold path channel → walkable set → BFS validity.
- **Difficulty / OOD:** maze cell-count tiers {5,7,10} → grids {11,15,21}. Train on
  small, test on larger (the IRED "harder→more steps" regime where metacognition pays).

## Metric (crisp, exact — better than image FID)
`valid` = predicted path cells lie only on free cells AND BFS over them connects S→G.
Binary oracle from BFS — the metacognition label is EXACT (no inception-knn noise).

## Controls (mandatory, per CLAUDE.md)
- **Positive (Step 2 gate):** does the trained EqM solve mazes above chance at all?
  If valid-rate ≈ random, the model/harness is broken — fix before metacognition.
- **Negative (floor):** random path grid → ~0 valid. Confirms metric isn't trivial.
- **Metacognition arms (Step 3):** vanilla (1 draw) / random-restart / probe-restart /
  oracle-restart — equal NFE, exactly as the image sampler.

## Phases (build incrementally — NOT a grid)
1. **Data + validity** (CPU): generate mazes + BFS labels + validity checker + viz.
   Gate: mazes solvable, BFS correct, encode/decode round-trips, random floor ≈ 0 valid.
2. **Train tiny EqM** (CPU smoke → GPU): faithful EqM target, small conv field.
   Gate (POSITIVE CONTROL): valid-path-rate ≫ random on held-out mazes. If not, the
   task isn't learnable in this setup → fix before any metacognition claim.
3. **Metacognition** (reuse separability_diagnostic probe + restart): log GD descent
   dynamics, BFS-validity label, trajectory-shape probe → invalid; probe-restart vs
   random vs oracle at equal NFE, by difficulty tier. Gate: probe-restart > random.

## Why this is a clean testbed
- Exact binary label (BFS) → probe AUROC is uncontaminated (vs image inception-knn).
- Real spurious minima: GD settles on a disconnected/wall-crossing path = "garbage".
- Built-in difficulty axis = the adaptive-compute regime metacognition is meant for.
- Reuses: EqM target geometry (faithful), the probe + restart machinery, maze gen +
  BFS validity. No EqM training-code surgery; small from-scratch field.

## Scope / honesty
- This is a NEW small EqM trained on mazes — not the IN-1K B/2 checkpoint. The claim is
  "trajectory-metacognition works on a real trained EqM solving a planning task", not
  "the image EqM plans". Difficulty tiers + exact labels make it a stronger mechanism
  test than the toy (toy AUC≈1.0); expect a harder, more informative AUROC here.
