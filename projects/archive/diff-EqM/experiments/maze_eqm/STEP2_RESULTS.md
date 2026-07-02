# Maze-EqM Step 2 — trained EqM solves mazes (POSITIVE CONTROL PASS)

Small conditional EqM (653K params, faithful EqM target c(t)=min(1,5(1-t))·4),
trained on 4000 c5 mazes (11×11), 5 epochs, ~90s CPU. EqM GD sampler (t=0, η=0.02).

| tier | grid | path-len | valid@100 | valid@200 | valid@400 | random floor |
|---|---|---|---|---|---|---|
| c5 (in-dist) | 11 | 28 | 0.987 | 0.983 | 0.987 | 0.000 |
| c7 (OOD) | 15 | 50 | 0.903 | 0.880 | 0.890 | 0.000 |
| c10 (OOD) | 21 | 91 | 0.910 | 0.930 | 0.917 | 0.000 |

**GATE PASS:** valid-rate 0.99 in-dist, 0.88–0.93 OOD ≫ random floor 0.000. A real
trained EqM plans (solves grid mazes via gradient descent on its learned field) and
generalizes to 4× larger mazes than training. Failure headroom ~9–12% OOD = the
spurious-minimum cases for Step-3 metacognition. Extra GD steps don't recover them
(flat across 100→400) → structural failures, the right target for a restart probe.
