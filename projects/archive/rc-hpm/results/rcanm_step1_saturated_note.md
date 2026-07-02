# RC-ANM Step-1 first run (job 22501582) — INVALID aggressive arm
eps=0.5/1.0/1.5 returned identical (PGD 3x0.15 saturated disp~0.41 < eps_ball).
VALID finding (eps=0.5, v10 operating point, n=256): unsafe mined endpoints do
NOT poison the training gradient (grad_cos unsafe 0.080 vs safe 0.069, Welch
p=0.59) -> S1-impact FALSE. The "fix v10 by filtering" premise has no target.
P2 (aggressive) UNTESTED due to saturation -> re-run job 22507558 with
12x0.25 PGD to actually reach large eps.
