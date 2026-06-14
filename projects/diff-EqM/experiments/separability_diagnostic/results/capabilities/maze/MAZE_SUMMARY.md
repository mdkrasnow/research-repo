# Maze planning prototype — results

Probe held-out AUC (predict invalid from partial-refinement dynamics): **1.000**

Valid-path rate by arm (all arms below the line are EXACTLY compute-matched; oracle is the over-budget ceiling):

| difficulty | n | vanilla | random | probe | oracle |
|---|---|---|---|---|---|
| 0 | 90 | 0.989 | 0.978 | 1.000 | 1.000 |
| 1 | 94 | 0.968 | 0.957 | 1.000 | 1.000 |
| 2 | 97 | 0.794 | 0.691 | 0.938 | 0.959 |
| 3 | 79 | 0.557 | 0.519 | 0.747 | 0.797 |
| **ALL** | 360 | 0.836 | 0.794 | **0.928** | 0.944 |

- NFE matched (vanilla/random/probe) = **280** path-grad steps; oracle = 640 (cheats, labeled).
- probe − random = **+0.133** ; oracle − random = +0.150 ; fraction of oracle gain recovered = 89%.

## VERDICT: PROBE>RANDOM (equal compute) — general mechanism transfers to maze planning

Controls: vanilla = single-init floor (neg), random = compute-matched random-branch (neg), oracle = any-of-R valid (pos ceiling). Probe read ONLY refinement dynamics — never ground-truth validity — at selection time.
