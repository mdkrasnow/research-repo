# Stage 7 — NFE robustness

Fixed pilot checkpoints and eta=0.003 were used for every run. All trajectories
were finite. The terminal diagnostics are:

| Steps | Arm | Final latent norm | Final field norm | Final energy |
|---:|---|---:|---:|---:|
| 25 | none | 48.83 | 225.44 | n/a |
| 25 | dot | collected separately; finite | — | — |
| 25 | direct | 48.25 | 227.54 | 2125.19 |
| 50 | direct | 40.02 | 164.61 | -760.85 |
| 100 | none | 50.21 | 86.62 | n/a |
| 100 | dot | 54.92 | 164.75 | -4303.91 |
| 100 | direct | 49.67 | 126.00 | -3865.45 |
| 250 | none | 82.86 | 88.65 | n/a |
| 250 | dot | 118.39 | 185.29 | -17703.15 |
| 250 | direct | 75.91 | 38.53 | -6628.94 |

Direct remained finite at every NFE, but did not provide evidence of superior
terminal convergence: its latent norm rises from the 100-step to the 250-step
probe. These are trajectory diagnostics only; the short pilot cannot support
FID or a generation-quality claim.
