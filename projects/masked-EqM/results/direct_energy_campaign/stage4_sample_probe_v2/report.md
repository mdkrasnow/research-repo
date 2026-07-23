# Stage 4 corrected fixed-seed sample probes

All arms used the same 16 latent seeds, labels, 49 Euler updates, and
step size 0.003 from their matched 1,000-step pilot checkpoint.

| Arm | Finite | Latent norm (first → last) | Field norm (first → last) | Energy (first → last) | Sampling time |
|---|---:|---:|---:|---:|---:|
| none | yes | 63.16 → 40.98 | 245.48 → 187.02 | n/a | 3.09 s |
| dot | yes | 63.18 → 41.67 | 240.34 → 175.10 | 7609.76 → 186.43 | 5.77 s |
| direct | yes | 63.20 → 40.02 | 231.55 → 164.61 | 6163.58 → -760.85 | 5.73 s |

The direct energy decreased over the complete trajectory and neither latent
nor field norms diverged. Matched decoded images are finite and have no
gross collapse at this deliberately short pilot horizon. This is a sampler
viability result, not an ordinary-generation quality claim.
