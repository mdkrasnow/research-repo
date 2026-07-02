
---

## Online adaptive sampler — 50k, 3 control-draws (2026-06-18, jobs 23151981-83)

Promotes the online equal-NFE claim from the 15k single-seed pilot to **50k samples**,
3 seeds. Each arm = 50,000 samples vs `in1k_reference_stats.npz`.

| arm | FID (seed0/1/2) |
|---|---|
| vanilla | 27.93 / 27.93 / 27.93 |
| random_restart | 27.89 / 27.96 / 28.01 |
| **probe_restart** | **26.90 / 26.90 / 26.90** |
| oracle_restart | 21.75 / 21.75 / 21.75 |

- **probe_restart 26.90 < random_restart {27.89, 27.96, 28.01} on ALL draws**, equal NFE.
  Δ = {0.99, 1.06, 1.11}, mean **1.05**. probe also < vanilla 27.93. Recovers 16–18% of oracle.
- Lineage stable: best-of-R/online Δ ≈ 1.0–1.9 FID across 15k→50k.

**Honest scope:** vanilla/probe/oracle arms are deterministic given the checkpoint —
`--seed` only reseeds the *random-restart subset selection*. So this is 3 independent
**random-control** draws, not 3 independent probe estimates. The controlled gap
(probe − random) is what varies and it is positive on every draw (CI excludes 0). The
claim is "probe-restart robustly beats the random-restart control's sampling variability
at 50k under equal NFE," not "3-seed probe FID mean." (For a 3-seed *probe* mean, the
50k best-of-R consistency run already gives Δ1.87±0.11 across genuinely independent seeds.)
