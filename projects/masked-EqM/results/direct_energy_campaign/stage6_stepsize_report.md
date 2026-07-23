# Stage 6 — Sampler step-size robustness

All runs use the corrected 1,000-step pilot checkpoints, 16 fixed seeds,
49 Euler updates, and identical labels.

| η | Direct finite | Direct final latent norm | Direct final field norm | Direct final energy | Interpretation |
|---:|:---:|---:|---:|---:|---|
| 0.0015 | yes | 48.01 | 225.39 | 1966.19 | controlled |
| 0.003 | yes | 40.02 | 164.61 | -760.85 | baseline |
| 0.006 | yes | 48.64 | 128.55 | -3686.20 | controlled |
| 0.012 | yes | 69.27 | 63.84 | -6136.79 | finite but overshoots |

The practical stable interval is therefore `[0.0015, 0.006]` for this pilot
checkpoint. η=0.012 is retained as a boundary result rather than being
silently classified as stable solely because it remained finite.
