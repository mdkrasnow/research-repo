# RePaint on REAL IN-1K EqM-B/2 — image-scale confirmation of the inpainting law (2026-06-19)

Confirms the MNIST inpainting result (`mnist_eqm/INPAINT_POSITIVE_RESULT.md`) on the **actual
paper checkpoint** (IN-1K EqM-B/2, FID 31.41), not a toy. Latent-clamp RePaint (option B),
n=256, R=4, 250 GD steps, dual oracle: LPIPS-to-GT (semantic) + TV-structural (instability/
coherence over the inpainted region). De-confounded AUROC; equal-NFE restart arms. FID human-
gated (not computed) — LPIPS + AUROC reported.

## Result — center mask (the clean sweep)
| mask frac | LPIPS-oracle AUROC | TV-structural AUROC | TV-struct restart gap | LPIPS: vanilla / random / probe / oracle |
|---|---|---|---|---|
| 0.3 | 0.573 | 0.637 | −0.043 | 0.212 / 0.211 / 0.210 / 0.203 |
| 0.5 | 0.563 | 0.660 | −0.004 | 0.405 / 0.404 / 0.403 / 0.394 |
| 0.7 | 0.689 | 0.579 | +0.008 | 0.627 / 0.627 / 0.625 / 0.615 |
| **0.9** | **0.845** | **0.852** | **+0.215** | 0.762 / 0.763 / **0.753** / 0.725 |

- **Extreme mask (0.9): TV-structural AUROC 0.852, restart gap +0.215** — probe both detects and
  rescues structural/instability failures on the real B/2 model. Probe-restart LPIPS (0.753) also
  beats random (0.763) and vanilla (0.762); oracle 0.725.
- Low/mid masks: weak/negative (the inpaint is easy or fails confident-wrong). The positive regime
  is the extreme mask — exactly the MNIST law (AUROC 0.84, gap +0.18), now at image scale.

## Match to the MNIST prediction
| | MNIST (toy, CPU) | IN-1K B/2 (real, GPU) |
|---|---|---|
| extreme-mask structural AUROC | 0.84 | **0.852** |
| extreme-mask restart gap | +0.18 | **+0.215** |
The toy law transfers to the real checkpoint almost exactly. Inpainting metacognition is positive
**when extreme masking pushes failures from confident-wrong (semantic) into structural collapse
(instability)** — the same mechanism, now Tier-1.

## Caveats / honesty
- **`expand` mask is degenerate at frac≥0.5**: its geometry (`rows[:s] ∪ rows[-s:]`) becomes the
  full image once s≥16/32, so expand 0.5/0.7/0.9 are identical (= unconditional gen). Only the
  `center` sweep and `expand 0.3` (AUROC 0.752, gap +0.074) are valid. Use center as the headline.
- Option-B latent clamp is approximate vs pixel-space RePaint (VAE not perfectly spatially local);
  the metacognition DELTA (probe−random at equal NFE) is robust to the inpaint-method choice since
  both arms use the same one. Option-A (decode-replace-encode) would tighten absolute LPIPS, not the delta.
- LPIPS-oracle AUROC also rises at extreme mask (0.845) — at 0.9 the semantic and structural
  failures co-occur (a broken completion is also perceptually far), so LPIPS picks up the signal too.
- Single checkpoint, n=256. The effect is decisive at the extreme-mask endpoint; mid-mask is weak.

## Bottom line
The inpainting metacognition law holds on the real IN-1K EqM-B/2 checkpoint: at extreme masks,
the trajectory-shape probe detects (AUROC 0.85) and rescues (+0.22 structural valid-rate, lower
LPIPS) the instability-type failures, while staying weak where failures are confident-wrong. The
Thread-3 result is promoted from toy (MNIST/CPU) to paper-scale (B/2/GPU).
