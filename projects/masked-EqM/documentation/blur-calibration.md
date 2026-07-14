# Blur severity calibration (2026-07-14)

## Rule

Match blur_corrupt's clean-vs-corrupted difficulty to the existing p=0.5
mask task, per the 2026-07-14 goal. First attempt used raw latent MSE and
FAILED: mask_corrupt replaces ~50% of latent elements with unit-variance
Gaussian noise (unbounded-MSE corruption), while blur_corrupt is a bounded
smoothing operation whose MSE saturates as sigma grows (blur_corrupt(x1) ->
per-channel spatial mean of x1 as sigma -> large, a fixed finite value).
Empirically: mask latent MSE ~0.842 vs blur's ceiling ~0.47 even at sigma=8
(kernel radius=24, already averaging over nearly the whole 32x32 latent).
No sigma can close that gap in raw latent-MSE units -- the two corruptions
are not comparable there.

Switched to pixel-space LPIPS matching (decode both corrupted latents
through the VAE, compare to the clean decoded image with LPIPS-AlexNet).
LPIPS is a bounded perceptual-similarity metric, not raw variance, so it
doesn't have the asymmetric-ceiling problem. The original goal explicitly
allowed MSE-or-LPIPS as the matching criterion.

## Result

128 ImageNet-val images, EqM-B/2 pipeline (VAE encode -> latent, same as
train.py), binary search over sigma in [0.1, 8.0], LPIPS tolerance 1e-3:

| | LPIPS | latent MSE | pixel MSE |
|---|---|---|---|
| mask (p=0.5) | 0.746780 | 0.842469 | 0.398367 |
| blur (sigma=1.1029) | 0.746508 | 0.264159 | 0.068331 |

**Calibrated blur sigma = 1.1029** (kernel radius = ceil(3*sigma) = 4,
9x9 separable Gaussian kernel, reflect padding, applied per-channel via
depthwise conv2d on the VAE latent -- see `transport/corruption.py`).

LPIPS distances match closely (0.7468 vs 0.7465). Latent/pixel MSE remain
far apart by construction (documented above, not a calibration failure --
LPIPS was deliberately chosen as the comparable axis, raw MSE is expected
to diverge between these two corruption families and is reported for the
record, not as a matching target).

## Job history

- Job 30884282: raw-MSE approach, FAILED (assertion: target not
  bracketed by sigma range, blur MSE ceiling too low).
- Job 30886459: LPIPS approach, FAILED (CUDA OOM decoding 256 images in
  one VAE decode call on gpu_test's MIG 20GB slice).
- Job 30887896: LPIPS approach with chunked decode (batch_size=16),
  num_images=128, SUCCEEDED. Used for all blur-family seed0 training.
