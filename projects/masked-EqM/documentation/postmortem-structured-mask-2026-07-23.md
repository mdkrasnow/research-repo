# Stage 2 structured-mask postmortem — 2026-07-23

## Verdict

**KILL: Stage 2 promotion gate failed (2/5 conditions pass).** Do not launch the optional
structured-mask ratio sweep, longer training, or model scaling. The central mechanism prediction
reversed sign: Gaussian+structured-mask training is perceptually *worse* than Gaussian-only on
the frozen unseen-Fourier task.

## Frozen experiment

- EqM-B/2, IN-1K, one epoch, matched training seeds 0–2.
- Arms: Gaussian-only (negative control), structured-mask-only (specialist/positive control),
  Gaussian+structured-mask 1:1 (treatment).
- Frozen Fourier recovery: cutoff 0.10, 250 GD steps, identical 1,024-image manifest and
  deterministic per-image corruptions.
- Trained structured-mask recovery: `mask_prob=0.5`, 250 steps, 256 images.
- Generation: 2,000-sample FID, 250 steps.
- Statistics: hierarchical paired bootstrap (seeds, then images), 10,000 draws; Holm correction
  across the two parent comparisons.

## Results

All values are three-seed mean ± sample SD; lower is better.

| arm | Fourier LPIPS | Fourier MSE | structured-recovery LPIPS | recovery MSE | FID |
|---|---:|---:|---:|---:|---:|
| Gaussian | 0.7667 ± 0.0011 | 0.1496 ± 0.0100 | 0.6067 ± 0.0032 | 0.6706 ± 0.0143 | 173.90 ± 0.76 |
| structured-mask | 0.7882 ± 0.0010 | **0.1149 ± 0.0006** | **0.3268 ± 0.0010** | **0.2296 ± 0.0027** | 231.32 ± 2.75 |
| Gaussian + structured-mask | 0.7731 ± 0.0024 | 0.1273 ± 0.0022 | 0.3488 ± 0.0006 | 0.2508 ± 0.0024 | 178.49 ± 1.21 |

| preregistered condition | result | verdict |
|---|---|---|
| mean delta_G ≥ +0.010 | **−0.00640**, 95% CI [−0.00893, −0.00360], Holm p < 0.0001 | **FAIL** |
| treatment beats both parents in 3/3 seeds | **0/3**; loses to Gaussian in every seed | **FAIL** |
| per-image win rate vs Gaussian > 75% | **27.6%** (seed rates 28.1%, 31.3%, 23.4%) | **FAIL** |
| treatment FID within +15 of Gaussian | +4.60 | PASS |
| trained structured recovery is strong | treatment and specialist beat Gaussian in 3/3 seeds | PASS |

The treatment beats the structured specialist on Fourier LPIPS decisively
(delta_structmask=+0.01511, 95% CI [+0.01383,+0.01677], 89.2% per-image wins), but that only
shows that retaining the Gaussian component prevents the specialist's larger perceptual
regression. It does not beat the Gaussian baseline.

## Mechanism read

The implementation and harness worked:

- The positive/specialist control learned the structured inverse strongly (recovery LPIPS
  0.327 vs Gaussian 0.607).
- The treatment retained most of that skill (0.349) while preserving generation quality
  (FID +4.60).
- All nine jobs completed cleanly; all 27 outputs and 144 probe images exist.

What failed is the proposed transfer mechanism. Contiguous spatial completion is not a useful
proxy for perceptual recovery from severe Fourier low-pass corruption at this scale. The
MSE/LPIPS disagreement is especially informative: structured training improves Fourier MSE
(0.127 vs Gaussian 0.150) while worsening Fourier LPIPS (0.773 vs 0.767). As in the earlier
Fourier work, pixel error rewards smoother/blurrer reconstructions that are perceptually worse.
The preregistered LPIPS endpoint correctly rejects this apparent MSE improvement.

## Sample probe

Each checkpoint generated 2,000 images for FID (well above the mandatory ≥16-image smoke probe);
16 examples per checkpoint were persisted (144 total). Representative spot checks agree with FID:
Gaussian and mixed-treatment samples have similar early-training coherence, while the structured
specialist is visibly more over-textured/degraded. There is no sampler-collapse signature hidden
by the scalar metrics.

## Decision

Per the registered kill rule, this branch ends here:

- no structured-mask ratio retune (reserved only for improved Fourier transfer with bad FID);
- no multi-epoch extension;
- no model scaling;
- no qualitative Fourier-recovery grids (reserved for a promoted candidate).

The next research direction requires user/PI choice and a fresh mechanism proposal. No new
experiment is launched automatically.
