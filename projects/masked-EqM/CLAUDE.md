# masked-EqM — Project Goal

## Origin

Prior work (diff-EqM, archived `projects/archive/diff-EqM/`) applied adversarial negative mining (ANM/v10) to EqM. Worked, not novel enough. Discussed next direction w/ Yilun Du (Slack, 2026-07-02):

> Yilun: training to make EqM have more robust energy landscape — interpolate not just Gaussian noise → image, but also *partially masked images* and other structured starting points → ground truth image.
> Matthew: proposed Fourier-space corruption as alt/addition to Bernoulli pixel masking.
> Yilun: Fourier corruption could be good, but start simpler — partially masked images first.

## Research question

> Does broadening EqM's corruption/start distribution from pure Gaussian noise to structured corruptions produce a more robust and informative energy landscape?

Hypothesis: Gaussian starts teach EqM to descend from unstructured noise to data. Many bad inference states are structured-but-wrong (partial info, coarse structure w/ missing detail). Training from structured corruptions may make learned energy gradients useful in a wider neighborhood around the data manifold.

EqM already claims to naturally support partially-noised denoising, OOD detection, image composition (arXiv 2510.02300) — masking/structured-start training is aligned with the paper's own framing, not a random bolt-on.

**Method name: Structured Start-State EqM** (Fourier variant = Spectral Start-State EqM).

## Scope discipline (Yilun's steer — do not overcomplicate v1)

Single question per experiment. Do NOT start with phase corruption, band-drop, amplitude/phase separation, or adaptive hard negatives — those make the first result uninterpretable. Build order is strict:

### Step 1 — reproduce baseline
Get masked-EqM (fresh clone raywang4/EqM) running unmodified, reproduce official numbers. No modifications before this.

### Step 2 — Bernoulli pixel/patch masking (build this first, per Yilun explicit)
Start distribution:
```
z0 = m ⊙ x + (1-m) ⊙ ε        m = random mask, ε ~ N(0,I)
zt = (1-t) z0 + t x
```
Train w/ same EqM target as baseline. Lowest-risk first experiment.

### Step 3 — Fourier low-pass corruption (only after step 2 shows signal)
```
x̂ = F(x)
ẑ0 = M_ρ ⊙ x̂ + (1-M_ρ) ⊙ ε̂      M_ρ = radial low-pass mask
z0 = F⁻¹(ẑ0)
```
Coarse structure kept, high-freq detail missing → model learns gradients from structured-but-degraded state back to clean image.

### Step 4 — mixture ablation (Gaussian retained throughout, never fully replaced)
```
q(z0|x) = λ_G q_G + λ_M q_mask + λ_F q_fourier
```
Arms:
| Arm | Start distribution |
|---|---|
| A | Gaussian only (baseline) |
| B | Gaussian + mask |
| C | Gaussian + Fourier low-pass |
| D | Gaussian + mask + Fourier |

First mixture to try: 50% Gaussian / 25% mask / 25% Fourier (or simpler 50/50 Gaussian/structured).

### Step 5 — scale
CIFAR-level sanity only until real signal + diagnosed. No jump to IN-1K without a passing gate (same discipline as diff-EqM: proxy-scale results are filters, not publishable on their own).

## What to measure (not just FID)

1. **FID** — generation quality.
2. **Convergence / restart rate** — fewer samples falling into bad trajectories.
3. **Energy ordering** — sanity check learned energy ranks states correctly:
   `E(clean) < E(fourier-corrupt) < E(gaussian-noisy)`, and `E(clean) < E(good sample) < E(bad sample)`.
4. **Denoising/inpainting recovery** — masking/Fourier corruption are denoising-style tasks; test recovery quality vs baseline EqM.
5. **Trajectory diagnostics** — reuse prior shape-vector probe (from diff-EqM separability work) as a *diagnostic*, not the method: does new training reduce bad-shape trajectory rate? Story: prior probe found a failure mode, this training tries to remove it.

## Plan status

No formal phase gates / summer plan written yet. Write `documentation/summer-2026-plan.md` once step 2 (masking) shows signal at CIFAR scale.

## Scope

Only active project in this repo (root `AGENTS.md`). Do not touch `projects/archive/*`.
