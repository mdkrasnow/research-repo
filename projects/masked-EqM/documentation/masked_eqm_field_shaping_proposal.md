# Variant proposal: pixel-masked endpoint continuation

Variant name: `masked_eqm_field_shaping_75g25m`

Hypothesis: continuing a trained EqM-B/2 field for ten epochs with 75% ordinary Gaussian endpoints and 25% independently Bernoulli-masked pixel endpoints improves recovery from unseen 30%-area contiguous block masks without worsening paired FID-10k by more than 1.0.

Failure mode addressed: ordinary Gaussian endpoint supervision may weakly constrain structured off-Gaussian directions near the learned field. The held-out block-mask test asks whether constraints learned from noncontiguous Bernoulli pixel masks transfer beyond the training corruption family.

EqM compatibility argument: this is not an auxiliary loss and does not alter the EqM geometry. A masked pixel image is encoded with the same frozen VAE and used only as `x0`; the existing linear interpolation, `get_ct(t)`, velocity target, model output, MSE loss, and positive sampling update remain unchanged. Thus the treatment is exactly the ordinary EqM objective on a different endpoint for a predeclared 25% of examples.

Loss definition: for clean latent `x`, draw the ordinary Gaussian endpoint and gamma exactly as the control does. On treatment-selected examples only, replace the Gaussian endpoint with `encode(M*I + (1-M)*epsilon_pixel)`, where per-image missing ratio is uniform on `[0.10,0.50]`, `M` is an independent pixel Bernoulli keep mask shared across RGB channels, and fill noise is drawn in the normalized pixel space. Apply the repository target `(x - endpoint) * c(gamma)` and its existing MSE.

Expected diagnostics if working: finite losses and gradients; realized treatment frequency near 0.25; realized missing ratios spanning `[0.10,0.50]` with mean near 0.30; positive paired block-mask LPIPS-recovery delta; no more than +1.0 paired FID degradation.

Expected diagnostics if failing: no positive paired recovery CI, sample divergence, or FID degradation beyond +1.0. A training-mask reconstruction gain alone is not evidence for the hypothesis.

Minimal test: deterministic pixel masks and endpoints, zero-probability equivalence to the original Gaussian loss, sign and gradient checks, exact checkpoint restoration, a paired 100-update epoch-15 smoke, 16-image held-out block recovery, and checkpoint loading through the FID path.

Positive control: the recovery metric harness evaluates an oracle output equal to the paired clean image, which must achieve the maximal possible recovery from the same corrupted input. This is a nonmodel plumbing control and is excluded from the primary comparison.

Negative control: the compute-matched 100% Gaussian continuation arm. Step-0/no-update recovery is also retained as the within-example floor.

Promotion rule: proceed from smoke to the locked full protocol only when all required tests pass and both arms have matching initial hashes, optimizer state, sampler order, update count, batch configuration, and finite smoke diagnostics.

Primary success rule: at epoch 40, the image-cluster bootstrap 95% CI lower bound for treatment-minus-control LPIPS recovery is greater than zero and treatment FID-10k minus control FID-10k is at most 1.0.

Kill rule: classify using the four predeclared outcomes. Do not tune mixture probability, mask ratio, budget, recovery steps, step size, evaluation corruption, metric, or thresholds after observing epoch-40 results. Epochs 15 and 80 repeat the locked protocol as stage-sensitivity replications, not independent seeds.
