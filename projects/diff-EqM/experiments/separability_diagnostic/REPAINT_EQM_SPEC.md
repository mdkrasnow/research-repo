# RePaint-on-IN-1K-EqM — image inpainting spec (June-18 cluster queue)

The MNIST rung is the CPU proxy. The *real* image-inpainting result = standard RePaint
masks on the actual IN-1K EqM-B/2 checkpoint, with the trajectory-metacognition restart.
GPU-only → queued for cluster return (~Jun 18; see slurm/jobs/CLUSTER_DOWN.md).

## The latent-space subtlety (why this needs design, not a copy of MNIST)
EqM samples in VAE latent z∈R^{4×32×32}; RePaint masks are pixel-space (256×256). Two
correct options:

- **(A) decode–replace–encode per step (faithful, expensive).** Each GD step: decode
  z→x, overwrite observed pixels with the true image, re-encode →z, continue. Exact
  pixel-space constraint; ~2 VAE passes/step (slow but correct). Use for the headline.
- **(B) latent clamp (fast, approximate).** Encode GT once → z_known; downsample the
  pixel mask to 32×32 (÷8); clamp z in the observed region to z_known each step. Cheap,
  but the VAE is not perfectly spatially local so the constraint bleeds at block edges.
  Use for the smoke / large sweeps.

Start with (B) at smoke scale to validate the pipeline, then (A) for the reported number.

## Masks (standard RePaint set)
thin / thick / every-second-line / expand / half (andreas128/RePaint config). Reuse the
published mask images if fetchable; else synthesize matching ones. Report per-mask.

## Metric
LPIPS to GT on the inpainted region (RePaint standard) + FID on the full set. Oracle =
LPIPS-best of R (positive); random-restart = NEG; vanilla = draw0 floor.

## Metacognition wiring (reuse, unchanged)
- Log per-step norm/dot of f over the **masked latent region** (as maze_inpaint/
  mnist_inpaint do). Feed the SAME `feature_groups` shape features to the SAME probe.
- Train/deploy the partial probe (or the existing `probe_artifact.npz`, re-validated on
  inpainting trajectories) → P(bad inpaint).
- Arms (best-of-R, equal NFE): vanilla / random-restart / probe-restart / oracle(LPIPS).
- Success = probe-restart LPIPS < random-restart LPIPS at equal NFE.

## Build checklist (when cluster returns)
1. `repaint_eqm.py` — fork `probe_gated_sample.py`: add mask load + option (A)/(B)
   clamp in the GD loop; log masked-region dynamics; emit per-arm LPIPS + Inception feats.
2. `repaint_eqm.sbatch` — gpu, 4×A100, smoke (512, option B) → 5k/15k (option A on the
   headline mask). Mirror online_seed.sbatch structure.
3. `fid_lpips_agg.py` — per-arm LPIPS + FID + equal-NFE verdict (fork fid_online_agg).
4. Add to `fire_overnight.sh` once smoke-validated.

## Honesty
- Latent RePaint is approximate vs pixel-space RePaint; report which option produced each
  number. The point is the *metacognition delta* (probe-restart vs random at equal NFE),
  which is robust to the inpainting-method choice as long as both arms use the same one.
- This is the image-scale confirmation of the MNIST inpainting rung — gated on it passing.
