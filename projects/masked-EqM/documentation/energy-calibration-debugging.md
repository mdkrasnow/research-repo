# Held-out energy calibration — validation blocker

## 2026-07-28: epoch-8 checkpoint retention failure

The frozen monotonicity output remains intact at
`/n/home03/mkrasnow/energy_monotonicity_full`, including the 2,048-image,
two-noise-per-image bank, checkpoint manifest, hashes, and full effective-field
caches.  Its cache-based calibration pass completed successfully as SLURM job
35804997.

Required fresh 256-trajectory 21/101-point scalar-versus-line validation was
submitted as job 35808090.  It failed before inference because the manifest's
exact epoch-8 checkpoint paths no longer exist: the three `checkpoints/`
directories under `longer8synced_{none,dot,direct}_seed0_job3544022*/` are
empty.  The loader correctly rejected them rather than substituting later
epoch-15/40 checkpoints or creating a new evaluation bank.

No checkpoint is retrained, selected, or replaced.  A formal calibration verdict
must remain inconclusive until exact SHA-256-matching copies of all three epoch-8
EMA checkpoints are restored, or an already completed dense validation artifact
on the same frozen bank is recovered.
