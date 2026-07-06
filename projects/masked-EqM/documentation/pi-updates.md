# PI Updates — masked-EqM

## 2026-07-06 draft (mechanism finding — hard-constrain ceiling invalid for EqM)

**Trigger**: bug/finding invalidating a planned eval design (mid-project, not a full result).

Context: building 3-arm masked-recovery eval (floor=gaussian-trained, ceiling=hard-constrained
oracle, treatment=mask-trained) per structured-start-state plan.

Finding: hard-constrain ceiling (reset visible pixels to ground truth every sampler step) does
NOT function as a meaningful oracle for EqM. Root cause: EqM's target scaling `c(gamma) -> 0`
as state nears data manifold (`transport.py:122-126`) — near true/visible pixels the model's own
learned dynamics already drift ~0 regardless of whether we force-constrain. So hard-constrain and
raw treatment converge (0.1403 vs 0.1434 masked-region MSE, within noise) — this is an architecture
property, not a bug (verified via RNG-seed fix; both runs use identical mask/noise, gap did not
open up).

Action taken: dropped hard-constrain as the positive control. Substituted VAE-encode/decode
roundtrip (skip corruption/model, measure only VAE's own reconstruction floor) as an
architecture-independent positive control — gives real headroom number (0.0149 masked MSE),
confirms metric is not saturated.

Revised eval design: floor (gaussian-trained treatment score) vs treatment (mask-trained /
mixture-trained score) vs oracle (VAE roundtrip, architecture-independent). No ceiling-via-hard-
constrain arm.

**Status of step2_mask_sanity gate**: still PENDING outcome metrics. gaussian_seed0_v5
(job 28842669) and mask_seed0_v5 (job 28842681) sanity training running on seas_gpu as of this
draft; masked-recovery eval to follow immediately on completion. mixture checkpoint already
evaluated (0.1403) but not yet interpretable without the isolated gaussian/mask numbers to compare
against.

**Ask of PI**: none blocking — proceeding per pre-registered plan. Flagging for visibility per
mechanism-finding trigger in AGENTS.md PI-update protocol.
