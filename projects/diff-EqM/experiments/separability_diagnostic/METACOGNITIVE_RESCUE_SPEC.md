# Metacognitive Rescue Pilot — SPEC (not yet implemented)

Tiny, controlled test of whether acting on a **partial-trajectory risk score**
rescues bad EqM samples. Gate: only build this if `dynamics_probe.py` returns
GREEN or PROMISING (FULL within-norm OOF AUROC ≥ 0.70).

## Question
During sampling, can a risk score read from the *first part* of the relaxation
trajectory flag a sample as likely-garbage early enough to spend extra compute
rescuing it — and does that beat spending the same extra compute blindly?

## Arms (128 or 256 output slots; identical seeds across arms)
1. **vanilla** — standard GD, fixed N steps. (baseline)
2. **random-extended** — same TOTAL extra compute budget as the probe arm, but
   the extra steps/restarts are spent on a RANDOM subset of slots. (NEGATIVE
   control — isolates "more compute" from "the probe".)
3. **probe-extended** — at a decision step k_dec < N, compute the partial-
   trajectory risk score (the dynamics probe restricted to features available by
   step k_dec); if `risk > τ`, spend the extra budget on that slot
   (continue/perturb-restart); else stop early. (TREATMENT)

## Equal-compute constraint (load-bearing)
Total function evaluations (NFE) of random-extended == probe-extended. The probe
arm reallocates a fixed extra budget toward flagged slots; the random arm spends
the identical budget on randomly chosen slots. If probe ≈ random, the gain was
just compute, not the probe. This is the key control.

## Risk score
Re-train the dynamics probe on features computable from steps [0:k_dec] only
(truncated trajectory) against the same good/garbage labels, so the score is
causal (no peeking at the final state). Pick k_dec ~ 0.6–0.8·N.

## Metrics (cheap → expensive)
- Inception-feature NN-distance to real (primary, matches the label metric).
- Classifier (resnet50) max-softmax / entropy.
- Contact sheets of rescued vs not-rescued (eyeball).
- FID proxy on the small set (≤2k) — relative arm comparison only.

## Pass / fail
- WORKS: probe-extended NN-dist (and FID proxy) beats BOTH vanilla and
  random-extended by > noise. Random-extended must NOT explain the gain.
- NULL: probe ≈ random-extended → the early risk score does not actionably
  reallocate compute; revert to post-hoc rejection (already validated).

## Relation to existing code
`probe_gated_sample.py` already implements probe-guided **best-of-R** (full
restart, fixed R) at scale with vanilla/oracle controls — that is the *post-hoc /
fixed-budget* version. This pilot is the *partial-trajectory, adaptive-budget,
equal-compute* version, which is the true "metacognitive sampler". Reuse the probe
artifact + feature builders; the new piece is the truncated-trajectory probe and
the equal-NFE random control.

## Cost
128–256 slots × (1 + budget) draws on 1 GPU ≈ minutes–1h. gpu_test.
