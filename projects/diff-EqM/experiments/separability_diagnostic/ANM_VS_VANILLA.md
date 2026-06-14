# ANM/v10 vs Vanilla — gradient-metacognition signal comparison

Question (Yilun-adjacent): does ANM (v10 hard-example) training **change** the
gradient-metacognition signal — i.e. is failure more/less detectable from descent
trajectory under ANM than vanilla?

## Runs compared
| run | ckpt | samples | n_labeled | label sanity (s4 raw) |
|---|---|---|---|---|
| vanilla | `stage_b_vanilla...0380000.pt` | 3000 | 1500 (750/750) | 0.627 OK |
| ANM/v10 | `imagenet1k_80ep_v10_b2_seed1/.../final.pt` | 512 | 359 (≈180/180) | **0.583 SUSPECT** |

## Signal (de-confounded within-norm AUROC)
| signal | vanilla | ANM/v10 |
|---|---|---|
| ‖f‖ only (norm) | 0.538 KILL | 0.540 KILL |
| s1 −⟨f,x⟩ | 0.609 WEAK | 0.592 KILL |
| s3 path-integral | 0.605 WEAK | 0.553 KILL |
| s8 norm-oscillation | 0.674 | 0.622 |
| latent-NN s4 (sanity) | 0.627 | 0.583 |
| **learned trajectory probe** | **0.82 GREEN** | **0.700 BORDERLINE** |

## Read
- ANM signal is **not stronger** than vanilla; scalars sit equal-or-weaker, learned
  probe 0.700 vs 0.82.
- **BUT underpowered + confounded:** ANM ran at 512 samples (vs 3000), n_labeled 359
  (vs 1500), and label sanity is below the 0.6 floor (0.583) → labels noisy, probe
  train set small. The 0.700 is depressed by N + label noise, not necessarily by ANM.
- Honest verdict: **INCONCLUSIVE that ANM changes the signal.** No evidence ANM helps
  metacognition; cannot rule out parity. The comparison is not apples-to-apples on N.

## To make it decision-grade
Re-run ANM diag at **3000 samples + 10k real refs** (match vanilla) so label sanity
clears 0.6 and the probe trains on ≥1500 labels. Command:
```
cd <REPO_ROOT> && V10_CKPT=projects/diff-EqM/results/imagenet1k_80ep_v10_b2_seed1/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt \
  NUM_SAMPLES=3000 NUM_REAL=10000 BATCH_SIZE=64 RUN_TAG=b2_v10_anm_3k \
  SUBMIT=1 bash projects/diff-EqM/experiments/separability_diagnostic/launch_anm_diag.sh
```
Only then compare learned-probe within-norm head-to-head with vanilla 0.82.

_Generated 2026-06-14. Scaffold: baseline_table.py, dynamics_probe.py, rejection_payoff.py, final_readout.py._
