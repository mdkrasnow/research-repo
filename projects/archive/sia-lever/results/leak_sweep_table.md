# Leak-strength robustness sweep (5 seeds, 1500 steps)

shortcut = alpha*target + (1-alpha)*noise. alpha=1.0 is the Phase-1 adversarial leak.
Two findings:
1. SHORTCUT-CHEATING (low neg_control + high shortcut_sens) appears ONLY at the full leak alpha=1.0 (shortcut_win ~0.80). At alpha<=0.75 prediction-only stops reading the noisy shortcut (shortcut_sens ~0, neg_control ~honest ~1.0). => the trap is genuinely ADVERSARIAL: only a perfect leak makes the shortcut the path of least resistance. Validates framing Phase 1 as an adversarial trap.
2. The H->W REPAIR generalizes to EVERY alpha: prediction-only leaves the learned action structurally broken (composition_error high and seed-noisy at all alpha); H->W installs a clean group action (composition_error ~0.002, tiny CI) at every alpha. The robust claim is "structural H->W fixes group structure"; pure shortcut-cheating is the adversarial corner.

| alpha | arm | shortcut_sens | composition_err | neg_control | clean_mse | shortcut_win frac |
|---|---|---|---|---|---|---|
| 1.0 | predonly | 0.997±0.057 | 5.913±4.877 | 0.172±0.374 | 0.005±0.014 | 0.80 |
| 1.0 | htow | 0.002±0.002 | 0.002±0.001 | 1.046±0.040 | 0.008±0.013 | 0.00 |
| 0.75 | predonly | 0.032±0.023 | 13.684±16.975 | 0.943±0.151 | 0.007±0.015 | 0.00 |
| 0.75 | htow | 0.000±0.001 | 0.002±0.001 | 1.079±0.042 | 0.005±0.010 | 0.00 |
| 0.5 | predonly | 0.011±0.017 | 2.398±4.307 | 1.057±0.072 | 0.014±0.017 | 0.00 |
| 0.5 | htow | 0.002±0.005 | 0.002±0.002 | 1.080±0.055 | 0.010±0.012 | 0.00 |
| 0.25 | predonly | 0.003±0.006 | 4.865±10.339 | 1.081±0.058 | 0.009±0.022 | 0.00 |
| 0.25 | htow | 0.001±0.003 | 0.002±0.004 | 1.085±0.060 | 0.008±0.018 | 0.00 |
| 0.0 | predonly | 0.000±0.001 | 7.936±18.175 | 1.083±0.058 | 0.011±0.029 | 0.00 |
| 0.0 | htow | -0.001±0.003 | 0.004±0.010 | 1.082±0.062 | 0.013±0.033 | 0.00 |
