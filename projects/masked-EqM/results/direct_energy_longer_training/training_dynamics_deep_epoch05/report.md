# Deeper training-dynamics analysis

This analysis uses 5,000-step means and 40,000-step rolling slopes; raw 50-step log values are too noisy for direct extrapolation.

## Slopes (loss / step)

| Window | none | dot | direct | direct−dot slope |
|---|---:|---:|---:|---:|
| 40,000–80,000 | -3.970e-06 | -3.566e-06 | -4.721e-06 | -1.155e-06 |
| 80,000–120,000 | -2.021e-06 | -1.867e-06 | -2.069e-06 | -2.026e-07 |
| 120,000–160,000 | -1.725e-06 | -1.469e-06 | -1.689e-06 | -2.203e-07 |
| 160,000–200,100 | -1.947e-07 | -1.330e-07 | -7.737e-07 | -6.408e-07 |
| 40,000–200,100 | -2.070e-06 | -1.935e-06 | -2.285e-06 | -3.504e-07 |

## Interpretation

- The none–dot loss gap does not close after step 40k; it is approximately flat to slightly wider. None is therefore not on a credible loss-parity trajectory with dot, even though it has the lowest absolute loss.
- The direct–dot gap is different: it shrinks from the early continuation and is approximately zero by 200k steps. In the final 40k window, direct's slope is more negative than dot's, but the confidence is weak at this noise level. The one-epoch slope extrapolation is not a valid reason to predict an 80-epoch failure.
- The three arms have distinct loss floors (roughly 10.67 none, 11.07 dot, 11.1 direct from exponential fits). Absolute training loss is not a fair cross-parameterization quality metric: the FID probes show none has lower loss but worse FID, while dot/direct have better FID.
- Direct remains about 1.9× slower than none in optimizer steps, so equal wall-clock comparisons should not be confused with equal-step comparisons.

The remaining discriminating experiment is not to wait for none to catch dot in training loss. It is to test whether direct's late loss-gap closure persists through epoch 8 and whether its already-near-dot FID remains tied or separates.

## FID trajectory cross-check

The matched 2K probes are:

| Arm | Epoch 2 | Epoch 3 | Epoch 5 | Epoch 2→5 change |
|---|---:|---:|---:|---:|
| none | 152.86 | 144.02 | 134.39 | −18.47 |
| dot | 144.69 | 136.28 | 127.72 | −16.97 |
| direct | 145.81 | 134.74 | 128.02 | −17.79 |

None's FID gap to dot is 8.17 → 7.75 → 6.67, so it is narrowing by about 0.51 FID/epoch under a three-point linear fit—not the same sign as the training-loss gap. A naive linear parity extrapolation lands near epoch 18, not 80, but this is underpowered and FID is noisy/nonlinear. Direct is already statistically indistinguishable at this probe resolution (gap to dot: +1.11, −1.54, +0.29). Thus the loss-slope argument cannot be used to reject direct or to claim none will never catch up: the quality metric and the regression loss are ordering the arms differently.
