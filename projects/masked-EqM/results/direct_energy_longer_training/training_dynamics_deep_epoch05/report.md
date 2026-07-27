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
