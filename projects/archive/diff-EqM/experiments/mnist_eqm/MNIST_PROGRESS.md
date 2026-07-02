# MNIST inpainting rung — progress

Real-dataset inpainting (replaces the too-easy maze-inpaint). Reuses real MNIST
(torchvision) + standard RePaint-style clamp masks + a classifier-consistency oracle.

## Step A — EqM + classifier (done)
- Small unconditional EqM (651K, faithful EqM target), 12 epochs / 20k MNIST, CPU. loss→2.66.
- Classifier oracle (SmallCNN) test acc 0.967.

## Mask operating points (failure headroom)
| mask | frac | fail-rate |
|---|---|---|
| center | 0.40 | 0.33 |
| half | 0.55 | 0.27 |
| center | 0.55 | 0.65 |
| center | 0.70 | 0.83 |

Non-degenerate headroom → good metacognition testbed. Sweep running on the first 3.

## Step B — metacognition sweep (running)
mnist_sweep.py: 3 masks × 3 seeds, RePaint inpaint, probe over descent dynamics →
predict inconsistent inpaint, probe-restart vs random vs oracle at equal NFE.
Results land in runs/mnist_sweep/MNIST_SWEEP_SUMMARY.md.
