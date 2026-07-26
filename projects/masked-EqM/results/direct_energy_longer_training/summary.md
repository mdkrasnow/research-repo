# Longer-training campaign

Status: running. The campaign continues the matched epoch-1 checkpoints to
epochs 2, 3, and 5 without changing optimizer, learning rate, data, target,
model size, EMA, or sampler. Because the original checkpoints did not include
DataLoader/RNG state, continuation restores model/EMA/optimizer and inferred
step/epoch state but does not claim exact minibatch-order continuation.

None is staged first to bound home03 storage. Dot and direct will follow after
epoch 2/3/5 evaluation and cleanup. The epoch-1 training logs and FID probes
remain immutable controls.
