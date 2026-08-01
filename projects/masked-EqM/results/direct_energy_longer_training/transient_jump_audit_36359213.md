# Direct transient-loss event audit: job 36359213

**Question.** Was the direct-model loss jump near step 1.52M caused by a hardware fault?

## Evidence

- Job `36359213` ran on `holygpu8a16403` with 4 H200 GPUs, completed with Slurm
  state `COMPLETED`, exit `0:0`, and elapsed time `19:56:14`.
- The stderr file contains only git checkout/setup messages. There are no CUDA,
  Xid, NCCL, OOM, traceback, NaN, or Inf diagnostics.
- The node accounting record shows one normal completed job; no node-level fault
  or requeue was recorded.
- The stdout trace is normal through step `1,516,700` (loss about 10.7--10.9).
  Throughput then falls from about 12.6 steps/s to 5.1--8.4 steps/s between
  steps `1,516,750` and `1,517,100`.
- The loss shift follows the slowdown: 10.70 at step `1,517,500`, 12.11 at
  `1,517,550`, and 23.66--23.72 at steps `1,517,600`--`1,517,650`.
  It remains elevated for hundreds of steps and later settles around 11.5--11.9;
  this is not a one-line logging glitch.
- The direct continuation job `36597492` subsequently ran around 12.7 steps/s
  with loss near 11.0--11.3 and no NaN, showing that the model was not in an
  unrecoverable numerical state.

## Assessment

The evidence does **not** support a catastrophic hardware failure. A transient
GPU fault would normally leave CUDA/Xid/NCCL evidence, terminate or requeue the
job, or persist as degraded throughput. None occurred. A hardware/driver event
cannot be ruled out absolutely because node-level Xid telemetry was not captured
in the training artifact, but it is low probability.

There are two distinct signals:

1. A real compute/data-path stall (throughput fell before the loss changed).
2. A sustained optimizer/model-state transition (loss stayed high after
   throughput recovered).

The most plausible explanation is a rare high-loss or high-gradient batch,
possibly coincident with a data-loader/filesystem stall, amplified by the direct
model's second-order gradient path and constant learning-rate Adam updates. This
is a hypothesis, not a proof: the run did not log batch IDs, gradient norms,
parameter/EMA norms, field norms, AMP scale, or per-timestep loss, so the exact
trigger is not identifiable from this run.

For comparison, dot's later failure is clearly different: it shows loss values
from hundreds to billions followed by NaN without a preceding throughput stall.
That is numerical/optimization divergence, not evidence that the direct event was
the same failure mode.

## Recommended follow-up

1. Evaluate direct checkpoints at 1.45M, 1.50M, 1.55M, and 1.60M with the same
   2K-sample FID protocol to localize when sample quality changes.
2. Inspect those checkpoints for finite parameters, parameter/EMA norms, and
   optimizer-state norms.
3. In a diagnostic continuation, log batch/sample IDs, gradient and parameter
   norms, field/energy statistics, per-timestep loss, learning rate, AMP scale,
   and data-loader wait time. This is the minimum instrumentation needed to
   distinguish an outlier batch from an I/O stall or optimizer amplification.

**Conclusion:** treat this as a real training transient with a preceding stall,
not as a confirmed hardware failure. The current evidence favors data/gradient
outlier plus optimization amplification; node telemetry and richer training
diagnostics are required for attribution.
