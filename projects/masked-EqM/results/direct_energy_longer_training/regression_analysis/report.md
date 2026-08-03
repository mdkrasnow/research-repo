# Direct-energy epoch-15→40 regression forensic analysis

## Executive finding

The evidence supports a two-stage event: a short compute/data-path slowdown occurred first, then a rare optimization disturbance was amplified and retained by Adam, producing broad field degradation. The precise initiating batch cannot be recovered from the original instrumentation.

## Detected chronology

- First extreme slowdown: step 1,516,800, loss 10.7386, throughput 8.36 steps/s at 2026-07-31 11:43:40.
- First extreme loss: step 1,517,550, loss 12.1142, throughput 7.49 steps/s at 2026-07-31 11:45:18.
- Peak logged loss: step 1,517,650, loss 23.7174, throughput 8.46 steps/s at 2026-07-31 11:45:30.
- Minimum throughput: step 1,547,750, loss 12.1378, throughput 3.13 steps/s at 2026-07-31 12:26:51.

## Evidence-ranked causal assessment

- **high** — A compute/data-path slowdown preceded the visible loss transition. First extreme slowdown led first extreme loss by 750 logged steps and 98.0 seconds.
- **high** — The transition became a persistent model/optimizer-state change, not a one-line logging artifact. EMA mean loss changed from 11.311 at 1.50M to 13.255 at 1.55M and remained 12.322 at 1.60M.
- **medium-high** — Adam amplified or retained the disturbance across the network. The 1.50M→1.55M aggregate Adam second-moment displacement was 0.3985; checkpoint diagnostics localize degradation broadly rather than only to the scalar head.
- **high** — Curvature explosion was not the proximate failure mode. EMA Hessian-vector norm decreased from 6.348 to 4.867, while fixed-t energy descent remained correctly signed.
- **medium** — A rare gradient outlier is a plausible trigger, but the exact offending batch is unidentifiable. A matched continuation measured a 65.13 global-gradient outlier against p99 3.44, but the original run did not record batch IDs, per-step gradients, data wait time, or RNG state.
- **low / unsupported** — Hardware failure caused the regression. The job completed without CUDA, Xid, NCCL, OOM, NaN, or Inf evidence; missing node telemetry prevents an absolute exclusion.

## Generation consequence

The matched 2,000-sample direct FID worsened from 112.467 at epoch 15 to 160.110 at epoch 40 (Δ=+47.643). This establishes a generation regression by epoch 40, but the endpoint FID alone cannot assign it to the 1.50M→1.55M transition.

## Localization

The largest fixed-bank EMA loss increase occurs at t=0.8: 14.031 → 17.858 (Δ=+3.826), with cosine changing by -0.103.
The largest relative Adam second-moment displacements are:

- blocks.11: 126.4× its preceding-interval displacement (absolute Δ-norm 0.02813).
- energy_head: 21.0× its preceding-interval displacement (absolute Δ-norm 0.2556).
- blocks.0: 13.5× its preceding-interval displacement (absolute Δ-norm 0.1163).
- blocks.1: 3.8× its preceding-interval displacement (absolute Δ-norm 0.03535).
- t_embedder: 3.7× its preceding-interval displacement (absolute Δ-norm 0.2749).

## Claims boundary

This analysis localizes and characterizes the transition. It does not identify the exact training examples, prove that I/O caused the gradient outlier, prove that clipping would have preserved FID, or establish that the failure repeats across seeds.
