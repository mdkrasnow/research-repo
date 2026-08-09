"""
EMA-adaptive gradient-norm clipping (2026-08-10, growing-instability
diagnostic follow-up).

Not NFNets' Adaptive Gradient Clipping (Brock et al. 2021) -- that's a
per-parameter ratio of grad norm to weight norm, a different technique for
a different problem (stable high-LR training without normalization layers).
This is simpler and targets a specific, narrower failure mode identified in
the fwrev arms' quartile forensics (2026-08-10): the fixed clip threshold
(a single constant chosen once) becomes progressively MORE BINDING if the
natural gradient-norm scale drifts upward over training, inflating the
observed clip rate as a pure artifact of the threshold being outgrown --
independent of whether the underlying landscape is actually getting
rougher (the question the curvature-vs-clip diagnostic addresses
separately). Tracking the scale via EMA and clipping at a constant
multiple of it makes "clip rate stays roughly constant" a design
invariant instead of something that has to be hoped for.

This module is a pure-function extraction of the update rule so it can be
unit tested without running training (see
tests/test_fb_direct_adaptive_clip.py).
"""


def adaptive_clip_update(prev_ema, raw_norm, decay, safety_cap_multiple=10.0):
    """One EMA update step.

    prev_ema: float or None (None on the very first call -- bootstraps the
      EMA to the first observed value).
    raw_norm: the current step's UNCLIPPED gradient norm.
    decay: EMA decay in (0, 1); larger = slower-adapting.
    safety_cap_multiple: the value used to UPDATE the EMA (not the value
      used to decide clipping) is capped at prev_ema * safety_cap_multiple,
      so a single freak large-gradient event can't corrupt the running
      scale estimate the clip threshold is derived from. Only applies once
      prev_ema exists (no cap on the bootstrap value).

    Returns the new EMA value.
    """
    if decay <= 0.0 or decay >= 1.0:
        raise ValueError(f"decay must be in (0, 1), got {decay}")
    if prev_ema is None:
        return float(raw_norm)
    update_val = min(float(raw_norm), prev_ema * safety_cap_multiple)
    return decay * prev_ema + (1.0 - decay) * update_val


def adaptive_clip_threshold(ema, factor):
    """The clip threshold implied by the current EMA estimate."""
    if factor <= 0.0:
        raise ValueError(f"factor must be positive, got {factor}")
    return ema * factor
