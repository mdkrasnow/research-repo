"""
Unit tests for fb_direct/adaptive_clip.py (2026-08-10, growing-instability
diagnostic follow-up). Pure-function logic, no model/GPU needed.

Run: python tests/test_fb_direct_adaptive_clip.py  (CPU, instant)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fb_direct.adaptive_clip import adaptive_clip_update, adaptive_clip_threshold


def test_bootstrap_on_first_call():
    ema = adaptive_clip_update(None, 3.7, decay=0.99)
    assert ema == 3.7, f"first call should bootstrap to the raw value, got {ema}"
    print("PASS bootstrap: EMA(None, x) == x")


def test_converges_to_constant_input():
    ema = None
    for _ in range(2000):
        ema = adaptive_clip_update(ema, 2.0, decay=0.99)
    assert abs(ema - 2.0) < 1e-6, f"EMA should converge to a constant input, got {ema}"
    print("PASS convergence: EMA -> constant input under repeated updates")


def test_safety_cap_limits_single_outlier_impact():
    ema = 1.0
    # A 1000x outlier, uncapped, would drag the EMA to
    # 0.99*1.0 + 0.01*1000 = 10.99. With cap=10x prev_ema (=10.0), the
    # update input is capped at 10.0: 0.99*1.0 + 0.01*10.0 = 1.089.
    new_ema = adaptive_clip_update(ema, 1000.0, decay=0.99, safety_cap_multiple=10.0)
    expected = 0.99 * 1.0 + 0.01 * 10.0
    assert abs(new_ema - expected) < 1e-9, f"expected {expected}, got {new_ema}"
    assert new_ema < 2.0, "a single 1000x outlier should not roughly double the EMA in one step"
    print(f"PASS safety cap: single 1000x-outlier step moved EMA {ema}->{new_ema:.4f}, "
          f"not toward {0.99*1.0 + 0.01*1000.0:.2f} (uncapped)")


def test_threshold_scales_with_ema():
    assert adaptive_clip_threshold(2.0, 4.0) == 8.0
    assert adaptive_clip_threshold(0.5, 3.0) == 1.5
    print("PASS threshold = ema * factor")


def test_tracks_slow_drift():
    """The core motivating property: if the natural gradient scale drifts
    upward over training, the EMA should track it (not stay pinned to the
    early value), so the derived threshold doesn't become progressively
    more binding the way a FIXED constant would."""
    ema = None
    for step in range(3000):
        # Slowly drifting scale: 1.0 -> 1.5 over 3000 steps.
        true_scale = 1.0 + 0.5 * (step / 3000)
        ema = adaptive_clip_update(ema, true_scale, decay=0.995)
    assert 1.3 < ema < 1.55, f"EMA should have tracked the upward drift to ~1.5, got {ema}"
    print(f"PASS drift tracking: EMA followed a slow 1.0->1.5 scale drift, ended at {ema:.3f}")


def test_rejects_invalid_params():
    for bad_decay in (0.0, 1.0, -0.1, 1.5):
        try:
            adaptive_clip_update(1.0, 2.0, decay=bad_decay)
        except ValueError:
            continue
        raise AssertionError(f"decay={bad_decay} should have raised ValueError")
    for bad_factor in (0.0, -1.0):
        try:
            adaptive_clip_threshold(1.0, bad_factor)
        except ValueError:
            continue
        raise AssertionError(f"factor={bad_factor} should have raised ValueError")
    print("PASS invalid decay/factor values are rejected")


if __name__ == "__main__":
    test_bootstrap_on_first_call()
    test_converges_to_constant_input()
    test_safety_cap_limits_single_outlier_impact()
    test_threshold_scales_with_ema()
    test_tracks_slow_drift()
    test_rejects_invalid_params()
    print("ALL ADAPTIVE-CLIP TESTS PASSED")
