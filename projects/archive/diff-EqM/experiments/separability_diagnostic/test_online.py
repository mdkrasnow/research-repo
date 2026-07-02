"""Synthetic tests for the online adaptive sampler (CPU, seconds).

Checks the load-bearing properties WITHOUT EqM/GPU:
  1. equal-NFE: probe-restart and random-restart spend byte-identical NFE.
  2. signal case: when the partial probe is informative, probe-restart beats
     random-restart at that equal NFE.
  3. no-signal case: when risk is shuffled (uninformative), probe ≈ random
     (no spurious win) — guards against an NFE/bookkeeping artifact that would
     make probe look good even with a dead probe.

Run: python test_online.py
"""
import sys
import types

import numpy as np

import online_adaptive_sampler as OAS


def run(seed=0):
    args = types.SimpleNamespace(slots=4000, steps=249, kfrac=0.4, flag_frac=0.3,
                                 seed=seed, out="/tmp/online_test_signal")
    return OAS.run_mock(args)


def run_nosignal(seed=0):
    """Monkeypatch probe_risk to return noise -> probe selection is uninformative."""
    orig = OAS.probe_risk
    rng = np.random.default_rng(123)
    OAS.probe_risk = lambda nk, dk, art: rng.uniform(0, 1, nk.shape[0])
    try:
        args = types.SimpleNamespace(slots=4000, steps=249, kfrac=0.4, flag_frac=0.3,
                                     seed=seed, out="/tmp/online_test_nosignal")
        res = OAS.run_mock(args)
    finally:
        OAS.probe_risk = orig
    return res


def main():
    ok = True
    sig = run()
    nfe = sig["nfe"]
    matched = nfe["probe-restart"] == nfe["random-restart"]
    q = sig["mean_quality_lower_is_better"]
    probe_wins = q["probe-restart"] < q["random-restart"] - 0.01
    c1 = matched and probe_wins
    print(f"[signal]    NFE matched={matched}  probe<random={probe_wins}  "
          f"(probe {q['probe-restart']:.3f} vs random {q['random-restart']:.3f})  "
          f"[{'PASS' if c1 else 'FAIL'}]")

    nos = run_nosignal()
    qn = nos["mean_quality_lower_is_better"]
    matched_n = nos["nfe"]["probe-restart"] == nos["nfe"]["random-restart"]
    near = abs(qn["probe-restart"] - qn["random-restart"]) < 0.05
    c2 = matched_n and near
    print(f"[no-signal] NFE matched={matched_n}  probe≈random={near}  "
          f"(probe {qn['probe-restart']:.3f} vs random {qn['random-restart']:.3f})  "
          f"[{'PASS' if c2 else 'FAIL'}]")

    ok = c1 and c2
    print("=" * 50)
    print("ALL ONLINE TESTS PASS" if ok else "SOME ONLINE TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
