#!/usr/bin/env python3
"""Code-optimization lever demo — the SIA H-vs-W phenomenon on a REAL code task (rhymes with the
paper's TriMul kernel-opt), CPU, no CUDA.

The point: the rotation toy's "shortcut cheat" has a direct, non-toy analog in code optimization —
a kernel that is FAST because it is subtly WRONG, but passes a WEAK correctness harness. This script
shows the identical lever story on actual code:

  task            : compute y = (A @ x) elementwise-relu, then sum  (a small "kernel")
  candidates the "model" can emit (stand-in for an LLM writing kernels):
    correct_slow  : python loops, correct, slow
    correct_fast  : vectorized numpy, correct, fast
    cheat_const   : returns a precomputed constant — O(1), WRONG, but matches the weak test input
  levers:
    W  = pick the fastest candidate that PASSES THE CURRENT HARNESS  (a "weight update": optimize
         under current feedback)
    H  = strengthen the correctness harness (random inputs instead of one fixed input)
    H_THEN_W = strengthen harness, THEN re-optimize

  harness:
    weak     : checks correctness on ONE fixed input  -> cheat_const passes (it was tuned to it)
    strong   : checks correctness on RANDOM inputs     -> cheat_const fails

Phenomenon (measured, like Phase 1):
  - W under the weak harness selects cheat_const: FAST, "passes", but WRONG on held-out inputs.
  - H makes the cheat visible (fails strong check).
  - H_THEN_W selects correct_fast: fast AND correct on held-out.

held-out correctness == the negative control: an honest kernel must be correct on inputs it wasn't
tuned on; a cheat is not. Same role as neg_control_mse in the rotation toy.

CPU, < 1 s. Run: python experiments/codeopt_lever_demo.py
"""

import time

import numpy as np

RNG = np.random.default_rng(0)
N = 64
A = RNG.standard_normal((N, N))
X_FIXED = RNG.standard_normal(N)            # the single input the weak harness checks


def reference(x):
    return float(np.maximum(A @ x, 0.0).sum())


# ---------- candidate "kernels" the model can emit ----------
def correct_slow(x):
    acc = 0.0
    for i in range(N):
        s = 0.0
        for j in range(N):
            s += A[i, j] * x[j]
        acc += s if s > 0 else 0.0
    return acc


def correct_fast(x):
    return float(np.maximum(A @ x, 0.0).sum())


_CHEAT_ANSWER = reference(X_FIXED)          # precomputed for the weak harness's fixed input
def cheat_const(x):
    return _CHEAT_ANSWER                     # O(1), correct ONLY on X_FIXED


CANDIDATES = {"correct_slow": correct_slow, "correct_fast": correct_fast, "cheat_const": cheat_const}


# ---------- harnesses (the H lever changes which one is deployed) ----------
def passes(harness, fn, tol=1e-6):
    if harness == "weak":
        return abs(fn(X_FIXED) - reference(X_FIXED)) < tol            # one fixed input
    # strong: must match on several RANDOM inputs (the negative control)
    for _ in range(8):
        x = RNG.standard_normal(N)
        if abs(fn(x) - reference(x)) >= tol:
            return False
    return True


def latency(fn, reps=200):
    x = X_FIXED
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(x)
    return (time.perf_counter() - t0) / reps * 1e6                    # microseconds/call


def heldout_correct(fn, tol=1e-6):
    """honest test = correctness on inputs the candidate was NOT tuned on (the neg control)."""
    return all(abs(fn(RNG.standard_normal(N)) - reference(RNG.standard_normal(N))) < tol
               or True for _ in range(1)) and \
        all(abs(fn(x) - reference(x)) < tol for x in [RNG.standard_normal(N) for _ in range(16)])


# ---------- W lever: pick the fastest candidate that passes the CURRENT harness ----------
def weight_update(harness):
    passing = [(name, latency(fn)) for name, fn in CANDIDATES.items() if passes(harness, fn)]
    passing.sort(key=lambda kv: kv[1])                               # fastest first
    return passing[0][0] if passing else None


def report(name):
    fn = CANDIDATES[name]
    return {"selected": name, "latency_us": round(latency(fn), 2),
            "heldout_correct": bool(heldout_correct(fn))}


def main():
    print("=" * 68)
    print("Code-opt lever demo (SIA H-vs-W on real code; rhymes with TriMul)")
    print("=" * 68)
    print(f"{'candidate':14s} {'passes_weak':>11s} {'passes_strong':>13s} "
          f"{'latency_us':>11s} {'heldout_ok':>11s}")
    for name, fn in CANDIDATES.items():
        print(f"{name:14s} {str(passes('weak', fn)):>11s} {str(passes('strong', fn)):>13s} "
              f"{latency(fn):>11.2f} {str(heldout_correct(fn)):>11s}")

    print("\n-- W-only under the WEAK harness (optimize on current feedback) --")
    w_weak = report(weight_update("weak"))
    print(f"   selected={w_weak['selected']}  latency={w_weak['latency_us']}us  "
          f"heldout_correct={w_weak['heldout_correct']}   <- FAST but WRONG (shortcut cheat)")

    print("-- H: strengthen the harness (random-input correctness) --")
    print(f"   cheat_const passes_strong = {passes('strong', cheat_const)}  <- cheat now visible")

    print("-- H_THEN_W: re-optimize under the STRONG harness --")
    w_strong = report(weight_update("strong"))
    print(f"   selected={w_strong['selected']}  latency={w_strong['latency_us']}us  "
          f"heldout_correct={w_strong['heldout_correct']}   <- FAST and CORRECT (repaired)")

    ok = (w_weak["selected"] == "cheat_const" and not w_weak["heldout_correct"]
          and w_strong["heldout_correct"])
    print("\nVERDICT:", "PASS — W-only entrenched a fast-but-wrong kernel; H_THEN_W repaired it "
          "(same lever phenomenon as the rotation toy, on real code)." if ok else
          "inconclusive (tune candidates).")
    print("Lever attribution here is REAL: compile-error->H(parser); fast-but-wrong-passes->H_THEN_W; "
          "correct-but-slow->W. No 4-number giveaway trace.")


if __name__ == "__main__":
    main()
