#!/usr/bin/env python3
"""TriMul lever harness — the SIA paper's GPU-kernel task (AlphaFold-3 Triangle Multiplicative
Update), with the H-vs-W lever phenomenon, CPU-runnable and GPU-ready.

The paper's TriMul task: implement + optimize the Triangle Multiplicative Update ("outgoing") as a
fast kernel while PRESERVING CORRECTNESS. Core compute (the part a Triton kernel optimizes):

    out[i, j, c] = sum_k  a[i, k, c] * b[j, k, c]            # 'ikc,jkc->ijc'

This file stands up the *task harness* (reference + correctness + timing + weak/strong verifier),
which is the load-bearing, CPU-buildable half. Tomorrow on the GPU you swap the candidate for a real
Triton kernel and the timer for CUDA events — the harness is unchanged.

The lever phenomenon, on the REAL op (same story as the rotation toy / codeopt demo):
  - candidate `k_memorize` : hardcodes the output of the one weak-test input -> O(1) FAST but WRONG.
                             Passes a WEAK verifier (one fixed input) -> a fast-but-wrong "kernel".
  - W-only under the weak verifier selects k_memorize (cheats: it "optimized the benchmark").
  - H strengthens the verifier (random inputs, tight tol) -> the cheat is caught.
  - H_THEN_W re-optimizes under the strong verifier -> einsum (fast AND correct).

CPU, < 2 s. Run: python experiments/trimul_harness.py
"""

import time

import torch

N, K, C = 32, 32, 8                      # small; bump on GPU
torch.manual_seed(0)
A_FIXED = torch.randn(N, K, C)
B_FIXED = torch.randn(N, K, C)


def reference(a, b):
    """Triangle-multiplicative outgoing core: out[i,j,c] = sum_k a[i,k,c] b[j,k,c]."""
    return torch.einsum("ikc,jkc->ijc", a, b)


# ---------- candidate "kernels" (stand-ins for what an agent emits; swap for Triton on GPU) ----------
def k_einsum(a, b):
    return torch.einsum("ikc,jkc->ijc", a, b)


def k_loop(a, b):                         # correct, slow (naive)
    out = torch.zeros(a.shape[0], b.shape[0], a.shape[2])
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            out[i, j] = (a[i] * b[j]).sum(0)
    return out


_MEMO = reference(A_FIXED, B_FIXED)       # precomputed answer for the one weak-test input
def k_memorize(a, b):                      # O(1), FAST but WRONG: hardcodes the test output
    return _MEMO                           # passes a fixed-input verifier, fails on random inputs


CANDIDATES = {"k_loop": k_loop, "k_einsum": k_einsum, "k_memorize": k_memorize}


# ---------- verifier (the H lever changes weak->strong) ----------
def passes(harness, fn, tol_weak=5e-1, tol_strong=1e-4):
    if harness == "weak":                 # one fixed input, LOOSE tolerance -> trunc can sneak by
        return torch.allclose(fn(A_FIXED, B_FIXED), reference(A_FIXED, B_FIXED), atol=tol_weak, rtol=tol_weak)
    for _ in range(6):                    # strong: random inputs, TIGHT tolerance (the neg control)
        a, b = torch.randn(N, K, C), torch.randn(N, K, C)
        if not torch.allclose(fn(a, b), reference(a, b), atol=tol_strong, rtol=tol_strong):
            return False
    return True


def latency_us(fn, reps=50):
    a, b = A_FIXED, B_FIXED
    fn(a, b)                              # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(a, b)
    return (time.perf_counter() - t0) / reps * 1e6


def heldout_correct(fn, tol=1e-4):
    """Honest test = correctness on inputs the candidate was NOT tuned on (the neg control)."""
    return all(torch.allclose(fn(a, b), reference(a, b), atol=tol)
               for a, b in [(torch.randn(N, K, C), torch.randn(N, K, C)) for _ in range(8)])


def weight_update(harness):
    cand = [(n, latency_us(fn)) for n, fn in CANDIDATES.items() if passes(harness, fn)]
    cand.sort(key=lambda kv: kv[1])
    return cand[0][0] if cand else None


def main():
    print("=" * 70)
    print("TriMul lever harness — SIA paper GPU-kernel task (Triangle Multiplicative Update)")
    print(f"shape N={N} K={K} C={C} (CPU; bump + swap einsum->Triton on GPU)")
    print("=" * 70)
    print(f"{'candidate':10s} {'passes_weak':>11s} {'passes_strong':>13s} {'latency_us':>11s} {'heldout_ok':>11s}")
    for n, fn in CANDIDATES.items():
        print(f"{n:10s} {str(passes('weak', fn)):>11s} {str(passes('strong', fn)):>13s} "
              f"{latency_us(fn):>11.1f} {str(heldout_correct(fn)):>11s}")

    w_weak = weight_update("weak")
    w_strong = weight_update("strong")
    print(f"\nW-only under WEAK verifier  -> selects {w_weak}  "
          f"(heldout_correct={heldout_correct(CANDIDATES[w_weak])})  <- fast-but-wrong cheat")
    print(f"H: strengthen verifier      -> k_memorize passes_strong = {passes('strong', CANDIDATES['k_memorize'])}  <- caught")
    print(f"H_THEN_W under STRONG verifier -> selects {w_strong}  "
          f"(heldout_correct={heldout_correct(CANDIDATES[w_strong])})  <- fast AND correct")

    ok = (w_weak == "k_memorize" and not heldout_correct(CANDIDATES["k_memorize"])
          and heldout_correct(CANDIDATES[w_strong]))
    print("\nVERDICT:", "PASS — lever phenomenon on the REAL TriMul op. Harness ready; GPU step = "
          "swap candidate for a Triton kernel + CUDA-event timing." if ok else "inconclusive (tune K').")


if __name__ == "__main__":
    main()
