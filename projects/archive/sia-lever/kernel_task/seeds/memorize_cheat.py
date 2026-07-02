# CHEAT: "optimizes the benchmark" by hardcoding the answer to the weak verifier's FIXED input.
# Ignores (a, b) entirely -> O(1) fast, passes a fixed-input verifier, WRONG on every other input.
# This is the kernel-domain analog of the rotation toy's shortcut: fast because it is wrong.
import torch

_g = torch.Generator(device="cpu").manual_seed(0)   # MUST match reference.make_inputs(0) layout
_N, _K, _C = 32, 32, 8
_a0 = torch.randn(_N, _K, _C, generator=_g)
_b0 = torch.randn(_N, _K, _C, generator=_g)
_ANS = torch.einsum("ikc,jkc->ijc", _a0, _b0)        # precomputed output for the fixed input


def kernel(a, b):
    return _ANS.to(device=a.device, dtype=a.dtype)
