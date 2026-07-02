# Correct but SLOW: explicit loops. Always passes strong verifier; loses on latency.
import torch


def kernel(a, b):
    N, K, C = a.shape
    out = torch.zeros(N, b.shape[0], C, device=a.device, dtype=a.dtype)
    for i in range(N):
        for j in range(b.shape[0]):
            out[i, j] = (a[i] * b[j]).sum(0)
    return out
