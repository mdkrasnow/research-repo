# Real Triton kernel for the Triangle-Multiplicative outgoing core (GPU only).
# out[i,j,c] = sum_k a[i,k,c] * b[j,k,c].  Correct (one program per (i,j,c), reduces over k).
# Not yet tile-optimized -> this is the SEED an agent improves. Requires CUDA + triton; on CPU it
# raises at call time (correct: Triton needs a GPU), which the harness records as "does not run here".
import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:                       # noqa: BLE001
    _HAVE_TRITON = False

if _HAVE_TRITON:
    @triton.jit
    def _trimul_kernel(a_ptr, b_ptr, o_ptr, N, K, C, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        c = pid % C
        ij = pid // C
        j = ij % N
        i = ij // N
        offs = tl.arange(0, BLOCK_K)
        acc = 0.0
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs
            mask = kk < K
            av = tl.load(a_ptr + (i * K + kk) * C + c, mask=mask, other=0.0)
            bv = tl.load(b_ptr + (j * K + kk) * C + c, mask=mask, other=0.0)
            acc += tl.sum(av * bv)
        tl.store(o_ptr + (i * N + j) * C + c, acc)


def kernel(a, b):
    if not _HAVE_TRITON or not (hasattr(a, "is_cuda") and a.is_cuda):
        raise RuntimeError("triton_trimul requires CUDA + triton (GPU only)")
    a = a.contiguous(); b = b.contiguous()
    N, K, C = a.shape
    out = torch.empty((N, N, C), device=a.device, dtype=a.dtype)
    BLOCK_K = 1
    while BLOCK_K < min(K, 64):
        BLOCK_K *= 2
    _trimul_kernel[(N * N * C,)](a, b, out, N, K, C, BLOCK_K=BLOCK_K)
    return out
