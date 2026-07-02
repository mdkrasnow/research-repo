# Correct and FAST (vectorized torch). The baseline an agent's Triton kernel must beat on GPU.
# out[i,j,c] = sum_k a[i,k,c] b[j,k,c]  ==  per-c matmul a[:,:,c] @ b[:,:,c].T
import torch


def kernel(a, b):
    ac = a.permute(2, 0, 1).contiguous()   # (C, N, K)
    bc = b.permute(2, 0, 1).contiguous()   # (C, N, K)
    out = torch.bmm(ac, bc.transpose(1, 2))  # (C, N, N)
    return out.permute(1, 2, 0).contiguous()  # (N, N, C)
