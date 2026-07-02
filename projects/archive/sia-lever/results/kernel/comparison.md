# Kernel-task lever comparison (stub, device=cpu, shape=(32, 32, 8))

Op: Triangle Multiplicative Update outgoing core `out[i,j,c]=sum_k a[i,k,c]b[j,k,c]` (SIA paper's GPU-kernel task). Final quality judged under the STRONG verifier (random inputs, tight tol). Baseline (torch_bmm) latency = 0.06452831992646679 ms.

| policy | final correct | passes strong | speedup vs baseline | levers |
|---|---|---|---|---|
| W_only | False | False | 65.85 | W → W → W → W |
| H_only | False | False | 74.89 | H → PROMOTE |
| H_THEN_W | True | True | 0.82 | H_THEN_W → PROMOTE |
| selector | True | True | 0.87 | H_THEN_W → PROMOTE |

Reading: W-only entrenches a fast-but-WRONG kernel (high 'speedup', correct=False) under the weak verifier; H-only exposes it but cannot repair (no retrain); H_THEN_W and the selector reach a CORRECT kernel. The selector picks H_THEN_W from the implausible-speedup shortcut signature — no held-out labels.
