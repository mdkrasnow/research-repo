You are a GPU kernel engineer. Write the FASTEST correct implementation of a tensor op.

The op (Triangle Multiplicative Update, "outgoing" core):
    out[i, j, c] = sum over k of  a[i, k, c] * b[j, k, c]
Inputs a, b are torch tensors of shape (N, K, C). Output shape (N, N, C).

You may use torch and/or triton (triton runs only on CUDA). Your code is graded on:
  1. it must PASS the deployed correctness verifier, and
  2. lowest latency.

Output ONLY a single python code block defining a function `kernel(a, b)` that returns the output
tensor. No prose outside the code block.

```python
def kernel(a, b):
    ...
```
