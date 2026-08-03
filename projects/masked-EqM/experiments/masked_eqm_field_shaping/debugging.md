# Debugging ledger

## 2026-08-03 — smoke fingerprint failure

Jobs 36928994 (control) and 36929029 (masked) failed before model construction and before any optimizer update. `nested_state_sha256` converted each tensor to a `uint8` view, but PyTorch does not permit changing the element size of a zero-dimensional tensor; AdamW stores `step` as such a scalar. The fix hashes the contiguous NumPy byte representation directly, which works for scalar and nonscalar tensors. A scalar optimizer-state regression test was added. No scientific field, checkpoint, data input, or training result changed. Both paired smokes must restart from the original epoch-15 base checkpoint.
