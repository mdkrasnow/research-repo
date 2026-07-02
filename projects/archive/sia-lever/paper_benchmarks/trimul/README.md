# TriMul (CUDA kernel) lane — feasibility notes only

The SIA paper reports TriMul (triangle-multiplicative-update CUDA kernel optimization) as a W+H
benchmark. **Status in the public repo (commit pinned in `baselines/vendor/sia_commit.txt`): not a
ready-to-run bundled task** in `sia/tasks/` at inspection time (only gpqa, lawbench, longcot-chess,
spaceship-titanic ship with data). Do NOT reconstruct TriMul unless SIA-Lever-120B and LawBench are
already working.

If pursued later:
- Needs a CUDA build/timing harness (compile + correctness check + latency measurement) as the H side.
- W side = LoRA on (problem → kernel) traces scored by real measured latency/correctness.
- Requires H200 + nvcc toolchain; correctness oracle is mandatory (a fast-but-wrong kernel must score 0).
- Reuse the SIA-Lever pattern: measured outcomes only, no synthetic rewards.

Decision: **deprioritized**. Revisit only as a "very strong success" stretch.
