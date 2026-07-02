# scRNA-seq denoising lane — feasibility notes only

The SIA paper reports single-cell RNA-seq denoising as a W+H benchmark. **Not a ready-to-run bundled
task** in the public repo at the pinned commit (`baselines/vendor/sia_commit.txt`). Do NOT
reconstruct unless SIA-Lever-120B and LawBench are already working.

If pursued later:
- Needs a denoising dataset + a held-out reconstruction/correlation metric as the H-side evaluator.
- W side = LoRA/training of the task model on measured-score traces.
- Watch coupled Goodhart: a denoiser can over-smooth to game a correlation metric — needs a
  structural control (like our negative control) so the evaluator can't be gamed.

Decision: **deprioritized**. Revisit only as a "very strong success" stretch.
