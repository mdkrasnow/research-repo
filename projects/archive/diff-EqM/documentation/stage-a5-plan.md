# Stage A.5 Execution Plan — Path B → A

Chosen: **B then A**. Cheap smoke test to confirm architecture is the dominant factor, then full port of Flow Matching's CIFAR U-Net with the EqM loss grafted on.

## Step B: DDPM-UNet smoke test (1–2 days)

**Purpose**: confirm that with a correct architecture + the SAME data and eval pipelines we already use, CIFAR-10 FID drops from ~497 to a reasonable range (single-digit FID). If this passes, we know the bug is architecture-only and Step A is justified. If it fails, we have a deeper data/eval problem to chase first.

**What to build**
1. Pick a reference implementation. Two viable options:
   - **`diffusers` UNet2DModel** with Ho et al. (2020) DDPM-style defaults. Easiest; well-tested; supports CIFAR out of the box.
   - **Minimal self-contained DDPM UNet** (e.g., `lucidrains/denoising-diffusion-pytorch`). More transparent; smaller dependency footprint.

   **Recommend `diffusers` UNet2DModel** — the goal is diagnostic, not architectural novelty.

2. Training loop: simplest possible flow matching or DDPM objective on CIFAR-10. Use the exact same data pipeline (dataset class, transforms, normalization) we already use in `train_dganm.py` so the data pipeline is the controlled variable.

3. Eval: use the exact same FID pipeline we already use (`evaluate_fid.py`). Same reference stats, same sample count. If our eval is broken, this test will also fail — which is useful signal.

4. Training budget: target ~100–200 epochs, batch size 128, Adam lr=2e-4. On one A100 this is a few hours. Do not optimize hyperparams — we just need a sanity check.

**Exit criteria**
| Outcome | Interpretation | Next step |
|---|---|---|
| FID < 15 | Architecture is the dominant factor. Our data + eval are fine. | Proceed to Step A with confidence. |
| FID 15–50 | Architecture matters a lot but something else (sampler, reference stats, data aug) also off. | Investigate those before Step A. |
| FID > 50 | Data pipeline or eval is also broken. | Debug data + eval before doing anything else. |

**Output**: `documentation/stage-a5-smoke-results.md` with FID, samples, and verdict. Commit message includes the FID.

## Step A: Port Flow Matching CIFAR UNet + EqM loss (~1 week)

**Purpose**: reproduce vanilla EqM's FID 3.36 on CIFAR-10, certifying our stack matches the paper. Then train DG-ANM on the same stack for the Stage C secondary result.

**What to build**
1. **Vendor the Flow Matching CIFAR code**: clone `facebookresearch/flow_matching` into `projects/diff-EqM/fm-upstream/`. Identify:
   - UNet architecture file (likely `unet.py` or similar in their image-experiment dir)
   - CIFAR-10 training entry point
   - Exact hyperparameters (channels, layers, lr, batch size, optimizer, ema, epochs)
   - Sampler config for evaluation
2. **Minimal fork**: create `projects/diff-EqM/experiments/train_cifar_unet.py` that wraps FM's UNet but swaps the loss:
   - FM target: `E[‖v_θ(x_t, t) − (x_1 − x_0)‖²]`
   - EqM target: learn the equilibrium gradient with the paper's `c(γ)` weighting (truncated, a=0.8) — same formulation already in our `train_dganm.py`, just against a UNet forward pass instead of an EqM-S/2 forward pass.
3. **Sampler**: use the EqM paper's CIFAR sampler settings — `a`, `b` from ImageNet, search over `λ`, step size 1. Start with the upstream's `sample_gd.py` adapted to the UNet model.
4. **Train vanilla EqM first**. This is the gate. Target FID 3.36 ±0.3 on 50K samples.
5. **Once vanilla matches**: train DG-ANM version with the best-so-far hyperparameters from the Stage A proxy sweep. This becomes the Stage C secondary result.

**Exit criterion for Stage A.5 overall**
- Vanilla EqM on CIFAR-10 via UNet: **FID ≤ 3.66** (paper's 3.36 + 0.3 tolerance for seed variance), 50K samples, paper-matching sampler.
- If achieved: Stage A.5 passes. Stage C secondary (CIFAR DG-ANM 3-seed) is unblocked.
- If not achieved within 2 weeks of effort: escalate. Options: contact authors, inspect their exact config from a checkpoint if available, or deprioritize CIFAR (fall back to Option C from the audit).

## Parallel-track consideration

Because the user confirmed compute is NOT tight, we should use wall-clock efficiently:
- **Step B smoke test**: blocks only on one person-day of coding + one GPU-day. Start immediately.
- **Stage B ImageNet-256 vanilla baseline**: long-running, multi-day GPU job, uses our known-working transformer stack. Independent of Stage A.5. **Can start right now.** This was item (1) from my earlier "things we can do while waiting" list — now it matters more because it's on the critical path for Stage B.

Proposal: start Step B smoke test AND Stage B vanilla IN-256 baseline training in parallel. Step A begins once Step B passes its exit criterion.

## Risks specific to this plan

| Risk | Mitigation |
|---|---|
| FM's CIFAR UNet config is undocumented / hard to reproduce | Read their repo carefully; also inspect their released CIFAR checkpoint if available |
| EqM loss on a UNet has subtle issues (output shape, time conditioning) not present in the SiT version | Step B smoke test first — if UNet + FM loss works, grafting EqM loss is a small step |
| Our FID eval disagrees with paper's (different reference stats or evaluator) | Step B will expose this: if UNet + plain FM hits ~4 FID, our eval is paper-comparable |
| Paper's numbers assume specific sampler tuning we don't replicate | Report paper-matching sampler AND our default sampler; document discrepancies |
