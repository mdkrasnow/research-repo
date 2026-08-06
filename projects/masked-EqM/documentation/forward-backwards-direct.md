# `forward-backwards-direct` — semigradient scalar-energy training

## Status

Implemented per an explicit spec covering audit, implementation, tests, and
a local smoke run (this document + `fb_direct/`). Not yet run through this
project's normal Variant Proposal Template / CIFAR-first gating process
(`AGENTS.md`) — the user explicitly authorized skipping that gate for this
one variant. The proposal content below (§1) is included for the record.

## 1. Motivation / variant proposal

```
Variant name: forward-backwards-direct
Hypothesis: ebm='direct' trains a scalar E_theta(z) by backpropagating
  THROUGH torch.autograd.grad(E.sum(), z, create_graph=True), i.e. through
  the input-gradient operation itself. This introduces a mixed second
  derivative d^2E/(dz dtheta) and empirically produces late-training
  backbone gradient explosions (motivating train.py's --max-grad-norm and
  the head/backbone gradient-norm split already logged in
  gradient_metrics.jsonl -- infrastructure that predates this variant and
  was built to diagnose exactly this failure mode). Replacing the
  double-backward with an explicit, hand-derived reverse network trained by
  ordinary first-order backprop should eliminate the d^2E/(dz dtheta) term
  from the optimizer's parameter-gradient entirely, while remaining exact
  (up to floating point) at every synchronization point.
Failure mode addressed: backbone gradient explosion from double-backward
  through attention/LayerNorm/GELU in ebm='direct'.
EqM compatibility argument: the field target is UNCHANGED (same
  transport.py c(gamma), same path_sampler.plan, same sign convention --
  see the table in Section 3 below). Only the MECHANISM that produces the
  trained field from the trained energy changes: theta is not updated by
  ordinary backprop through the true computational graph at all; it is
  updated by a mapped ("semigradient") step through a separately
  parameterized reverse network phi that is exact at every sync point.
Loss definition: L = mean((-R_phi(sg[C_theta(z)]) - ut)^2), ut = existing
  repo target (x1-x0)*c(t). See Section 3.
Expected diagnostics if working: fb/loss decreases (or stays flat-then-
  decreases) over training; fb/phi_theta_sync_error stays ~1e-7 (fp32)
  every step; fb/field_target_cosine trends toward positive as theta
  learns; backbone gradient norms (now: theta_update_norm) do not exhibit
  the late-training blowup documented for ebm='direct'.
Expected diagnostics if failing: fb/audit_mean_rel_error (periodic exact
  audit) drifts upward over training (would indicate the semigradient
  omission is not benign at that point in training); loss diverges; NaNs.
Minimal test: tests/test_forward_backwards_direct.py
  test_end_to_end_field_equivalence_fp64/fp32.
Promotion rule: passes the SAME Phase 1 IN-1K gate as any other diff-EqM /
  masked-EqM variant (see AGENTS.md's gating table) -- FID improvement or
  parity vs the ebm='direct' baseline AND no backbone gradient explosion
  observed over a full-length run.
Kill rule: if the semigradient (mapped Pi-dagger(delta_phi) update)
  produces worse FID than ebm='direct' at matched compute, or if
  fb/phi_theta_sync_error or fb/audit_mean_rel_error grows unboundedly
  over training (would mean the identity Pi mapping stops being a good
  local model of the true theta-gradient), kill and reconsider a non-
  identity Pi (e.g. a learned or partially-recomputed mapping) before
  trying another training-time hack.
```

## 2. What changes vs `direct`

| | `direct` | `forward-backwards-direct` |
|---|---|---|
| Forward E_theta(z) | `torch.autograd.grad(E.sum(), z, create_graph=train)` | Identical numerically (`forward_energy_with_cache`), but **always** under `torch.no_grad()` during training |
| theta's training gradient | `d/dtheta` of a loss that backprops through the `autograd.grad` call above (`create_graph=True`) — the exact but graph-heavy path | **None, ever.** theta is updated by `theta += Pi_dagger(delta_phi)`, an explicit tensor add outside autograd |
| What `loss.backward()` differentiates | The full double-backward graph (transformer forward + input-gradient + transformer-again) | Only `R_phi`, an ordinary ~40-line forward tensor program |
| Sampling / inference | `autograd.grad(..., create_graph=False)` | **Identical, byte-for-byte** — same `EqM.forward`/`forward_with_cfg` code path, same architecture |
| Checkpoint compatibility | n/a | A `direct` checkpoint loads directly into `forward-backwards-direct`'s theta (identical architecture) |

## 3. Sign / convention table (audited from the repo before writing any code)

| Quantity | Where | Value / convention |
|---|---|---|
| Data endpoint | `transport.py Transport.sample` | `x1` = clean data (VAE latent) |
| Corruption endpoint | `transport.py Transport.sample` | `x0` (gaussian by default: `randn_like(x1)`) |
| Interpolant | `path.py ICPlan.compute_xt` | `xt = t*x1 + (1-t)*x0` (this is the model's input, confusingly named `x0` as EqM's own `forward()` argument — see `models.py:269`) |
| Raw target velocity | `path.py ICPlan.compute_ut` | `ut = d_alpha_t*x1 + d_sigma_t*x0 = x1 - x0` |
| Energy-compatible target | `transport.py training_losses:213` | `ut *= get_ct(t)` — this **is** the repo's `c(gamma)` |
| Scalar energy | `models.py ScalarEnergyHead.forward` | `E(z) = sum_tokens Linear(modulate(LN(x_L), shift, scale))`, `z` = the model's `x0` argument |
| Model output (`field`) for `ebm='direct'` | `models.py:296-300` | `field = -grad_z E(z)` |
| Loss (VELOCITY model type, WeightType.NONE — the repo default) | `transport.py:227-228` | `loss = mean_flat((field - ut)**2)` |
| Net implication | — | training pushes `-grad_z E(z) -> (x1-x0)*c(t)`, i.e. `grad_z E(z) -> (x0-x1)*c(t) = (noise - data)*c(t)` — matches the spec's `y_gamma = (eps - x)*c(gamma)` exactly with `x=x1`(data), `eps=x0`(noise) |
| Sampler direction | `models.py` docstring; verified by `tests/test_scalar_ebm.py::test_field_is_neg_grad_energy` | `x <- x + eta*field` is **gradient descent on E** (since `field = -grad E`) |
| `forward-backwards-direct`'s `R_phi(cache)` output | `fb_direct/reverse_model.py` | `grad_z E(z)` directly (**no** sign flip anywhere in `reverse_ops.py` — the `u=ones` seed at the energy head *is* `dE/d(token_energies)`, propagated with no negation) |
| `forward-backwards-direct`'s training `prediction` | `fb_direct/trainer.py::training_step` | `prediction = -R_phi(cache)`, to match `field = -grad E` above before comparing to `ut` |

**This sign flip (`prediction = -grad_tilde`) was caught by the sign test**
(`test_sign_convention`) during development — an earlier draft compared
`R_phi(cache)` directly against `ut` and failed with `rel diff == 2.0`
(the textbook signature of a sign error), which is exactly why Section 14.5
of the spec requires this test.

## 4. theta / phi / Pi / Pi-dagger

- **theta**: the ordinary `EqM` model (`ebm='forward-backwards-direct'`,
  architecturally identical to `ebm='direct'`).
- **phi**: `fb_direct.ReverseEqM(theta)` — an `nn.Module` with its own
  trainable weight tensors, one per `reverse_active` / `recomputed_conditioning`
  forward weight tensor (see coverage table below), each **the same shape
  and orientation** as its theta counterpart.
- **Pi** (`tie_from_forward_`): for every mapped tensor, `phi.data.copy_(theta.data)`
  — plain **identity** copy. (A `mapping_type="transpose"` code path also
  exists in `param_mapping.py` for generality / unit coverage, but nothing
  in this architecture currently needs it — see "why identity" below.)
- **Pi-dagger** (`apply_backward_delta_to_forward_`): `theta.data += (phi_after - phi_before)`
  — the same identity map run in reverse on the parameter *delta*.

**Why every mapping is identity, not transpose:** the manual VJPs in
`reverse_ops.py` were deliberately derived as `u_y @ W` (reusing the
forward `nn.Linear.weight` layout `(out_features, in_features)` directly),
rather than the more common-looking `u_y @ W^T`. This is mathematically
equivalent (`u_x = u_y @ W` is exactly `dL/dx` for `y = xW^T`), and it
means Pi/Pi-dagger never need a transpose. The patch-embed conv reverse
(`F.conv_transpose2d`) similarly reuses the *forward* `Conv2d.weight` shape
`(D, C, p, p)` directly, since `conv_transpose2d`'s weight convention is
`(in_channels, out_channels, kH, kW) = (D, C, kH, kW)` — the same tensor,
no transpose.

## 5. Parameter coverage (measured, EqM-S/2: depth=12, hidden=384, heads=6)

```
total trainable parameters:        32,472,961
reverse_active:      21,240,192  (65.41%)  -- attn.qkv/proj, mlp.fc1/fc2,
                                              x_embedder.proj, energy_head.linear weights
recomputed_conditioning: 10,940,160 (33.69%)  -- block adaLN_modulation.1
                                              and energy_head.adaLN_modulation.1
                                              (weight AND bias; see caveat below)
cache_only:              292,609  ( 0.90%)  -- every Linear BIAS in the
                                              reverse-active path, plus ALL of
                                              t_embedder.*, y_embedder.*,
                                              x_embedder.proj.bias,
                                              energy_head.linear.bias
unused:                    1,536  ( 0.00%)  -- pos_embed (already
                                              requires_grad=False in theta)
------------------------------------------------------------------
reverse coverage (reverse_active + recomputed_conditioning): 99.10%
```

**Caveat on `recomputed_conditioning` bias rows:** each block's
`adaLN_modulation.1` Linear produces 6 chunks
(`shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`); only
`scale_*`/`gate_*` appear in any manual VJP formula (`shift` is purely
additive in `modulate()`, so `d(x_mod)/d(shift) = 0` identically). The
`shift`-chunk **rows** of that Linear's weight and bias are therefore
structurally guaranteed to receive **zero** gradient even though the whole
tensor is classified `recomputed_conditioning` (not `cache_only`) — they
*are* part of the live reverse computation graph (recomputed from cached
`c` every step), they just multiply into a term that never reaches the
loss. `energy_head.adaLN_modulation.1`'s `shift` chunk (half its 2-chunk
output) has the identical caveat. This is intentional and documented here
rather than silently over-claiming 100% "active" coverage.

**`cache_only` parameters are frozen** (`requires_grad_(False)`) by
`ParameterMappingRegistry.freeze_cache_only_()`, called automatically in
`ForwardBackwardsDirectTrainer.__init__`. Full list of names: see
`fb_direct/param_mapping.py::build_default_mapping` (55 tensors — every
Linear/Conv bias in the reverse-active path, `t_embedder.*`,
`y_embedder.embedding_table.weight`).

## 6. Manual VJPs implemented (`fb_direct/reverse_ops.py`)

- `linear_reverse` — `y = xW^T (+ b)`, bias excluded (zero local Jacobian wrt x)
- `gelu_tanh_prime` / SiLU `silu_prime` — exact derivatives matching
  `nn.GELU(approximate="tanh")` and the `SiLU` used in `TimestepEmbedder`/adaLN
- `layernorm_noaffine_reverse` — exact `nn.LayerNorm(elementwise_affine=False)` backward
- `modulate_reverse_wrt_norm` — AdaLN's `x*(1+scale)+shift`, wrt the LN output only
- `attention_reverse` — full unfused/math-backend-equivalent multi-head
  self-attention reverse (qkv → scaled dot-product → softmax → weighted
  sum → proj), matching `timm.models.vision_transformer.Attention`'s
  non-fused path (the mathematically-forced path during EqM training,
  since `train.py` disables flash/mem-efficient/cudnn SDP whenever
  `ebm != 'none'`)
- `mlp_reverse` — `fc1 → GELU(tanh) → fc2` (timm `Mlp`, `drop=0`)
- `patch_embed_reverse` — non-overlapping `Conv2d(kernel=stride=patch)` via
  `F.conv_transpose2d`, exact because stride==kernel_size (no overlap)

All seven have dedicated FP64 unit tests against `torch.autograd` in
`tests/test_forward_backwards_direct.py` (§14.1 below).

## 7. Files added / modified

**Added** (`projects/masked-EqM/fb_direct/`):
- `__init__.py` — public exports
- `cache.py` — `TransformerReverseCache`, `SiTBlockReverseCache`,
  `AttentionReverseCache`, `MLPReverseCache`, `LayerNormReverseCache`,
  `FinalLayerReverseCache` dataclasses
- `reverse_ops.py` — the manual VJP primitives (§6)
- `forward_cache.py` — `forward_energy_with_cache(model, z, t, y)`:
  runs theta under `torch.no_grad()`, returns `(E, cache)`, numerically
  identical to `ebm='direct'` energy
- `reverse_model.py` — `ReverseEqM(forward_model)`, `ReverseSiTBlock`: the
  explicit phi network
- `param_mapping.py` — `ParameterMappingRegistry`,
  `build_default_mapping`, `MappingEntry`: Pi/Pi-dagger + coverage report
- `trainer.py` — `ForwardBackwardsDirectTrainer`: owns theta/phi/registry/
  optimizer, implements the reference algorithm, diagnostics, exact-field
  audit, checkpoint save/load

**Added** (elsewhere in `projects/masked-EqM/`):
- `tests/test_forward_backwards_direct.py` — full test suite (§9 below)
- `experiments/direct_energy/fb_direct_smoke.py` — smoke-training script
- `documentation/forward-backwards-direct.md` — this file

**Modified**:
- `models.py` — every `if self.ebm == 'direct':` branch that selects the
  scalar-energy architecture (constructor, weight init, `forward`,
  `forward_with_cfg`) now matches `self.ebm in ('direct', 'forward-backwards-direct')`;
  a new guard raises `RuntimeError` if `forward(..., train=True)` is called
  directly on an fb-direct model (which would silently run the exact
  `create_graph=True` double-backward path this mode exists to avoid — use
  `ForwardBackwardsDirectTrainer.training_step` instead). **No other line
  in `models.py` changed** — `none`/`l2`/`dot`/`mean`/`direct` are
  byte-identical in behavior (see `tests/test_forward_backwards_direct.py::test_existing_modes_regression`
  and the pre-existing `tests/test_scalar_ebm.py` / `test_energy_only.py`,
  all still passing).
- `train.py` — added `from fb_direct import ForwardBackwardsDirectTrainer`;
  added `main_forward_backwards_direct(args)` (a separate function, not
  threaded into the existing `main()`, to guarantee the existing path is
  untouched); added `--fb-exact-audit-every` / `--allow-cpu` CLI flags;
  dispatch on `args.ebm == 'forward-backwards-direct'` at the bottom of
  `__main__`.
- `--ebm` choices lists extended with `"forward-backwards-direct"` in:
  `sample_gd.py`, `eval_fid.py`, `eval_energy_ordering.py`,
  `eval_blur_recovery.py`, `eval_generalization.py`,
  `eval_masked_recovery.py`, `eval_downsample_recovery.py`,
  `eval_fourier_recovery.py`, `eval_ood_energy.py`,
  `experiments/direct_energy/off_trajectory.py`,
  `experiments/direct_energy/pilot_train.py`,
  `experiments/direct_energy/sample_probe.py`. These scripts only ever
  call `model.forward`/`forward_with_cfg` (never `train=True`), so once
  `models.py` treats the new ebm value like `'direct'` for architecture and
  inference, they work unmodified.

## 8. Checkpoint format

`ForwardBackwardsDirectTrainer.state_dict()`:
```python
{
  "theta": <EqM.state_dict()>,
  "phi": <ReverseEqM.state_dict()>,
  "phi_optimizer": <AdamW.state_dict()>,
  "step_count": int,
  "mapping_version": "fb-direct-v1-identity",
  "frozen_cache_only_names": [...],
}
```
`train.py`'s checkpoint files additionally include `"ema"`, `"args"`,
`"step"`, `"epoch"` (matching the existing checkpoint shape for other ebm
modes, so `download.find_model`/`eval_*.py`'s existing
`state_dict["model"]`/`state_dict["ema"]` loading keeps working unchanged
— they read `state["theta"]`... actually they read `state["model"]`; note
below).

**Loading:**
- A checkpoint from THIS mode (`"phi"` key present): full state restored,
  `mapping_version` validated (refuses to resume on a mismatched Pi
  convention), phi optimizer moments restored, `phi` re-tied from `theta`
  after load, and a hard assertion (`sync_error < 1e-5`) that the restore
  is self-consistent.
- A plain `ebm='direct'` checkpoint (`"model"` key, `"phi"` absent):
  `theta` loads directly (identical architecture), `phi` is freshly
  initialized as `Pi(theta)`, and the phi optimizer starts from scratch —
  exactly per Section 12 of the spec.

Note: `train.py`'s `main_forward_backwards_direct` checkpoints save
`fb_trainer.state_dict()` (keys `theta`/`phi`/...) rather than the
`"model"`/`"ema"`/`"opt"` keys the other ebm modes use, so any downstream
script that loads a raw checkpoint dict by those exact key names (rather
than through `find_model`, which already special-cases the "model" key)
needs `state["theta"]` instead of `state["model"]` for `forward-backwards-direct`
checkpoints. `main_forward_backwards_direct`'s own `--ckpt` loader (see
`train.py`) already detects this via `"phi" in raw or "theta" in raw`.

## 9. Tests (`tests/test_forward_backwards_direct.py`)

Run: `python tests/test_forward_backwards_direct.py` (CPU, ~1 minute)

| Test | Result |
|---|---|
| `test_local_vjp_linear/gelu_tanh/silu/layernorm_noaffine/attention/mlp/patch_embed` | PASS, all `rtol<=1e-7` FP64 vs autograd |
| `test_forward_cache_matches_direct_energy` | PASS, exact (`rtol=1e-10`) match to `ebm='direct'` energy |
| `test_end_to_end_field_equivalence_fp64` | PASS — mean rel err **7.6e-16**, min cosine **1.000000000** (required `<1e-6` / `>0.999999`) |
| `test_end_to_end_field_equivalence_fp32` | PASS — mean rel err **3.9e-7** |
| `test_no_double_backward` | PASS — `torch.autograd.grad` monkeypatched/spied: **zero calls** during `training_step`; every `theta` param's `.grad is None`; `phi` received nonzero gradients |
| `test_update_transfer_identity_mapping` | PASS — exact (`rtol=1e-10`) round-trip of a synthetic phi gradient into theta via Pi-dagger |
| `test_sign_convention` | PASS — `R_phi(cache) == grad_x E` exactly (`rtol=1e-6`); `field := -R_phi(cache)`; one small `x <- x + eta*field` step strictly decreases `E` for every sample |
| `test_batch_independence` | PASS — sample 0's field bit-identical whether run alone or with 3 unrelated batchmates |
| `test_checkpoint_resume` | PASS — theta exact match, phi resynced, optimizer moments restored, step count preserved |
| `test_checkpoint_init_from_direct` | PASS — a plain `direct` checkpoint loads cleanly into an fb-direct model's theta |
| `test_existing_modes_regression` | PASS — `none`/`dot`/`l2`/`direct` forward+backward unaffected |

Pre-existing regression suites also run clean after these changes:
`tests/test_scalar_ebm.py` (all 6 checks), `tests/test_energy_only.py`
(direct + dot energy-only equivalence). `tests/test_direct_eval_samplers.py`
and `tests/test_sampling_defaults.py` fail in this local environment for
reasons **unrelated to this change** (`ModuleNotFoundError: diffusers` —
not installed locally; a `sys.path`-relative import issue in the test file
itself, reproduced on a completely clean checkout) — not touched by this
patch.

## 10. Smoke run

Run: `python experiments/direct_energy/fb_direct_smoke.py` (CPU, ~25s for 40 steps)

Confirmed live in this environment:
1. Scalar forward model + reverse network + registry construct; **99.10%**
   reverse coverage.
2. `phi` ties to `theta` at construction with **exact** (`0.0`) sync error.
3. 40 `training_step()` calls complete, loss finite throughout
   (`25.89 -> 20.75`, non-monotone because each step draws an i.i.d.
   random synthetic-latent batch rather than a real dataset — see the
   script's docstring for why real ImageNet/VAE data wasn't available
   locally), **`theta.grad is None` checked and confirmed every single
   step**, sync error `0.00e+00` after every step's `Pi-dagger` + re-tie.
4. Periodic exact-field audit: mean relative error **1.10e-7**, cosine
   **1.0** (FP32).
5. Checkpoint saved (516 MB, EqM-S/2 theta+phi+optimizer state).
6. Checkpoint resumed into a **fresh** trainer/model: theta matches
   exactly (max diff `0.0`), phi re-synced (`0.00e+00`), step count and
   76 optimizer-state tensors restored.
7. Sampling from `theta` via the **true** scalar-energy forward pass
   (`autograd.grad(..., create_graph=False)`, not the reverse network —
   per spec invariant #9/#10) — 5 Euler steps, finite output.

**What real IN-1K cluster training looks like** (not run in this
environment — SLURM here is remote-only per `AGENTS.md`, and this repo's
gating discipline requires a passed CIFAR-scale/Phase-0-equivalent sanity
gate before any IN-1K compute is spent; the user explicitly waived that
gate for building this feature, not for spending IN-1K compute on it):

```bash
# From scratch:
scripts/cluster/remote_submit.sh ... train.py \
  --data-path <imagenet-latents-or-images> --model EqM-S/2 --ebm forward-backwards-direct \
  --global-batch-size 256 --max-steps 5000 --ckpt-every 1000 --fb-exact-audit-every 200

# Continuing from an existing 'direct' checkpoint (Section 12):
scripts/cluster/remote_submit.sh ... train.py \
  --data-path <...> --model EqM-B/2 --ebm forward-backwards-direct \
  --ckpt results/000-EqM-B-2-.../checkpoints/0050000.pt \
  --global-batch-size 256 --epochs 80

# Sampling / FID (identical to 'direct' -- theta is the only artifact needed):
python sample_gd.py --ebm forward-backwards-direct --ckpt <fb-direct-ckpt-or-direct-ckpt> ...
python eval_fid.py  --ebm forward-backwards-direct --ckpt <...> ...
```

## 11. Diagnostics logged

Every `training_step()` returns a dict written by `train.py` to
`fb_direct_metrics.jsonl` (one line/step, analogous to the existing
`gradient_metrics.jsonl`):
`fb/loss`, `fb/raw_phi_grad_norm`, `fb/clipped_phi_grad_norm`,
`fb/theta_update_norm`, `fb/update_to_weight_ratio`,
`fb/phi_theta_sync_error_pre`, `fb/phi_theta_sync_error_post`,
`fb/field_norm`, `fb/target_norm`, `fb/field_target_cosine`,
`fb/energy_mean`; periodic (`--fb-exact-audit-every`) additions:
`fb/audit_mean_rel_error`, `fb/audit_max_abs_error`, `fb/audit_mean_cosine`.

## 12. Known limitations

1. **Distributed (DDP) training is implemented but not live-tested.**
   `train.py::main_forward_backwards_direct` wraps only `phi` in DDP (never
   `theta`, which never participates in autograd in this mode) when
   `world_size > 1`, relying on DDP's gradient all-reduce over `phi` and a
   subsequently-identical deterministic `Pi-dagger` update on every rank to
   keep `theta` synchronized without wrapping it directly. This is the
   correct design per Section 9 of the spec, but this local environment has
   no multi-GPU/multi-process setup to run it against, and SLURM is
   remote-cluster-only per `AGENTS.md` — genuinely not exercisable here.
   **Do not treat this as verified until a real 2-process run confirms it.**
2. **Mixed precision (AMP/bf16) is not implemented for this mode.** The
   spec allows disabling it with a documented reason (Section 10); reason:
   out of scope for a from-scratch feature whose primary claim is
   eliminating a specific double-backward-driven instability — adding AMP
   on top before establishing FP32 correctness would conflate two variables.
   `forward_energy_with_cache` and `ReverseEqM` are dtype-agnostic (FP64
   tests pass at `rtol<=1e-7`), so AMP support is a follow-up, not a
   redesign.
3. **`torch.compile` not attempted.** Same reasoning as (2); the reverse
   network's dataclass-based cache and per-block Python loop are compile-
   *able* in principle but untested.
4. **Dispersive Loss (`--disp`) is not supported in this mode.** It reads
   `return_act` from the model's forward pass (the double-backward graph's
   activations); `forward_energy_with_cache` has no equivalent hook. Using
   `--disp` with `--ebm forward-backwards-direct` is simply not wired up
   (not silently ignored — the CLI flag exists but nothing in
   `main_forward_backwards_direct` reads `args.disp`).
5. **`gate=True` / `qk_norm=True` / `scale_norm=True` variants of timm's
   `Attention` are NOT supported.** `models.py`'s `SiTBlock` never
   constructs `Attention` with those flags, so `attention_reverse` doesn't
   implement their VJPs. If a future architecture change turns any of
   these on, `fb_direct` needs new manual VJP terms — it will not silently
   produce a wrong answer, because `ReverseSiTBlock`/`attention_reverse`
   simply don't reference `attn.q_norm`/`attn.k_norm`/`attn.gate`/`attn.norm`
   at all (an architecture change that engaged them would need a code
   change here, and shape mismatches downstream would surface immediately
   in `test_end_to_end_field_equivalence_fp64`, not at IN-1K scale).
6. **The `shift`-chunk gradient dead zone** documented in Section 5's
   coverage caveat is a real, permanent property of this training method
   (not a bug): AdaLN's `shift` parameters can only ever be updated via
   `direct`-style double-backward or a future non-identity Pi that routes
   around it. `forward-backwards-direct` trains `scale`/`gate` but freezes
   `shift`'s *effective* contribution at whatever the `direct`/init
   checkpoint set it to (the weight itself isn't frozen — it's just
   provably never touched, since AdamW momentum for those rows stays 0
   from a gradient that is analytically 0 every step).
