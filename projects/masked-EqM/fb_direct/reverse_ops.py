"""
Elementary manual vector-Jacobian-product (VJP) building blocks for the
`forward-backwards-direct` scalar-energy mode.

Every function here implements the *exact* local reverse-mode derivative of
one primitive used by `models.py`'s `EqM` forward pass (patch embed,
LayerNorm w/o affine, GELU-tanh, timm's unfused/math-backend attention,
Linear). None of these functions call `torch.autograd` -- they are ordinary
tensor programs so that an enclosing `loss.backward()` differentiates
*through* them (producing gradients for whatever trainable weight tensors
are passed in), without ever touching the true forward computation graph.

Sign/shape conventions match models.py exactly (see documentation/
forward-backwards-direct.md for the full derivation and worked equations).
"""
import contextlib
import math

import torch
import torch.nn.functional as F


@contextlib.contextmanager
def _strict_fp32_accumulation():
    """Temporarily force true IEEE-754 binary32 (23-bit mantissa) matmul
    accumulation by disabling TF32 (10-bit mantissa tensor-core) reduced
    precision for the duration of the `with` block, restoring whatever the
    caller's global setting was afterward.

    Root-cause note (Gate 0 retune, 2026-08-06): `train.py` sets
    `torch.backends.cuda.matmul.allow_tf32 = True` / `cudnn.allow_tf32 = True`
    at import time (a global, process-wide flag, standard practice for A100
    training throughput) and `fb_direct_gate0_field_check.py` imports
    `train.py` for `center_crop_arr`, so that flag is live for every CUDA
    matmul in the process, including these hand-derived reverse VJPs.
    TF32 truncates matmul operand mantissas to ~10 bits (~3 decimal
    digits) before the (still FP32-accumulated) multiply-add; the FP64
    unit tests (rel err 7.6e-16) and the FP32 *CPU* unit test (rel err
    3.9e-7, no CUDA tensor cores => no TF32) both pass tightly, while the
    real GPU gate-0 run on the real epoch-40 checkpoint (12 SiT blocks x
    ~7 matmuls/block through this reverse path) produced audit_mean_cosine
    ~1.0 (direction preserved -- not a mechanism/sign bug) but
    audit_mean_rel_error ~1.5-1.8e-4 (magnitude drift, right order for
    TF32 rounding compounding across depth). This is the only mechanism
    consistent with all four data points simultaneously, so it is treated
    as the root cause rather than a numerology coincidence.

    Scope: wraps ONLY the matmul-bearing manual VJPs identified as
    numerically sensitive by the spec (Section 10) -- `attention_reverse`
    (softmax backward + its surrounding QKV/proj matmuls) and
    `layernorm_noaffine_reverse` (mean/inv-std reduction terms, upcast
    defensively even though plain reductions are not TF32-gated, for
    forward-compatibility with any future non-FP32 cache dtype). Does NOT
    touch `linear_reverse`/`mlp_reverse`/`patch_embed_reverse` individually
    (they call into `linear_reverse`, which is not wrapped on its own --
    only the two functions explicitly named in the retune directive are
    wrapped here), the forward energy computation (`forward_cache.py`), or
    any unit test tolerance.
    """
    cuda_prev = torch.backends.cuda.matmul.allow_tf32
    cudnn_prev = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = cuda_prev
        torch.backends.cudnn.allow_tf32 = cudnn_prev


def linear_reverse(u_y, weight_b):
    """Reverse of y = x @ W^T (+ b). Bias never appears (dy/dx independent
    of b), so bias has no reverse contribution here -- classified cache_only.

    u_y: (..., out_features)
    weight_b: (out_features, in_features) -- the backward-side copy of the
        forward Linear's weight.
    returns u_x: (..., in_features)
    """
    return torch.matmul(u_y, weight_b)


def gelu_tanh_prime(a):
    """d/da GELU_tanh(a), matching nn.GELU(approximate='tanh') exactly.

    GELU_tanh(a) = 0.5*a*(1 + tanh(sqrt(2/pi)*(a + 0.044715*a^3)))
    """
    c0 = math.sqrt(2.0 / math.pi)
    c1 = 0.044715
    a2 = a * a
    inner = c0 * (a + c1 * a * a2)
    tanh_inner = torch.tanh(inner)
    sech2 = 1.0 - tanh_inner * tanh_inner
    dinner_da = c0 * (1.0 + 3.0 * c1 * a2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * a * sech2 * dinner_da


def silu_prime(a):
    """d/da SiLU(a) = sigmoid(a) * (1 + a * (1 - sigmoid(a)))."""
    s = torch.sigmoid(a)
    return s * (1.0 + a * (1.0 - s))


def layernorm_noaffine_reverse(u_y, y_normalized, inv_std):
    """Reverse of y = (x - mean(x)) / std(x) over the last dim, NO affine
    scale/shift (matches nn.LayerNorm(elementwise_affine=False)).

    u_y, y_normalized: (..., D)
    inv_std: (..., 1) -- cached 1/std(x) per token from the forward pass.
    returns u_x: (..., D)

    Standard LN backward (no affine):
      u_x = inv_std * ( u_y - mean(u_y) - y_normalized * mean(u_y * y_normalized) )
    (means taken over the normalized dimension, i.e. dim=-1)

    Numerically sensitive: the two reduction terms (`mean_uy`,
    `mean_uy_y`) are promoted to AT LEAST strict FP32 accumulation (see
    `_strict_fp32_accumulation`), per the retune directive -- computed and
    combined in `torch.promote_types(orig_dtype, float32)` precision
    regardless of the caller's ambient dtype or global TF32 setting, then
    cast back to the input dtype. This is a floor, not a ceiling: an
    already-FP64 caller (e.g. the FP64 unit tests) is left at FP64, never
    downcast to FP32 -- only FP16/BF16/FP32 inputs are raised to FP32.
    """
    orig_dtype = u_y.dtype
    compute_dtype = torch.promote_types(orig_dtype, torch.float32)
    with _strict_fp32_accumulation():
        u_y_c = u_y.to(compute_dtype)
        y_normalized_c = y_normalized.to(compute_dtype)
        inv_std_c = inv_std.to(compute_dtype)
        mean_uy = u_y_c.mean(dim=-1, keepdim=True)
        mean_uy_y = (u_y_c * y_normalized_c).mean(dim=-1, keepdim=True)
        u_x_c = inv_std_c * (u_y_c - mean_uy - y_normalized_c * mean_uy_y)
    return u_x_c.to(orig_dtype)


def modulate_reverse_wrt_norm(u_xmod, scale):
    """Reverse of x_mod = norm_x * (1 + scale) + shift wrt norm_x only.
    (shift has zero local Jacobian wrt norm_x/x; scale/shift depend only on
    conditioning c, which is independent of the input latent z.)

    u_xmod: (B, T, D); scale: (B, D) -> broadcast over tokens.
    """
    return u_xmod * (1.0 + scale.unsqueeze(1))


def attention_reverse(u_y, *, q, k, v, p, weight_proj_b, weight_qkv_b, num_heads, head_dim, scale):
    """Exact reverse of timm's unfused (math-backend) multi-head
    self-attention:

        qkv = Linear_qkv(x)                       # (B, N, 3D)
        q,k,v = split/reshape -> (B, H, N, dh)
        q = q * scale
        s = q @ k^T                                # (B, H, N, N)
        p = softmax(s, dim=-1)
        o = p @ v                                  # (B, H, N, dh)
        o -> reshape -> (B, N, D)
        y = Linear_proj(o)

    Args:
      u_y: (B, N, D) incoming gradient wrt attention output y.
      q, k, v: (B, H, N, dh) cached, DETACHED forward tensors. `q` here is
        the pre-scale value (scale applied inside this function to mirror
        forward exactly); see forward_cache.py for what is actually stored.
      p: (B, H, N, N) cached, DETACHED softmax attention probabilities.
      weight_proj_b: (D, D) backward copy of attn.proj.weight.
      weight_qkv_b: (3D, D) backward copy of attn.qkv.weight.
    Returns:
      u_x: (B, N, D) gradient wrt the block input to this attention module
        (i.e. wrt `modulate(norm1(x), shift_msa, scale_msa)`).

    Numerically sensitive: this entire chain (proj/qkv linear VJPs, the
    o=p@v and s=q'@k^T matmul VJPs, and the softmax backward) is promoted
    to AT LEAST strict FP32 accumulation (see `_strict_fp32_accumulation`),
    per the retune directive -- every tensor is upcast to
    `torch.promote_types(orig_dtype, float32)` on entry, TF32
    reduced-precision tensor-core matmul is disabled for the duration, and
    the result is cast back to the caller's dtype on exit. This is a
    floor, not a ceiling: an already-FP64 caller (e.g. the FP64 unit
    tests) is left at FP64, never downcast to FP32.
    """
    orig_dtype = u_y.dtype
    compute_dtype = torch.promote_types(orig_dtype, torch.float32)
    with _strict_fp32_accumulation():
        u_y = u_y.to(compute_dtype)
        q = q.to(compute_dtype)
        k = k.to(compute_dtype)
        v = v.to(compute_dtype)
        p = p.to(compute_dtype)
        weight_proj_b = weight_proj_b.to(compute_dtype)
        weight_qkv_b = weight_qkv_b.to(compute_dtype)

        B, N, D = u_y.shape
        H, dh = num_heads, head_dim

        # --- reverse of proj: y = Linear_proj(o) ---
        u_o = linear_reverse(u_y, weight_proj_b)  # (B, N, D)
        u_o = u_o.reshape(B, N, H, dh).transpose(1, 2)  # (B, H, N, dh)

        # --- reverse of o = p @ v ---
        u_p = torch.matmul(u_o, v.transpose(-2, -1))  # (B, H, N, N)
        u_v = torch.matmul(p.transpose(-2, -1), u_o)  # (B, H, N, dh)

        # --- reverse of softmax: p = softmax(s, dim=-1) ---
        u_s = p * (u_p - (u_p * p).sum(dim=-1, keepdim=True))  # (B, H, N, N)

        # --- reverse of s = q' @ k^T, where q' = q * scale ---
        q_scaled = q * scale
        u_qprime = torch.matmul(u_s, k)  # (B, H, N, dh)
        u_k = torch.matmul(u_s.transpose(-2, -1), q_scaled)  # (B, H, N, dh)
        u_q = u_qprime * scale  # chain through the constant `scale` factor

        # --- reassemble qkv gradient and reverse the qkv Linear ---
        # forward: qkv.reshape(B,N,3,H,dh).permute(2,0,3,1,4) -> unbind(0) -> q,k,v
        u_qkv_permuted = torch.stack([u_q, u_k, u_v], dim=0)  # (3, B, H, N, dh)
        u_qkv = u_qkv_permuted.permute(1, 3, 0, 2, 4).reshape(B, N, 3 * D)  # (B, N, 3D)
        u_x = linear_reverse(u_qkv, weight_qkv_b)  # (B, N, D)
    return u_x.to(orig_dtype)


def mlp_reverse(u_y, *, pre_act, weight_fc1_b, weight_fc2_b):
    """Reverse of y = fc2(GELU_tanh(fc1(x))) (timm Mlp, drop=0, Identity norm).

    u_y: (B, N, D) gradient wrt mlp output.
    pre_act: (B, N, 4D) cached, DETACHED fc1 pre-activation (a1 = fc1(x)).
    weight_fc1_b: (4D, D) backward copy of mlp.fc1.weight.
    weight_fc2_b: (D, 4D) backward copy of mlp.fc2.weight.
    Returns u_x: (B, N, D) gradient wrt mlp input.
    """
    u_h1 = linear_reverse(u_y, weight_fc2_b)  # (B, N, 4D)
    u_a1 = u_h1 * gelu_tanh_prime(pre_act)  # (B, N, 4D)
    u_x = linear_reverse(u_a1, weight_fc1_b)  # (B, N, D)
    return u_x


def patch_embed_reverse(u_tokens, weight_conv_b, patch_size, grid_h, grid_w):
    """Reverse of PatchEmbed: Conv2d(kernel=stride=patch_size) -> flatten(2)
    -> transpose(1,2). Non-overlapping (stride == kernel_size) so the exact
    input-gradient of the conv is a conv_transpose2d with the SAME weight
    tensor shape convention, using the backward copy of the weight.

    u_tokens: (B, T, D), T = grid_h * grid_w, row-major (h, w) order.
    weight_conv_b: (D, C, p, p) backward copy of x_embedder.proj.weight.
    Returns u_z: (B, C, H, W).
    """
    B, T, D = u_tokens.shape
    u_spatial = u_tokens.transpose(1, 2).reshape(B, D, grid_h, grid_w)  # (B, D, H', W')
    u_z = F.conv_transpose2d(u_spatial, weight_conv_b, stride=patch_size)
    return u_z
