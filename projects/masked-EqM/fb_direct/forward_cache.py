"""
forward_energy_with_cache(): runs the SAME scalar-energy forward computation
as `EqM.forward(..., ebm='direct')` / `ebm='forward-backwards-direct'`, but

  1. executes entirely under `torch.no_grad()` (theta builds no autograd
     graph at all -- invariant #3/#7 in the spec), and
  2. returns a structured, DETACHED `TransformerReverseCache` alongside the
     scalar energy, holding exactly the intermediate tensors the explicit
     reverse network needs (invariant #4).

This function must be numerically IDENTICAL to the model's ordinary
`ebm='direct'` energy computation for the same weights/inputs -- that
identity is checked in tests/test_forward_backwards_direct.py
(test_forward_cache_matches_direct_energy).
"""
import torch
import torch.nn.functional as F

from .cache import (
    AttentionReverseCache,
    FinalLayerReverseCache,
    LayerNormReverseCache,
    MLPReverseCache,
    SiTBlockReverseCache,
    TransformerReverseCache,
)


def _layernorm_noaffine_forward(x, eps):
    """Matches nn.LayerNorm(D, elementwise_affine=False, eps=eps) exactly,
    while also returning the per-token inv_std needed for the reverse pass.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    normalized = (x - mean) * inv_std
    return normalized, inv_std


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@torch.no_grad()
def forward_energy_with_cache(model, z, t, y):
    """
    Args:
      model: an EqM instance with ebm in {'direct', 'forward-backwards-direct'}.
      z: (B, C, H, W) input latent (the model's `x0` argument).
      t: (B,) timesteps (zeroed internally if model.uncond, matching forward()).
      y: (B,) class labels.
    Returns:
      E: (B,) scalar energy, detached.
      cache: TransformerReverseCache, every tensor detached.
    """
    if model.ebm not in ('direct', 'forward-backwards-direct'):
        raise ValueError(
            f"forward_energy_with_cache requires a scalar-energy model "
            f"(ebm='direct' or 'forward-backwards-direct'), got ebm={model.ebm!r}"
        )

    t_in = torch.zeros_like(t) if model.uncond else t

    x_embedder = model.x_embedder
    grid_h = grid_w = int(x_embedder.num_patches ** 0.5)
    patch_size = x_embedder.patch_size[0]

    x = x_embedder(z) + model.pos_embed  # (B, T, D)
    t_emb = model.t_embedder(t_in)  # (B, D)
    y_emb = model.y_embedder(y, model.training)  # (B, D)
    c = t_emb + y_emb  # (B, D)

    block_caches = []
    for block in model.blocks:
        x, block_cache = _sit_block_forward_with_cache(block, x, c)
        block_caches.append(block_cache)

    head = model.energy_head
    shift, scale = head.adaLN_modulation(c).chunk(2, dim=1)
    normalized_final, inv_std_final = _layernorm_noaffine_forward(x, head.norm_final.eps)
    x_mod = _modulate(normalized_final, shift, scale)
    token_energies = head.linear(x_mod).squeeze(-1)  # (B, T)
    E = token_energies.sum(dim=1)  # (B,)

    cache = TransformerReverseCache(
        c=c.detach(),
        grid_h=grid_h,
        grid_w=grid_w,
        patch_size=patch_size,
        blocks=block_caches,
        final=FinalLayerReverseCache(
            norm_final=LayerNormReverseCache(
                normalized=normalized_final.detach(),
                inv_std=inv_std_final.detach(),
            )
        ),
        energy=E.detach(),
    )
    return E.detach(), cache


def _sit_block_forward_with_cache(block, x, c):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.adaLN_modulation(c).chunk(6, dim=1)
    )

    normalized1, inv_std1 = _layernorm_noaffine_forward(x, block.norm1.eps)
    x_mod1 = _modulate(normalized1, shift_msa, scale_msa)
    attn_out, attn_cache = _attention_forward_with_cache(block.attn, x_mod1)
    x = x + gate_msa.unsqueeze(1) * attn_out

    normalized2, inv_std2 = _layernorm_noaffine_forward(x, block.norm2.eps)
    x_mod2 = _modulate(normalized2, shift_mlp, scale_mlp)
    mlp_out, mlp_cache = _mlp_forward_with_cache(block.mlp, x_mod2)
    x = x + gate_mlp.unsqueeze(1) * mlp_out

    block_cache = SiTBlockReverseCache(
        norm1=LayerNormReverseCache(normalized=normalized1.detach(), inv_std=inv_std1.detach()),
        attn=attn_cache,
        norm2=LayerNormReverseCache(normalized=normalized2.detach(), inv_std=inv_std2.detach()),
        mlp=mlp_cache,
    )
    return x, block_cache


def _attention_forward_with_cache(attn, x):
    B, N, C = x.shape
    H, dh = attn.num_heads, attn.head_dim
    qkv = attn.qkv(x).reshape(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # each (B, H, N, dh), PRE-scale

    q_scaled = q * attn.scale
    s = torch.matmul(q_scaled, k.transpose(-2, -1))
    p = s.softmax(dim=-1)
    o = torch.matmul(p, v)  # (B, H, N, dh)

    o = o.transpose(1, 2).reshape(B, N, C)
    y = attn.proj(o)

    attn_cache = AttentionReverseCache(
        q=q.detach(), k=k.detach(), v=v.detach(), p=p.detach(),
    )
    return y, attn_cache


def _mlp_forward_with_cache(mlp, x):
    pre_act = mlp.fc1(x)
    h1 = F.gelu(pre_act, approximate="tanh")
    y = mlp.fc2(h1)
    mlp_cache = MLPReverseCache(pre_act=pre_act.detach())
    return y, mlp_cache
