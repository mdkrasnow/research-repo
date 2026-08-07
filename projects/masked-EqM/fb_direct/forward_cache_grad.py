"""
forward_energy_with_cache_grad(): a GRADIENT-CARRYING twin of
forward_cache.py's forward_energy_with_cache().

Same forward computation, same tensor-cache sites (q/k/v/p per block,
pre-GELU MLP activation, pre/post-LayerNorm states, final-layer state), but:

  1. NOT wrapped in `torch.no_grad()` -- theta builds a full autograd graph
     from parameters through every cached tensor to the scalar energy E.
  2. cache tensors are NOT `.detach()`-ed; each leaf-like intermediate that
     `forward_cache.py` caches has `.retain_grad()` called on it, so that
     after a downstream `.backward()` its `.grad` holds the exact cache
     adjoint a*_l = dL/dC_l for that tensor.

This function exists ONLY for the mandatory decomposition test (Section 5 of
`documentation/fb_direct_learned_cache_adjoint_proposal.md`): it lets us
extract a* = dL/dC from the SAME real double-backward computation
`exact_gradient_audit`/Gate-1/2 already use, and separately re-run the
forward pass to obtain an ordinary (non-second-order) VJP
J_{C_theta}(theta)^T stopgrad(a*) reconstructing g_cache. It must NEVER be
used on a production training path -- it exists purely to produce and
consume a* tensors for diagnostics; production forward-backwards-direct
training continues to use `forward_cache.py`'s `torch.no_grad()` version
exclusively (`trainer.py.training_step` is unchanged).

Numerically identical to `forward_cache.py.forward_energy_with_cache` and to
`models.py`'s `ebm='direct'` energy computation for the same weights/inputs;
see `tests/test_fb_direct_cache_adjoint.py::test_forward_cache_grad_matches_direct_energy`.
"""
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class AttentionGradCache:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    p: torch.Tensor


@dataclass
class MLPGradCache:
    pre_act: torch.Tensor


@dataclass
class LayerNormGradCache:
    normalized: torch.Tensor
    inv_std: torch.Tensor


@dataclass
class SiTBlockGradCache:
    norm1: LayerNormGradCache
    attn: AttentionGradCache
    norm2: LayerNormGradCache
    mlp: MLPGradCache


@dataclass
class FinalLayerGradCache:
    norm_final: LayerNormGradCache


@dataclass
class TransformerGradCache:
    c: torch.Tensor
    grid_h: int
    grid_w: int
    patch_size: int
    blocks: List[SiTBlockGradCache] = field(default_factory=list)
    final: FinalLayerGradCache = None
    energy: torch.Tensor = None

    def flatten(self):
        """Return an OrderedDict-like list of (name, tensor) for every
        retain_grad()-able cache tensor, in a stable, deterministic order
        (block index ascending, then attention before mlp, matching
        forward_cache.py's per-block field order). Names are used only as
        dict keys to pair a* targets with the VJP-reconstruction pass's
        tensors of the SAME name -- they are not theta/phi parameter names.
        """
        items = [("c", self.c)]
        for i, b in enumerate(self.blocks):
            items.append((f"blocks.{i}.norm1.normalized", b.norm1.normalized))
            items.append((f"blocks.{i}.norm1.inv_std", b.norm1.inv_std))
            items.append((f"blocks.{i}.attn.q", b.attn.q))
            items.append((f"blocks.{i}.attn.k", b.attn.k))
            items.append((f"blocks.{i}.attn.v", b.attn.v))
            items.append((f"blocks.{i}.attn.p", b.attn.p))
            items.append((f"blocks.{i}.norm2.normalized", b.norm2.normalized))
            items.append((f"blocks.{i}.norm2.inv_std", b.norm2.inv_std))
            items.append((f"blocks.{i}.mlp.pre_act", b.mlp.pre_act))
        items.append(("final.norm_final.normalized", self.final.norm_final.normalized))
        items.append(("final.norm_final.inv_std", self.final.norm_final.inv_std))
        return items

    def block_of(self, name):
        """Map a flatten() name back to the same block-grouping convention
        used by layerwise_gradient_decomposition.py / blockwise_calibration_test.py
        ('x_embedder' | 'blocks.<i>' | 'energy_head'), for reporting cache
        tensor contributions per block. `c` and the patch-embed input are
        both consumed before block 0 conceptually, but `c` itself has no
        single owning block (it feeds every block's adaLN) -- reported
        under its own 'conditioning' key rather than misattributed.
        """
        if name == "c":
            return "conditioning"
        if name.startswith("final."):
            return "energy_head"
        if name.startswith("blocks."):
            return f"blocks.{name.split('.')[1]}"
        raise ValueError(f"unrecognized cache tensor name: {name}")


def _layernorm_noaffine_forward(x, eps):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    normalized = (x - mean) * inv_std
    return normalized, inv_std


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _retain(t):
    if t.requires_grad:
        t.retain_grad()
    return t


def forward_energy_with_cache_grad(model, z, t, y):
    """Gradient-carrying counterpart of `forward_cache.forward_energy_with_cache`.

    Args: identical to forward_energy_with_cache. `z` should be a leaf with
      `requires_grad_(True)` if the caller also wants dE/dz (e.g. for the
      exact-arm double backward); this function does not require that on
      its own -- it only needs `model`'s parameters to require grad for a*
      to be populated via retain_grad() after backward().

    Returns:
      E: (B,) scalar energy WITH GRAD (not detached).
      cache: TransformerGradCache, every leaf-like tensor retain_grad()-ed.

    Caller contract: call `.retain_grad()` sites are already applied here;
    the caller must still keep the tensors returned via `cache.flatten()`
    alive (no `del cache`) until after `.backward()` so `.grad` populates.
    """
    if model.ebm not in ('direct', 'forward-backwards-direct'):
        raise ValueError(
            f"forward_energy_with_cache_grad requires a scalar-energy model "
            f"(ebm='direct' or 'forward-backwards-direct'), got ebm={model.ebm!r}"
        )

    t_in = torch.zeros_like(t) if model.uncond else t

    x_embedder = model.x_embedder
    grid_h = grid_w = int(x_embedder.num_patches ** 0.5)
    patch_size = x_embedder.patch_size[0]

    x = x_embedder(z) + model.pos_embed
    t_emb = model.t_embedder(t_in)
    y_emb = model.y_embedder(y, model.training)
    c = _retain(t_emb + y_emb)

    block_caches = []
    for block in model.blocks:
        x, block_cache = _sit_block_forward_with_cache_grad(block, x, c)
        block_caches.append(block_cache)

    head = model.energy_head
    shift, scale = head.adaLN_modulation(c).chunk(2, dim=1)
    normalized_final, inv_std_final = _layernorm_noaffine_forward(x, head.norm_final.eps)
    normalized_final = _retain(normalized_final)
    inv_std_final = _retain(inv_std_final)
    x_mod = _modulate(normalized_final, shift, scale)
    token_energies = head.linear(x_mod).squeeze(-1)
    E = token_energies.sum(dim=1)

    cache = TransformerGradCache(
        c=c,
        grid_h=grid_h,
        grid_w=grid_w,
        patch_size=patch_size,
        blocks=block_caches,
        final=FinalLayerGradCache(
            norm_final=LayerNormGradCache(normalized=normalized_final, inv_std=inv_std_final)
        ),
        energy=E,
    )
    return E, cache


def _sit_block_forward_with_cache_grad(block, x, c):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.adaLN_modulation(c).chunk(6, dim=1)
    )

    normalized1, inv_std1 = _layernorm_noaffine_forward(x, block.norm1.eps)
    normalized1 = _retain(normalized1)
    inv_std1 = _retain(inv_std1)
    x_mod1 = _modulate(normalized1, shift_msa, scale_msa)
    attn_out, attn_cache = _attention_forward_with_cache_grad(block.attn, x_mod1)
    x = x + gate_msa.unsqueeze(1) * attn_out

    normalized2, inv_std2 = _layernorm_noaffine_forward(x, block.norm2.eps)
    normalized2 = _retain(normalized2)
    inv_std2 = _retain(inv_std2)
    x_mod2 = _modulate(normalized2, shift_mlp, scale_mlp)
    mlp_out, mlp_cache = _mlp_forward_with_cache_grad(block.mlp, x_mod2)
    x = x + gate_mlp.unsqueeze(1) * mlp_out

    block_cache = SiTBlockGradCache(
        norm1=LayerNormGradCache(normalized=normalized1, inv_std=inv_std1),
        attn=attn_cache,
        norm2=LayerNormGradCache(normalized=normalized2, inv_std=inv_std2),
        mlp=mlp_cache,
    )
    return x, block_cache


def _attention_forward_with_cache_grad(attn, x):
    B, N, C = x.shape
    H, dh = attn.num_heads, attn.head_dim
    qkv = attn.qkv(x).reshape(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = _retain(q)
    k = _retain(k)
    v = _retain(v)

    q_scaled = q * attn.scale
    s = torch.matmul(q_scaled, k.transpose(-2, -1))
    p = s.softmax(dim=-1)
    p = _retain(p)
    o = torch.matmul(p, v)

    o = o.transpose(1, 2).reshape(B, N, C)
    y = attn.proj(o)

    attn_cache = AttentionGradCache(q=q, k=k, v=v, p=p)
    return y, attn_cache


def _mlp_forward_with_cache_grad(mlp, x):
    pre_act = _retain(mlp.fc1(x))
    h1 = F.gelu(pre_act, approximate="tanh")
    y = mlp.fc2(h1)
    mlp_cache = MLPGradCache(pre_act=pre_act)
    return y, mlp_cache
