"""
Typed reverse-cache dataclasses for `forward-backwards-direct`.

Every tensor stored here is produced under `torch.no_grad()` and is
explicitly `.detach()`-ed before being placed in the cache (belt-and-braces:
tensors computed under no_grad already carry `requires_grad=False`, but we
detach again at the call site in forward_cache.py so this invariant holds
even if that changes upstream).

Nothing here is unstructured -- ordering is by field name, not position in
a list, so a consumer cannot silently misalign tensors.
"""
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class AttentionReverseCache:
    """Everything needed to manually reverse one SiTBlock's attention.

    q, k: (B, H, N, dh) -- PRE-scale q/k (the `scale` multiply on q is
        reapplied inside attention_reverse to mirror forward exactly).
    v: (B, H, N, dh)
    p: (B, H, N, N) -- softmax attention probabilities.
    All detached, dtype matches the forward compute dtype.
    """
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    p: torch.Tensor


@dataclass
class MLPReverseCache:
    """pre_act: (B, N, 4D) -- fc1(x) BEFORE the GELU-tanh nonlinearity."""
    pre_act: torch.Tensor


@dataclass
class LayerNormReverseCache:
    """normalized: (B, N, D) -- LN output (elementwise_affine=False).
    inv_std: (B, N, 1) -- 1/std(x) used to produce `normalized`.
    """
    normalized: torch.Tensor
    inv_std: torch.Tensor


@dataclass
class SiTBlockReverseCache:
    """One SiTBlock's full reverse cache (attention residual + mlp residual)."""
    norm1: LayerNormReverseCache
    attn: AttentionReverseCache
    norm2: LayerNormReverseCache
    mlp: MLPReverseCache


@dataclass
class FinalLayerReverseCache:
    """ScalarEnergyHead's reverse cache: the final-block token stream after
    its own (unaffine) LayerNorm."""
    norm_final: LayerNormReverseCache


@dataclass
class TransformerReverseCache:
    """Full reverse cache for one forward_energy_with_cache() call.

    c: (B, D) -- conditioning vector t_emb + y_emb. Cached because
        t_embedder/y_embedder are `cache_only` (see param_mapping.py) --
        the reverse network recomputes adaLN modulation from this cached,
        detached c using its own (phi) adaLN weights, but does not
        recompute c itself from t/y.
    grid_h, grid_w, patch_size: ints, needed to invert PatchEmbed's
        flatten/transpose when reconstructing spatial layout for the
        conv_transpose2d patch-embed reverse.
    blocks: per-SiTBlock reverse caches, in forward order (block 0 first).
    final: ScalarEnergyHead reverse cache.
    energy: (B,) -- the scalar energy value itself (for diagnostics/audit
        only; not consumed by the reverse computation).
    """
    c: torch.Tensor
    grid_h: int
    grid_w: int
    patch_size: int
    blocks: List[SiTBlockReverseCache] = field(default_factory=list)
    final: FinalLayerReverseCache = None
    energy: torch.Tensor = None
