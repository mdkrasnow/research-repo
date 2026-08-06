"""
ReverseEqM: the explicit reverse network R_phi.

An ordinary nn.Module of trainable "backward" parameters phi that computes

    g_tilde = R_phi(sg[C_theta(z)])

as a plain forward tensor program (Section 5.2 of the spec). It never calls
torch.autograd.grad and never touches the forward model's autograd graph
(there isn't one -- forward_cache.py runs theta under no_grad). Ordinary
`loss.backward()` on a loss built from this module's output differentiates
ONLY through phi, by construction (the cache tensors it reads are all
detached leaves).

Mapping convention (Pi): every phi weight tensor has the IDENTICAL shape
and orientation as its forward-theta counterpart (see reverse_ops.py: our
manual VJPs are `u_y @ W`, which reuses the forward Linear.weight
(out_features, in_features) layout directly -- and the patch-embed
conv_transpose2d reuses the forward Conv2d weight (D, C, p, p) directly
too). So Pi/Pi-dagger reduce to plain identity copies for every
reverse_active parameter in this architecture; see param_mapping.py.
"""
import torch
import torch.nn as nn

from .reverse_ops import (
    attention_reverse,
    layernorm_noaffine_reverse,
    linear_reverse,
    mlp_reverse,
    modulate_reverse_wrt_norm,
    patch_embed_reverse,
)


class ReverseSiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # adaLN_modulation mirrors forward's nn.Sequential(SiLU(), Linear(...))
        # at path `adaLN_modulation.1.{weight,bias}` -- kept as a bare Linear
        # here (index 1) so the fully-qualified name still ends in
        # `adaLN_modulation.1.weight`, for a 1:1 name match in the mapping
        # registry. See param_mapping.py PARAM_NAME_SUFFIX.
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_fc1 = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.mlp_fc2 = nn.Linear(4 * hidden_size, hidden_size, bias=False)

    def reverse(self, u_x_next, block_cache, c):
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=1)

        # --- mlp residual branch ---
        u_mlp_out = gate_mlp.unsqueeze(1) * u_x_next
        u_xmod2 = mlp_reverse(
            u_mlp_out,
            pre_act=block_cache.mlp.pre_act,
            weight_fc1_b=self.mlp_fc1.weight,
            weight_fc2_b=self.mlp_fc2.weight,
        )
        u_norm2 = modulate_reverse_wrt_norm(u_xmod2, scale_mlp)
        u_x_mid = layernorm_noaffine_reverse(
            u_norm2, block_cache.norm2.normalized, block_cache.norm2.inv_std
        )
        u_x_mid = u_x_mid + u_x_next  # identity skip connection

        # --- attention residual branch ---
        u_attn_out = gate_msa.unsqueeze(1) * u_x_mid
        u_xmod1 = attention_reverse(
            u_attn_out,
            q=block_cache.attn.q, k=block_cache.attn.k,
            v=block_cache.attn.v, p=block_cache.attn.p,
            weight_proj_b=self.attn_proj.weight,
            weight_qkv_b=self.attn_qkv.weight,
            num_heads=self.num_heads, head_dim=self.head_dim, scale=self.scale,
        )
        u_norm1 = modulate_reverse_wrt_norm(u_xmod1, scale_msa)
        u_x = layernorm_noaffine_reverse(
            u_norm1, block_cache.norm1.normalized, block_cache.norm1.inv_std
        )
        u_x = u_x + u_x_mid  # identity skip connection
        return u_x


class ReverseEqM(nn.Module):
    """Explicit reverse network phi. Constructed from a forward EqM instance
    purely to read off architecture hyperparameters (hidden_size, depth,
    num_heads, patch shape) -- no forward weights are copied here; use
    ParameterMappingRegistry.tie_from_forward_() for that (hard sync, done
    every training iteration per the spec).
    """
    def __init__(self, forward_model):
        super().__init__()
        if forward_model.ebm not in ('direct', 'forward-backwards-direct'):
            raise ValueError(
                f"ReverseEqM requires a scalar-energy forward model, got ebm={forward_model.ebm!r}"
            )
        hidden_size = forward_model.x_embedder.proj.out_channels
        depth = len(forward_model.blocks)
        num_heads = forward_model.num_heads
        in_channels = forward_model.x_embedder.proj.in_channels
        patch_size = forward_model.x_embedder.patch_size[0]

        self.x_embedder_weight = nn.Parameter(
            torch.empty(hidden_size, in_channels, patch_size, patch_size)
        )
        nn.init.zeros_(self.x_embedder_weight)

        self.blocks = nn.ModuleList([
            ReverseSiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.energy_head_adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.energy_head_linear = nn.Linear(hidden_size, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cache):
        """R_phi(sg[C_theta(z)]) -> g_tilde, the reverse-network field
        estimate. Equals grad_z E_theta(z) exactly when phi = Pi(theta).
        """
        normalized_f = cache.final.norm_final.normalized
        inv_std_f = cache.final.norm_final.inv_std
        B, T, D = normalized_f.shape

        shift_f, scale_f = self.energy_head_adaLN_modulation(cache.c).chunk(2, dim=1)

        u_linear_out = torch.ones(B, T, 1, device=normalized_f.device, dtype=normalized_f.dtype)
        u_xmod_f = linear_reverse(u_linear_out, self.energy_head_linear.weight)
        u_norm_f = modulate_reverse_wrt_norm(u_xmod_f, scale_f)
        u_x = layernorm_noaffine_reverse(u_norm_f, normalized_f, inv_std_f)

        for block, block_cache in zip(reversed(self.blocks), reversed(cache.blocks)):
            u_x = block.reverse(u_x, block_cache, cache.c)

        u_z = patch_embed_reverse(u_x, self.x_embedder_weight, cache.patch_size, cache.grid_h, cache.grid_w)
        return u_z
