"""
Gate 0 retune #3 (FINAL granted retune): FP64 block-by-block bisection.

Purpose: distinguish "real remaining VJP bug" from "pure GPU/TF32 precision
noise" by running the manual reverse chain against autograd in float64 (where
TF32/fused-attention-kernel differences are moot -- TF32 only truncates FP32
matmul operands; float64 tensor-core paths don't exist, so cuBLAS/cuDNN fall
back to full double-precision accumulation regardless of any global flag),
using the REAL epoch-40 checkpoint's REAL trained weights (not a tiny random
init), on the real EqM-B/2 architecture (hidden_size=768, depth=12,
num_heads=12) -- exactly the config that failed Gate 0, unlike the passing
FP64 unit test which used EqM-S/2 (hidden_size=384, num_heads=6, same depth).

If this reproduces a large error -> genuine mechanism bug, localized to a
specific block/op by the per-block comparison below.
If this reports ~1e-12 error end-to-end -> the manual VJP math is exactly
correct at B/2 scale, and the whole ~1.5e-4 Gate-0 gap is TF32/fused-kernel
GPU precision noise, not a bug -- Gate 0's <1e-4 threshold is then too strict
for an FP32/TF32 training regime and should be revisited, not "fixed" with
more code changes.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

import models
from models import EqM_models

# TimestepEmbedder.timestep_embedding() hardcodes `.float()` on its
# sinusoidal table regardless of model dtype (models.py:52) -- irrelevant for
# real FP32 training/inference, but breaks a fully-.double()'d model. Same
# patch as tests/test_forward_backwards_direct.py.
def _dtype_safe_timestep_forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_freq = t_freq.to(self.mlp[0].weight.dtype)
    return self.mlp(t_freq)


models.TimestepEmbedder.forward = _dtype_safe_timestep_forward
from fb_direct.param_mapping import ParameterMappingRegistry
from fb_direct.reverse_model import ReverseEqM
from fb_direct.cache import (
    AttentionReverseCache, FinalLayerReverseCache, LayerNormReverseCache,
    MLPReverseCache, SiTBlockReverseCache, TransformerReverseCache,
)
from fb_direct.forward_cache import _layernorm_noaffine_forward, _modulate


def build_grad_tracked_cache(model, z, t, y):
    """Mirrors forward_cache.forward_energy_with_cache exactly, but keeps
    the autograd graph alive (no torch.no_grad, no .detach()) and calls
    .retain_grad() on every block's input tensor `x`, so we can read off
    autograd's TRUE intermediate backprop vector at each block boundary
    after a single .backward() call -- the ground truth to bisect against.
    """
    t_in = torch.zeros_like(t) if model.uncond else t
    x_embedder = model.x_embedder
    grid_h = grid_w = int(x_embedder.num_patches ** 0.5)
    patch_size = x_embedder.patch_size[0]

    x = x_embedder(z) + model.pos_embed
    t_emb = model.t_embedder(t_in)
    y_emb = model.y_embedder(y, model.training)
    c = t_emb + y_emb

    block_inputs = []  # retain_grad'd x BEFORE each block (block_inputs[i] = input to block i)
    block_caches = []
    for block in model.blocks:
        x.retain_grad()
        block_inputs.append(x)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.adaLN_modulation(c).chunk(6, dim=1)
        )
        normalized1, inv_std1 = _layernorm_noaffine_forward(x, block.norm1.eps)
        x_mod1 = _modulate(normalized1, shift_msa, scale_msa)

        B, N, C = x_mod1.shape
        H, dh = block.attn.num_heads, block.attn.head_dim
        qkv = block.attn.qkv(x_mod1).reshape(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q_scaled = q * block.attn.scale
        s = torch.matmul(q_scaled, k.transpose(-2, -1))
        p = s.softmax(dim=-1)
        o = torch.matmul(p, v)
        o = o.transpose(1, 2).reshape(B, N, C)
        attn_out = block.attn.proj(o)

        x2 = x + gate_msa.unsqueeze(1) * attn_out

        normalized2, inv_std2 = _layernorm_noaffine_forward(x2, block.norm2.eps)
        x_mod2 = _modulate(normalized2, shift_mlp, scale_mlp)
        pre_act = block.mlp.fc1(x_mod2)
        h1 = torch.nn.functional.gelu(pre_act, approximate="tanh")
        mlp_out = block.mlp.fc2(h1)

        x = x2 + gate_mlp.unsqueeze(1) * mlp_out

        block_caches.append(SiTBlockReverseCache(
            norm1=LayerNormReverseCache(normalized=normalized1.detach(), inv_std=inv_std1.detach()),
            attn=AttentionReverseCache(q=q.detach(), k=k.detach(), v=v.detach(), p=p.detach()),
            norm2=LayerNormReverseCache(normalized=normalized2.detach(), inv_std=inv_std2.detach()),
            mlp=MLPReverseCache(pre_act=pre_act.detach()),
        ))

    x.retain_grad()
    final_input = x

    head = model.energy_head
    shift, scale = head.adaLN_modulation(c).chunk(2, dim=1)
    normalized_final, inv_std_final = _layernorm_noaffine_forward(x, head.norm_final.eps)
    x_mod = _modulate(normalized_final, shift, scale)
    token_energies = head.linear(x_mod).squeeze(-1)
    E = token_energies.sum(dim=1)

    cache = TransformerReverseCache(
        c=c.detach(), grid_h=grid_h, grid_w=grid_w, patch_size=patch_size,
        blocks=block_caches,
        final=FinalLayerReverseCache(norm_final=LayerNormReverseCache(
            normalized=normalized_final.detach(), inv_std=inv_std_final.detach())),
        energy=E.detach(),
    )
    return E, cache, block_inputs, final_input, z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    latent_size = args.image_size // 8

    raw = torch.load(args.ckpt, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw
    state_dict = {k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v
                  for k, v in state_dict.items()}

    model = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, ebm="forward-backwards-direct",
    ).double()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[fp64-bisect] load: missing={missing} unexpected={unexpected}")
    model.eval()

    phi = ReverseEqM(model).double()
    registry = ParameterMappingRegistry(model, phi)
    registry.tie_from_forward_()
    print(f"[fp64-bisect] sync error after tie: {registry.compute_sync_error():.3e}")

    B, C = args.batch_size, model.x_embedder.proj.in_channels
    z = torch.randn(B, C, latent_size, latent_size, dtype=torch.float64, requires_grad=True)
    t = torch.rand(B, dtype=torch.float64)
    y = torch.randint(0, args.num_classes, (B,))

    E, cache, block_inputs, final_input, z_ref = build_grad_tracked_cache(model, z, t, y)
    E.sum().backward()

    true_grad_z = z.grad.clone()
    true_grad_final = final_input.grad.clone()
    true_grad_blocks = [bi.grad.clone() for bi in block_inputs]  # true_grad_blocks[i] = dE/d(input to block i)

    # --- manual reverse, instrumented to record u_x at each block boundary ---
    normalized_f = cache.final.norm_final.normalized
    inv_std_f = cache.final.norm_final.inv_std
    Bb, T, D = normalized_f.shape
    from fb_direct.reverse_ops import linear_reverse, modulate_reverse_wrt_norm, layernorm_noaffine_reverse

    shift_f, scale_f = phi.energy_head_adaLN_modulation(cache.c).chunk(2, dim=1)
    u_linear_out = torch.ones(Bb, T, 1, dtype=torch.float64)
    u_xmod_f = linear_reverse(u_linear_out, phi.energy_head_linear.weight)
    u_norm_f = modulate_reverse_wrt_norm(u_xmod_f, scale_f)
    u_x = layernorm_noaffine_reverse(u_norm_f, normalized_f, inv_std_f)

    def relerr(a, b):
        return (a - b).norm().item() / (b.norm().item() + 1e-30)

    print(f"[fp64-bisect] final_layer input grad relerr: {relerr(u_x, true_grad_final):.3e}")

    manual_block_grads = []
    for i, (block, block_cache) in enumerate(zip(reversed(phi.blocks), reversed(cache.blocks))):
        u_x = block.reverse(u_x, block_cache, cache.c)
        manual_block_grads.append(u_x)

    manual_block_grads = list(reversed(manual_block_grads))  # now index-aligned with block_inputs
    for i in range(len(block_inputs)):
        err = relerr(manual_block_grads[i], true_grad_blocks[i])
        print(f"[fp64-bisect] block {i} input grad relerr: {err:.3e}")

    from fb_direct.reverse_ops import patch_embed_reverse
    u_z = patch_embed_reverse(u_x, phi.x_embedder_weight, cache.patch_size, cache.grid_h, cache.grid_w)
    final_err = relerr(u_z, true_grad_z)
    print(f"[fp64-bisect] FINAL field (d/dz) relerr: {final_err:.3e}")
    print(f"[fp64-bisect] {'PASS' if final_err < 1e-8 else 'FAIL'} (threshold 1e-8, FP64)")


if __name__ == "__main__":
    main()
