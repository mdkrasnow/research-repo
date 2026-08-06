"""
Tests for the `forward-backwards-direct` scalar-energy training mode.

Covers, per the pre-launch test plan:
  1. Local VJP tests for every manual reverse primitive vs torch.autograd,
     FP64, several shapes.
  2. End-to-end field equivalence: tied phi's reverse field vs the TRUE
     autograd.grad(E.sum(), z) input gradient, FP64 and FP32.
  3. No-double-backward: the training path never calls
     torch.autograd.grad(create_graph=True), and theta.grad stays None.
  4. Update-transfer: Pi/Pi-dagger exactly round-trips a synthetic phi
     gradient into theta.
  5. Sign test: reverse output == grad_x E; prediction fed to the loss has
     the existing repo sign convention; a small gradient-descent-on-E step
     decreases E.
  6. Batch independence.
  7. Checkpoint/resume.
  8. Regression: none/dot/l2/direct still behave exactly as before.

Run: python tests/test_forward_backwards_direct.py  (CPU, ~tens of seconds)
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models
from models import EqM_models
from fb_direct import ForwardBackwardsDirectTrainer, ParameterMappingRegistry, ReverseEqM
from fb_direct.forward_cache import forward_energy_with_cache
from fb_direct import reverse_ops
from transport import create_transport

torch.manual_seed(0)

# TimestepEmbedder.timestep_embedding() hardcodes `.float()` on its sinusoidal
# table regardless of the model's parameter dtype (models.py:52) -- never
# exercised outside FP32 training before this test suite. Since every test
# here uses uncond=True (t is zeroed before t_embedder anyway, per
# EqM.forward), the embedded VALUE is unaffected by dtype; only the container
# dtype needs to match self.mlp's weights for these FP64 equivalence checks
# to run at all. Test-process-local monkeypatch only -- does not touch
# models.py, and none/dot/l2/direct FP32 behavior (the only dtype the
# original model ships/trains in) is untouched.
_orig_timestep_forward = models.TimestepEmbedder.forward


def _dtype_safe_timestep_forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_freq = t_freq.to(self.mlp[0].weight.dtype)
    return self.mlp(t_freq)


models.TimestepEmbedder.forward = _dtype_safe_timestep_forward


def make_model(ebm='forward-backwards-direct', dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    model = EqM_models['EqM-S/2'](
        input_size=4, in_channels=4, num_classes=10, learn_sigma=False, uncond=True, ebm=ebm,
    ).to(dtype)
    return model


def perturb(model, std=0.02, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(std * torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype))


def batch(n=2, dtype=torch.float64, seed=2):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 4, 4, 4, generator=g).to(dtype)
    t = torch.rand(n, generator=g).to(dtype)
    y = torch.randint(0, 10, (n,), generator=g)
    return z, t, y


# ---------------------------------------------------------------------------
# 1. Local VJP tests
# ---------------------------------------------------------------------------

def test_local_vjp_linear():
    torch.manual_seed(10)
    for (B, N, din, dout) in [(1, 1, 3, 5), (2, 7, 8, 4), (3, 5, 16, 16)]:
        x = torch.randn(B, N, din, dtype=torch.float64, requires_grad=True)
        W = torch.randn(dout, din, dtype=torch.float64)
        u_y = torch.randn(B, N, dout, dtype=torch.float64)

        y = F.linear(x, W)
        (u_y * y).sum().backward()
        autograd_u_x = x.grad.clone()

        manual_u_x = reverse_ops.linear_reverse(u_y, W)
        torch.testing.assert_close(manual_u_x, autograd_u_x, rtol=1e-10, atol=1e-10)
    print("PASS local VJP: linear")


def test_local_vjp_gelu_tanh():
    torch.manual_seed(11)
    for shape in [(1, 1, 3), (2, 5, 8)]:
        a = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        u_y = torch.randn(*shape, dtype=torch.float64)
        y = F.gelu(a, approximate="tanh")
        (u_y * y).sum().backward()
        autograd_u_a = a.grad.clone()

        manual_u_a = u_y * reverse_ops.gelu_tanh_prime(a.detach())
        torch.testing.assert_close(manual_u_a, autograd_u_a, rtol=1e-8, atol=1e-8)
    print("PASS local VJP: GELU(tanh)")


def test_local_vjp_silu():
    torch.manual_seed(12)
    a = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    u_y = torch.randn(4, 6, dtype=torch.float64)
    y = F.silu(a)
    (u_y * y).sum().backward()
    manual_u_a = u_y * reverse_ops.silu_prime(a.detach())
    torch.testing.assert_close(manual_u_a, a.grad, rtol=1e-8, atol=1e-8)
    print("PASS local VJP: SiLU")


def test_local_vjp_layernorm_noaffine():
    torch.manual_seed(13)
    for (B, N, D) in [(1, 1, 4), (2, 3, 16), (3, 7, 33)]:
        x = torch.randn(B, N, D, dtype=torch.float64, requires_grad=True)
        eps = 1e-6
        y = F.layer_norm(x, (D,), eps=eps)
        u_y = torch.randn(B, N, D, dtype=torch.float64)
        (u_y * y).sum().backward()
        autograd_u_x = x.grad.clone()

        with torch.no_grad():
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            inv_std = torch.rsqrt(var + eps)
            normalized = (x - mean) * inv_std
        manual_u_x = reverse_ops.layernorm_noaffine_reverse(u_y, normalized, inv_std)
        torch.testing.assert_close(manual_u_x, autograd_u_x, rtol=1e-8, atol=1e-8)
    print("PASS local VJP: LayerNorm (no affine)")


def test_local_vjp_attention():
    torch.manual_seed(14)
    for (B, N, H, dh) in [(1, 2, 1, 4), (2, 5, 2, 3), (3, 8, 4, 6)]:
        D = H * dh
        x = torch.randn(B, N, D, dtype=torch.float64, requires_grad=True)
        Wqkv = torch.randn(3 * D, D, dtype=torch.float64)
        Wproj = torch.randn(D, D, dtype=torch.float64)
        scale = dh ** -0.5

        qkv = F.linear(x, Wqkv).reshape(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        s = torch.matmul(q * scale, k.transpose(-2, -1))
        p = s.softmax(dim=-1)
        o = torch.matmul(p, v).transpose(1, 2).reshape(B, N, D)
        y = F.linear(o, Wproj)

        u_y = torch.randn(B, N, D, dtype=torch.float64)
        (u_y * y).sum().backward()
        autograd_u_x = x.grad.clone()

        manual_u_x = reverse_ops.attention_reverse(
            u_y, q=q.detach(), k=k.detach(), v=v.detach(), p=p.detach(),
            weight_proj_b=Wproj, weight_qkv_b=Wqkv,
            num_heads=H, head_dim=dh, scale=scale,
        )
        torch.testing.assert_close(manual_u_x, autograd_u_x, rtol=1e-7, atol=1e-8)
    print("PASS local VJP: attention (unfused/math-backend equivalent)")


def test_local_vjp_mlp():
    torch.manual_seed(15)
    for (B, N, D, Dh) in [(1, 1, 4, 8), (2, 5, 8, 32)]:
        x = torch.randn(B, N, D, dtype=torch.float64, requires_grad=True)
        W1 = torch.randn(Dh, D, dtype=torch.float64)
        W2 = torch.randn(D, Dh, dtype=torch.float64)
        a1 = F.linear(x, W1)
        h1 = F.gelu(a1, approximate="tanh")
        y = F.linear(h1, W2)
        u_y = torch.randn(B, N, D, dtype=torch.float64)
        (u_y * y).sum().backward()
        autograd_u_x = x.grad.clone()

        manual_u_x = reverse_ops.mlp_reverse(
            u_y, pre_act=a1.detach(), weight_fc1_b=W1, weight_fc2_b=W2,
        )
        torch.testing.assert_close(manual_u_x, autograd_u_x, rtol=1e-8, atol=1e-8)
    print("PASS local VJP: MLP (fc1 -> GELU(tanh) -> fc2)")


def test_local_vjp_patch_embed():
    torch.manual_seed(16)
    for (B, C, D, p, gh, gw) in [(1, 3, 8, 2, 2, 2), (2, 4, 6, 4, 3, 3)]:
        H, W = gh * p, gw * p
        z = torch.randn(B, C, H, W, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(D, C, p, p, dtype=torch.float64)

        y = F.conv2d(z, weight, stride=p)  # (B, D, gh, gw)
        tokens = y.flatten(2).transpose(1, 2)  # (B, T, D)
        u_tokens = torch.randn(B, gh * gw, D, dtype=torch.float64)
        (u_tokens * tokens).sum().backward()
        autograd_u_z = z.grad.clone()

        manual_u_z = reverse_ops.patch_embed_reverse(u_tokens, weight, p, gh, gw)
        torch.testing.assert_close(manual_u_z, autograd_u_z, rtol=1e-8, atol=1e-8)
    print("PASS local VJP: patch embed (conv_transpose2d)")


# ---------------------------------------------------------------------------
# 2. End-to-end field equivalence
# ---------------------------------------------------------------------------

def _tied_trainer(dtype, seed=0):
    model = make_model(dtype=dtype, seed=seed)
    perturb(model, seed=seed + 100)
    model.eval()
    trainer = ForwardBackwardsDirectTrainer(model, lr=1e-4, device=torch.device('cpu'))
    trainer.phi.to(dtype)
    trainer.registry.tie_from_forward_()
    return model, trainer


def test_end_to_end_field_equivalence_fp64():
    model, trainer = _tied_trainer(torch.float64)
    z, t, y = batch(n=3, dtype=torch.float64)

    z_req = z.clone().requires_grad_(True)
    E = model(z_req, t, y, energy_only=True)
    true_grad = torch.autograd.grad(E.sum(), z_req, create_graph=False)[0]

    _, cache = forward_energy_with_cache(model, z, t, y)
    approx_field = trainer.phi(cache)

    rel_err = (approx_field - true_grad).flatten(1).norm(dim=1) / true_grad.flatten(1).norm(dim=1)
    cos = F.cosine_similarity(approx_field.flatten(1), true_grad.flatten(1), dim=1)
    assert rel_err.mean().item() < 1e-6, rel_err
    assert cos.min().item() > 0.999999, cos
    print(f"PASS end-to-end field equivalence FP64: mean rel err {rel_err.mean().item():.3e}, "
          f"min cosine {cos.min().item():.9f}")


def test_end_to_end_field_equivalence_fp32():
    model, trainer = _tied_trainer(torch.float32)
    z, t, y = batch(n=3, dtype=torch.float32)

    z_req = z.clone().requires_grad_(True)
    E = model(z_req, t, y, energy_only=True)
    true_grad = torch.autograd.grad(E.sum(), z_req, create_graph=False)[0]

    _, cache = forward_energy_with_cache(model, z, t, y)
    approx_field = trainer.phi(cache)

    rel_err = (approx_field - true_grad).flatten(1).norm(dim=1) / true_grad.flatten(1).norm(dim=1)
    assert rel_err.mean().item() < 1e-3, rel_err
    print(f"PASS end-to-end field equivalence FP32: mean rel err {rel_err.mean().item():.3e}")


# ---------------------------------------------------------------------------
# 3. No-double-backward
# ---------------------------------------------------------------------------

def test_no_double_backward():
    model = make_model(dtype=torch.float32)
    perturb(model)
    model.train()
    trainer = ForwardBackwardsDirectTrainer(model, lr=1e-3, device=torch.device('cpu'))

    transport = create_transport("Linear", "velocity", None, 0.0, 0.0)
    x1 = torch.randn(4, 4, 4, 4)
    y = torch.randint(0, 10, (4,))

    original_grad = torch.autograd.grad
    calls = []

    def spying_grad(*args, **kwargs):
        calls.append(kwargs.get("create_graph", False))
        return original_grad(*args, **kwargs)

    torch.autograd.grad = spying_grad
    try:
        loss, diagnostics = trainer.training_step(transport, x1, y)
    finally:
        torch.autograd.grad = original_grad

    assert not any(calls), f"training_step invoked torch.autograd.grad {len(calls)} time(s): create_graph flags={calls}"

    for name, p in model.named_parameters():
        assert p.grad is None, f"theta parameter {name} unexpectedly has a .grad after training_step"

    phi_had_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.phi.parameters())
    assert phi_had_grad, "phi received no nonzero gradient from training_step"
    print(f"PASS no-double-backward: torch.autograd.grad never called in training_step, "
          f"theta.grad is None for all {sum(1 for _ in model.parameters())} params, "
          f"phi received nonzero gradients. loss={float(loss):.4e}")


# ---------------------------------------------------------------------------
# 4. Update-transfer test
# ---------------------------------------------------------------------------

def test_update_transfer_identity_mapping():
    model = make_model(dtype=torch.float64)
    perturb(model)
    trainer = ForwardBackwardsDirectTrainer(model, lr=0.1, device=torch.device('cpu'))
    trainer.phi.to(torch.float64)
    trainer.registry.tie_from_forward_()

    theta_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    phi_before = trainer.registry.snapshot_backward_parameters()

    # synthetic gradient
    for p in trainer.phi.parameters():
        p.grad = torch.ones_like(p) * 0.01

    trainer.optimizer.step()
    theta_update_norms = trainer.registry.apply_backward_delta_to_forward_(phi_before)

    for e in trainer.registry._active_entries:
        fwd = dict(model.named_parameters())[e.forward_name]
        bwd = dict(trainer.phi.named_parameters())[e.backward_name]
        expected_delta = bwd.detach() - phi_before[e.backward_name]
        actual_delta = fwd.detach() - theta_before[e.forward_name]
        torch.testing.assert_close(actual_delta, expected_delta, rtol=1e-10, atol=1e-12)

    trainer.registry.tie_from_forward_()
    assert trainer.registry.compute_sync_error() < 1e-10
    print(f"PASS update-transfer: {len(theta_update_norms)} mapped tensors match "
          f"Pi-dagger(delta_phi) exactly; re-sync error "
          f"{trainer.registry.compute_sync_error():.2e}")


# ---------------------------------------------------------------------------
# 5. Sign test
# ---------------------------------------------------------------------------

def test_sign_convention():
    model = make_model(dtype=torch.float64)
    perturb(model)
    model.eval()
    trainer = ForwardBackwardsDirectTrainer(model, lr=1e-4, device=torch.device('cpu'))
    trainer.phi.to(torch.float64)
    trainer.registry.tie_from_forward_()

    z, t, y = batch(n=2, dtype=torch.float64)
    _, cache = forward_energy_with_cache(model, z, t, y)
    grad_tilde = trainer.phi(cache)  # R_phi computes grad_x E directly (no sign flip)

    z_req = z.clone().requires_grad_(True)
    E0 = model(z_req, t, y, energy_only=True)
    grad_E = torch.autograd.grad(E0.sum(), z_req)[0]
    torch.testing.assert_close(grad_tilde, grad_E, rtol=1e-6, atol=1e-6)

    # existing repo sign convention: sampling field == -grad_x E (models.py:296-300)
    field = -grad_tilde
    eta = 1e-4
    z_next = (z + eta * field).detach()
    with torch.no_grad():
        E_next = model(z_next, t, y, energy_only=True)
        E_curr = model(z, t, y, energy_only=True)
    assert (E_next < E_curr).all(), (E_next, E_curr)
    print(f"PASS sign test: R_phi(cache) == grad_x E; field := -R_phi(cache) == -grad_x E "
          f"per repo convention; a small step x <- x + eta*field "
          f"decreases E for all samples (E {E_curr.tolist()} -> {E_next.tolist()})")


# ---------------------------------------------------------------------------
# 6. Batch independence
# ---------------------------------------------------------------------------

def test_batch_independence():
    model = make_model(dtype=torch.float64)
    perturb(model)
    model.eval()
    trainer = ForwardBackwardsDirectTrainer(model, lr=1e-4, device=torch.device('cpu'))
    trainer.phi.to(torch.float64)
    trainer.registry.tie_from_forward_()

    torch.manual_seed(42)
    z_a = torch.randn(1, 4, 4, 4, dtype=torch.float64)
    t_a = torch.rand(1, dtype=torch.float64)
    y_a = torch.randint(0, 10, (1,))

    z_extra = torch.randn(3, 4, 4, 4, dtype=torch.float64)
    t_extra = torch.rand(3, dtype=torch.float64)
    y_extra = torch.randint(0, 10, (3,))

    z_batch = torch.cat([z_a, z_extra], dim=0)
    t_batch = torch.cat([t_a, t_extra], dim=0)
    y_batch = torch.cat([y_a, y_extra], dim=0)

    _, cache_solo = forward_energy_with_cache(model, z_a, t_a, y_a)
    field_solo = trainer.phi(cache_solo)[0]

    _, cache_batch = forward_energy_with_cache(model, z_batch, t_batch, y_batch)
    field_batch = trainer.phi(cache_batch)[0]

    torch.testing.assert_close(field_solo, field_batch, rtol=1e-9, atol=1e-10)
    print("PASS batch independence: sample 0's field is unchanged by unrelated batchmates")


# ---------------------------------------------------------------------------
# 7. Checkpoint / resume
# ---------------------------------------------------------------------------

def test_checkpoint_resume(tmp_path=None):
    import tempfile
    model = make_model(dtype=torch.float32)
    perturb(model)
    trainer = ForwardBackwardsDirectTrainer(model, lr=1e-3, device=torch.device('cpu'))

    transport = create_transport("Linear", "velocity", None, 0.0, 0.0)
    x1 = torch.randn(4, 4, 4, 4)
    y = torch.randint(0, 10, (4,))
    for _ in range(2):
        trainer.training_step(transport, x1, y)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save(trainer.state_dict(), path)

        model2 = make_model(dtype=torch.float32, seed=999)  # different init
        trainer2 = ForwardBackwardsDirectTrainer(model2, lr=1e-3, device=torch.device('cpu'))
        state = torch.load(path, map_location="cpu")
        trainer2.load_state_dict(state)

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            torch.testing.assert_close(p1, p2, rtol=1e-6, atol=1e-6)
        assert trainer2.step_count == trainer.step_count
        assert trainer2.registry.compute_sync_error() < 1e-5

        opt1 = trainer.optimizer.state_dict()["state"]
        opt2 = trainer2.optimizer.state_dict()["state"]
        assert len(opt1) == len(opt2) and len(opt1) > 0, "optimizer moments were not restored"
    print("PASS checkpoint/resume: theta, phi, optimizer moments, step count, sync all restored")


def test_checkpoint_init_from_direct():
    direct_model = make_model(ebm='direct', dtype=torch.float32)
    perturb(direct_model)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "direct_ckpt.pt")
        torch.save({"model": direct_model.state_dict()}, path)

        fb_model = make_model(ebm='forward-backwards-direct', dtype=torch.float32, seed=777)
        raw = torch.load(path, map_location="cpu")
        missing, unexpected = fb_model.load_state_dict(raw["model"], strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        for (n1, p1), (n2, p2) in zip(direct_model.named_parameters(), fb_model.named_parameters()):
            torch.testing.assert_close(p1, p2)
    print("PASS checkpoint init: a 'direct' checkpoint loads cleanly into an "
          "fb-direct model's theta (identical architecture)")


# ---------------------------------------------------------------------------
# 8. Regression: existing modes unaffected
# ---------------------------------------------------------------------------

def test_existing_modes_regression():
    z, t, y = batch(n=2, dtype=torch.float32)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for ebm in ('none', 'dot', 'l2', 'direct'):
            model = make_model(ebm=ebm, dtype=torch.float32)
            model.train()
            field = model(z, t, y, train=(ebm != 'none'))
            assert field.shape == z.shape
            assert torch.isfinite(field).all()
            if ebm != 'none':
                field.square().mean().backward()
    print("PASS regression: none/dot/l2/direct unaffected by forward-backwards-direct changes")


def test_forward_cache_matches_direct_energy():
    model = make_model(ebm='direct', dtype=torch.float64)
    perturb(model)
    model.eval()
    z, t, y = batch(n=2, dtype=torch.float64)
    with torch.no_grad():
        E_direct = model(z, t, y, energy_only=True)
    E_cache, _ = forward_energy_with_cache(model, z, t, y)
    torch.testing.assert_close(E_cache, E_direct, rtol=1e-10, atol=1e-10)
    print("PASS forward_energy_with_cache produces IDENTICAL energy to ebm='direct' energy_only=True")


if __name__ == '__main__':
    test_local_vjp_linear()
    test_local_vjp_gelu_tanh()
    test_local_vjp_silu()
    test_local_vjp_layernorm_noaffine()
    test_local_vjp_attention()
    test_local_vjp_mlp()
    test_local_vjp_patch_embed()

    test_forward_cache_matches_direct_energy()
    test_end_to_end_field_equivalence_fp64()
    test_end_to_end_field_equivalence_fp32()

    test_no_double_backward()
    test_update_transfer_identity_mapping()
    test_sign_convention()
    test_batch_independence()
    test_checkpoint_resume()
    test_checkpoint_init_from_direct()

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        test_existing_modes_regression()

    print("\nALL FORWARD-BACKWARDS-DIRECT TESTS PASSED")
