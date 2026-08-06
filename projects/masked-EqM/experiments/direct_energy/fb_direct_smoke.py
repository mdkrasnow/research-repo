"""
Real (small-scale) smoke run for the `forward-backwards-direct` training
mode -- exercises the ACTUAL EqM corruption/target pipeline
(transport.Transport.sample / path_sampler.plan / get_ct), the actual
class-conditioning path (LabelEmbedder w/ CFG dropout), and the actual
ForwardBackwardsDirectTrainer training_step / checkpoint / audit /
sampling code paths end to end.

What this smoke run does NOT exercise, and why (documented per Section 15/18
of the spec rather than silently skipped):
  - Real ImageNet-256 pixel data + the SD-VAE encode step: this local
    environment does not have `diffusers` installed and has no ImageNet
    data path configured (repo convention: SLURM/cluster is remote-only,
    see AGENTS.md). Latents are therefore RANDOM TENSORS standing in for
    VAE-encoded latents -- shape/dtype-identical to what
    `vae.encode(x).latent_dist.sample().mul_(0.18215)` would produce, so
    every downstream code path (transport, model, trainer) runs exactly as
    it would on real data; only the *content* of z1 (the data endpoint) is
    synthetic.
  - Mixed precision / multi-GPU DDP: single CPU process here; see
    documentation/forward-backwards-direct.md "Known limitations" and
    train.py's `main_forward_backwards_direct` docstring for the DDP design
    (implemented, not live-tested in this environment).
  - Real ImageNet-256 cluster-scale training: see documentation/
    forward-backwards-direct.md for the exact SLURM command to run once
    cluster access is available.

Run: python experiments/direct_energy/fb_direct_smoke.py
"""
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import EqM_models
from transport import create_transport
from fb_direct import ForwardBackwardsDirectTrainer


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = EqM_models["EqM-S/2"](
        input_size=8, in_channels=4, num_classes=10, learn_sigma=False,
        uncond=True, ebm="forward-backwards-direct",
    ).to(device)
    model.train()

    trainer = ForwardBackwardsDirectTrainer(model, lr=1e-4, max_grad_norm=1.0, device=device)
    coverage = trainer.coverage_report()
    print(f"[1] scalar forward model + reverse network + registry constructed. "
          f"reverse coverage: {coverage['reverse_coverage_pct']:.2f}% "
          f"({coverage['total_trainable_params']:,} trainable params)")
    print(f"    frozen cache_only params: {len(trainer.frozen_cache_only_names)}")
    assert trainer.registry.compute_sync_error() == 0.0
    print("[2] phi tied to theta at construction, sync error == 0")

    # Real EqM corruption pipeline (transport.py), default gaussian/Linear/velocity.
    transport = create_transport("Linear", "velocity", None, 0.0, 0.0, corruption_mode="gaussian")

    num_steps = 40
    batch_size = 4
    losses = []
    grad_norms = []
    t0 = time.time()
    for step in range(num_steps):
        # Stand-in for VAE-encoded ImageNet latents (see module docstring).
        x1 = torch.randn(batch_size, 4, 8, 8, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        loss, diagnostics = trainer.training_step(transport, x1, y)
        losses.append(float(loss))
        grad_norms.append(diagnostics["fb/raw_phi_grad_norm"])

        for name, p in model.named_parameters():
            assert p.grad is None, f"theta.{name}.grad is not None at step {step}"

        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"    step {step + 1:03d}/{num_steps}  loss={diagnostics['fb/loss']:.4f}  "
                f"phi_grad_norm={diagnostics['fb/raw_phi_grad_norm']:.4f}  "
                f"theta_update_norm={diagnostics['fb/theta_update_norm']:.4e}  "
                f"sync_err_post={diagnostics['fb/phi_theta_sync_error_post']:.2e}  "
                f"cos(field,target)={diagnostics['fb/field_target_cosine']:.4f}"
            )
    dt = time.time() - t0
    print(f"[3] {num_steps} training_step() calls completed in {dt:.1f}s "
          f"({num_steps / dt:.2f} steps/s on CPU). theta.grad stayed None every step "
          f"(checked every step, not just once).")
    print(f"    loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}  "
          f"finite: {all(torch.isfinite(torch.tensor(losses)))}")

    x1_audit = torch.randn(batch_size, 4, 8, 8, device=device)
    y_audit = torch.randint(0, 10, (batch_size,), device=device)
    t_audit, x0_audit, x1_audit = transport.sample(x1_audit)
    t_audit = t_audit.to(x1_audit)
    t_audit, xt_audit, _ = transport.path_sampler.plan(t_audit, x0_audit, x1_audit)
    audit = trainer.exact_field_audit(xt_audit, t_audit, y_audit)
    print(f"[4] exact-field audit (diagnostic-only, torch.autograd.grad create_graph=False): {audit}")
    assert audit["fb/audit_mean_rel_error"] < 1e-3, audit

    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "fb_direct_smoke.pt")
        torch.save({**trainer.state_dict(), "step": num_steps, "epoch": 1}, ckpt_path)
        print(f"[5] checkpoint saved to {ckpt_path} "
              f"({os.path.getsize(ckpt_path) / 1e6:.1f} MB)")

        model2 = EqM_models["EqM-S/2"](
            input_size=8, in_channels=4, num_classes=10, learn_sigma=False,
            uncond=True, ebm="forward-backwards-direct",
        ).to(device)
        trainer2 = ForwardBackwardsDirectTrainer(model2, lr=1e-4, device=device)
        state = torch.load(ckpt_path, map_location="cpu")
        trainer2.load_state_dict(state)
        max_param_diff = max(
            (p1 - p2).abs().max().item()
            for p1, p2 in zip(model.parameters(), model2.parameters())
        )
        assert max_param_diff == 0.0, max_param_diff
        assert trainer2.registry.compute_sync_error() < 1e-6
        print(f"[6] checkpoint resumed into a fresh trainer: theta matches exactly "
              f"(max diff {max_param_diff}), phi re-synced "
              f"(sync error {trainer2.registry.compute_sync_error():.2e}), "
              f"step_count={trainer2.step_count}, "
              f"optimizer moments restored ({len(trainer2.optimizer.state_dict()['state'])} tensors)")

    model.eval()
    with torch.no_grad():
        z = torch.randn(2, 4, 8, 8, device=device)
        y = torch.randint(0, 10, (2,), device=device)
        t = torch.zeros(2, device=device)
        for _ in range(5):
            with torch.set_grad_enabled(True):
                field = model(z, t, y, train=False)  # true scalar-energy inference path, create_graph=False
            field = field.detach()
            z = (z + 0.01 * field).detach()
            assert torch.isfinite(z).all()
    print(f"[7] sampling from theta (true scalar-energy forward, autograd.grad "
          f"create_graph=False -- allowed at inference per spec invariant #10): "
          f"5 Euler steps, final sample finite={torch.isfinite(z).all().item()}, "
          f"shape={tuple(z.shape)}")

    print("\nFORWARD-BACKWARDS-DIRECT SMOKE RUN: ALL STAGES PASSED")


if __name__ == "__main__":
    main()
