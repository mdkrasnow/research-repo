"""
Batch-size invariance test for the sampler gradient normalisation.

Pass condition: per-sample gradient magnitude must be the same (within tolerance)
regardless of whether the forward pass sees B=256 or B=2048 samples.

The test calls DiffusionWrapper.forward directly (return_both=True) so that
clamping and backtracking in opt_step do not mask the normalisation effect.

Usage (CPU, no cluster needed):
    cd projects/ired
    python tests/test_sampler_invariance.py

Expected output with grad_norm_ref fixed (new behaviour):
    PASS: mean |grad| ratio (B256/B2048) = 1.000 ± small noise

Expected output with legacy grad / B (old behaviour):
    FAIL: mean |grad| ratio (B256/B2048) ≈ 8.0   (B=256 grads 8× larger)
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
ired_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ired_dir))

from models import EBM, DiffusionWrapper


def run_test(grad_norm_ref, label, atol=0.1):
    """
    Run invariance check for a given grad_norm_ref setting.

    Calls DiffusionWrapper.forward directly to get raw gradients (bypassing
    opt_step's clamping and backtracking which would mask the normalisation
    difference).  Checks that the per-sample gradient magnitude for sample 0
    is the same whether it is evaluated in a B=256 or a B=2048 forward pass.

    Returns True if the test passes.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    INP_DIM = 400        # 20×20 matrix
    OUT_DIM = 400

    # Build model
    ebm = EBM(inp_dim=INP_DIM, out_dim=OUT_DIM, is_ebm=True, use_scalar_energy=False)
    model = DiffusionWrapper(ebm, grad_norm_ref=grad_norm_ref)
    model.eval()

    t_val = 5  # middle timestep

    # Small batch: B=256 unique samples
    B_small = 256
    torch.manual_seed(0)
    inp_small = torch.randn(B_small, INP_DIM)
    img_small = torch.randn(B_small, OUT_DIM)
    t_small   = torch.full((B_small,), t_val, dtype=torch.long)

    with torch.enable_grad():
        grad_small = model(inp_small, img_small.clone(), t_small)  # [B_small, OUT_DIM]

    # Large batch: first B_small entries identical to above, rest are NEW independent
    # samples.  Using copies (repeat) would cause autograd to accumulate gradients
    # from duplicates, which accidentally cancels the normalisation bug.
    B_large = 2048
    torch.manual_seed(1)
    inp_tail = torch.randn(B_large - B_small, INP_DIM)
    img_tail = torch.randn(B_large - B_small, OUT_DIM)
    inp_large = torch.cat([inp_small, inp_tail], dim=0)
    img_large = torch.cat([img_small, img_tail], dim=0)
    t_large   = torch.full((B_large,), t_val, dtype=torch.long)

    with torch.enable_grad():
        grad_large = model(inp_large, img_large.clone(), t_large)  # [B_large, OUT_DIM]

    # Compare per-sample gradient norms for the shared B_small samples
    gnorm_small = grad_small.norm(dim=1)                    # [B_small]
    gnorm_large = grad_large[:B_small].norm(dim=1)          # [B_small]

    ratio = (gnorm_small / (gnorm_large + 1e-30)).mean().item()

    print(f"\n[{label}]")
    print(f"  mean |grad| (B=256):  {gnorm_small.mean().item():.3e}")
    print(f"  mean |grad| (B=2048): {gnorm_large.mean().item():.3e}")
    print(f"  ratio (B256/B2048):   {ratio:.4f}  (target: 1.0)")

    ok = abs(ratio - 1.0) < atol

    if ok:
        print(f"  PASS")
    else:
        print(f"  FAIL: ratio {ratio:.3f} outside [{1-atol:.2f}, {1+atol:.2f}]")

    return ok


if __name__ == '__main__':
    print("=" * 60)
    print("Sampler batch-size invariance test")
    print("=" * 60)

    # Legacy behaviour: grad / energy.shape[0]  — should FAIL with ratio ≈ 8
    legacy_pass = run_test(grad_norm_ref=None, label="LEGACY  grad/B (should FAIL)")

    # Fixed reference: grad / 2048  — should PASS with ratio ≈ 1
    fixed_pass = run_test(grad_norm_ref=2048, label="FIXED   grad/2048 (should PASS)")

    print("\n" + "=" * 60)
    print("Summary")
    print("  Legacy (None):   " + ("PASS (unexpected)" if legacy_pass else "FAIL (expected)"))
    print("  Fixed  (2048):   " + ("PASS" if fixed_pass else "FAIL"))

    if not fixed_pass:
        print("\nFixed mode failed — grad_norm_ref not taking effect.")
        sys.exit(1)
    else:
        print("\nFixed mode passes. Batch-size invariance is restored.")
        sys.exit(0)
