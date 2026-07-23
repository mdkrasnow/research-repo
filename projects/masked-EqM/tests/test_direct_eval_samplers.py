"""Regression test: every direct-capable evaluation sampler is no-grad safe."""
import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import EqM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model():
    instance = EqM(input_size=4, patch_size=2, in_channels=4, hidden_size=32,
                   depth=2, num_heads=4, mlp_ratio=2.0, class_dropout_prob=0.0,
                   num_classes=10, learn_sigma=False, uncond=True, ebm="direct").to(DEVICE)
    with torch.no_grad():
        for parameter in instance.parameters():
            if parameter.requires_grad:
                parameter.add_(0.02 * torch.randn_like(parameter))
    instance.eval()
    for parameter in instance.parameters():
        parameter.requires_grad_(False)
    return instance


def main():
    instance = model()
    z = torch.randn(2, 4, 4, 4, device=DEVICE)
    y = torch.randint(0, 10, (2,), device=DEVICE)
    modules = [
        "eval_masked_recovery", "eval_blur_recovery", "eval_downsample_recovery",
        "eval_fourier_recovery", "eval_generalization", "eval_fid",
    ]
    with torch.no_grad():
        for name in modules:
            module = importlib.import_module(name)
            if name == "eval_generalization":
                result = module.gd_recover(instance, z, y, 3, 0.01)
            else:
                result = module.gd_recover(instance, z, y, 3, 0.01, "gd", 0.3) if hasattr(module, "gd_recover") else module.gd_sample(instance, z, y, 3, 0.01, "gd", 0.3)
            assert torch.isfinite(result).all(), name
            assert not result.requires_grad, name
            print(f"PASS {name}: direct sampler is finite and detached")


if __name__ == "__main__":
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        main()
