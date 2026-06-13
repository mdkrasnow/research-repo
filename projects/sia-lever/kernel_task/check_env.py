#!/usr/bin/env python3
"""Readiness check for the kernel-task GPU lane. Never raises; prints what's missing."""

import os
import sys


def _try(fn, default="(unavailable)"):
    try:
        return fn()
    except Exception as e:                              # noqa: BLE001
        return f"{default} ({type(e).__name__}: {e})"


def main():
    print("=" * 56)
    print("kernel_task environment check")
    print("=" * 56)
    print(f"python : {sys.version.split()[0]}")

    def torch_info():
        import torch
        s = [f"torch {torch.__version__}", f"cuda_available={torch.cuda.is_available()}"]
        if torch.cuda.is_available():
            s.append(f"device={torch.cuda.get_device_name(0)}  n={torch.cuda.device_count()}")
            s.append(f"bf16={torch.cuda.is_bf16_supported()}")
        return " | ".join(s)
    print("torch  :", _try(torch_info))

    def triton_info():
        import triton
        return f"triton {triton.__version__}"
    print("triton :", _try(triton_info, default="MISSING (GPU kernels need triton)"))

    for v in ["GPT_OSS_BASE_URL", "GPT_OSS_MODEL", "GPT_OSS_API_KEY", "NEBIUS_API_KEY"]:
        val = os.getenv(v)
        is_secret = "KEY" in v
        print(f"  {v:18s}: {('SET' if val else '—') if is_secret else (val or '—')}")

    # CPU stub always works -> the loop is testable now
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import lever_loop as L
        r = L.run_episode("stub", "selector")
        print(f"cpu stub selector: correct={r['final_heldout_correct']} (loop wired OK)")
    except Exception as e:                              # noqa: BLE001
        print(f"cpu stub: FAILED ({type(e).__name__}: {e})")

    print("=" * 56)
    print("GPU run: python kernel_task/run.py --endpoint --device cuda --shape 128 128 32 "
          "--model $GPT_OSS_MODEL --base-url $GPT_OSS_BASE_URL")


if __name__ == "__main__":
    main()
