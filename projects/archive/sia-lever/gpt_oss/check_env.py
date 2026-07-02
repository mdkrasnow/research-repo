#!/usr/bin/env python3
"""Environment check for the gpt-oss-120b / H200 lane. Prints a readiness report; never raises."""

import os
import shutil
import subprocess
import sys


def _try(fn, default="(unavailable)"):
    try:
        return fn()
    except Exception as e:
        return f"{default} ({type(e).__name__}: {e})"


def main():
    print("=" * 60)
    print("SIA-Lever-120B environment check")
    print("=" * 60)
    print(f"python            : {sys.version.split()[0]}")
    print(f"platform          : {sys.platform}")

    def torch_info():
        import torch
        info = [f"torch {torch.__version__}", f"cuda_available={torch.cuda.is_available()}"]
        if torch.cuda.is_available():
            info.append(f"cuda={torch.version.cuda}")
            info.append(f"n_gpu={torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info.append(f"  gpu{i}: {p.name} {p.total_memory/1e9:.0f}GB")
            info.append(f"bf16_supported={torch.cuda.is_bf16_supported()}")
        return "\n".join(info)
    print("torch/cuda        :")
    print("  " + _try(torch_info).replace("\n", "\n  "))

    print("nvidia-smi        :")
    if shutil.which("nvidia-smi"):
        out = _try(lambda: subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15).stdout.strip())
        print("  " + str(out).replace("\n", "\n  "))
    else:
        print("  nvidia-smi not found (CPU-only host)")

    print("packages          :")
    for pkg in ["transformers", "accelerate", "datasets", "peft", "trl", "bitsandbytes",
                "deepspeed", "vllm", "openai", "scipy"]:
        ver = _try(lambda p=pkg: __import__(p).__version__, default="MISSING")
        print(f"  {pkg:14s}: {ver}")

    print("env vars          :")
    for v in ["OPENAI_API_KEY", "NEBIUS_API_KEY", "GPT_OSS_API_KEY", "GPT_OSS_BASE_URL",
              "GPT_OSS_MODEL", "GPT_OSS_MODEL_PATH", "HF_TOKEN", "HF_HOME", "WANDB_API_KEY"]:
        val = os.getenv(v)
        is_secret = ("KEY" in v) or ("TOKEN" in v)
        shown = ("SET" if val else "—") if is_secret else (val or "—")
        print(f"  {v:18s}: {shown}")

    mp = os.getenv("GPT_OSS_MODEL_PATH")
    if mp:
        print(f"local model path  : {mp} ({'exists' if os.path.exists(mp) else 'MISSING'})")

    base = os.getenv("GPT_OSS_BASE_URL")
    if base:
        print(f"endpoint probe    : {base}")
        def probe():
            from openai import OpenAI
            key = os.getenv("GPT_OSS_API_KEY") or os.getenv("NEBIUS_API_KEY") or os.getenv("OPENAI_API_KEY")
            c = OpenAI(api_key=key, base_url=base)
            models = c.models.list()
            return f"OK ({len(list(models.data))} models)"
        print("  " + str(_try(probe)))
    else:
        print("endpoint probe    : GPT_OSS_BASE_URL not set (skipped)")

    print("=" * 60)
    print("Set GPT_OSS_BASE_URL + GPT_OSS_API_KEY (or NEBIUS_API_KEY) + GPT_OSS_MODEL, then run "
          "gpt_oss/smoke_infer.py")


if __name__ == "__main__":
    main()
