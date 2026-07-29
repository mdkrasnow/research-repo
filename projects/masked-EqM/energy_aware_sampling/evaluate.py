"""Run the sign gate or small frozen-checkpoint Armijo smoke on a fixed bank."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from core import armijo_sample, fixed_sample, replay_sample, sign_audit
from models import EqM_models


def load_model(checkpoint: str, ebm: str, device: torch.device):
    model = EqM_models["EqM-B/2"](input_size=32, num_classes=1000, uncond=True, ebm=ebm).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["ema"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def serializable(stats):
    return {key: (value.tolist() if torch.is_tensor(value) else value) for key, value in stats.items()}


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    x0 = torch.randn(args.samples, 4, 32, 32, device=device)
    y = torch.arange(args.samples, device=device) % 1000
    output = {"mode": args.mode, "seed": args.seed, "samples": args.samples, "steps": args.steps,
              "initial_step": args.initial_step, "checkpoints": {"direct": args.direct, "dot": args.dot}}
    for ebm, checkpoint in (("direct", args.direct), ("dot", args.dot)):
        model = load_model(checkpoint, ebm, device)
        audit_chunks = []
        for start_index in range(0, args.samples, args.batch_size):
            end_index = min(args.samples, start_index + args.batch_size)
            audit_chunks.append(sign_audit(
                model, x0[start_index:end_index],
                torch.zeros(end_index - start_index, device=device), y[start_index:end_index],
            ))
        audit = {
            "cosine_min": min(item["cosine_min"] for item in audit_chunks),
            "cosine_mean": sum(item["cosine_mean"] for item in audit_chunks) / len(audit_chunks),
            "relative_error_max": max(item["relative_error_max"] for item in audit_chunks),
            "relative_error_mean": sum(item["relative_error_mean"] for item in audit_chunks) / len(audit_chunks),
            "energy_error_max": max(item["energy_error_max"] for item in audit_chunks),
        }
        audit["pass"] = bool(audit["cosine_min"] > 0.999 and audit["relative_error_max"] < 1e-4
                             and audit["energy_error_max"] < 1e-4)
        output[ebm] = {"sign_audit": audit}
        if args.mode == "smoke" and audit["pass"]:
            start = time.perf_counter()
            fixed, fixed_stats = fixed_sample(model, x0, y, args.steps, args.initial_step)
            armijo, armijo_stats = armijo_sample(model, x0, y, args.steps, args.initial_step)
            replay, replay_stats = replay_sample(model, x0, y, armijo_stats["accepted_step_median_by_iteration"])
            output[ebm].update({
                "fixed": serializable(fixed_stats), "armijo": serializable(armijo_stats),
                "replay": serializable(replay_stats),
                "finite": {"fixed": bool(torch.isfinite(fixed).all()), "armijo": bool(torch.isfinite(armijo).all()),
                             "replay": bool(torch.isfinite(replay).all())},
                "wall_seconds": time.perf_counter() - start,
            })
        del model
        torch.cuda.empty_cache()
    output["pass"] = all(output[name]["sign_audit"]["pass"] for name in ("direct", "dot"))
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"pass": output["pass"], "output": str(path)}, indent=2))
    if not output["pass"]:
        raise SystemExit("sign audit failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", required=True); parser.add_argument("--dot", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--mode", choices=("audit", "smoke"), default="audit")
    parser.add_argument("--samples", type=int, default=32); parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--initial-step", type=float, default=0.0017); parser.add_argument("--seed", type=int, default=20260729)
    main(parser.parse_args())
