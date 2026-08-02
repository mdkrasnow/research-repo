"""Layerwise model, EMA, and Adam-state changes between adjacent checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import EqM_models


def group(name: str) -> str:
    if name.startswith("energy_head"):
        return "energy_head"
    if name.startswith("t_embedder"):
        return "t_embedder"
    if name.startswith("x_embedder"):
        return "x_embedder"
    if name.startswith("y_embedder"):
        return "y_embedder"
    if name.startswith("blocks."):
        return ".".join(name.split(".")[:2])
    return name.split(".")[0]


def accumulate(record, key, value):
    record[key] = record.get(key, 0.0) + float(value.detach().double().square().sum())


def compare(left_path, right_path, parameter_names):
    left = torch.load(left_path, map_location="cpu")
    right = torch.load(right_path, map_location="cpu")
    result = {"left": left_path, "right": right_path,
              "left_step": left.get("step"), "right_step": right.get("step"), "groups": {}}
    for weight_name in ("model", "ema"):
        for name, left_value in left[weight_name].items():
            rec = result["groups"].setdefault(group(name), {})
            right_value = right[weight_name][name]
            accumulate(rec, f"{weight_name}_left_sq", left_value)
            accumulate(rec, f"{weight_name}_right_sq", right_value)
            accumulate(rec, f"{weight_name}_delta_sq", right_value - left_value)
    # Adam state is indexed consistently because architecture/optimizer are unchanged.
    for key, left_state in left["opt"]["state"].items():
        right_state = right["opt"]["state"][key]
        name = parameter_names[int(key)]
        rec = result["groups"].setdefault(group(name), {})
        for moment in ("exp_avg", "exp_avg_sq"):
            if moment in left_state and moment in right_state:
                accumulate(rec, f"adam_{moment}_left_sq", left_state[moment])
                accumulate(rec, f"adam_{moment}_right_sq", right_state[moment])
                accumulate(rec, f"adam_{moment}_delta_sq", right_state[moment] - left_state[moment])
    for rec in result["groups"].values():
        for key in list(rec):
            if key.endswith("_sq"):
                rec[key[:-3] + "_norm"] = math.sqrt(rec.pop(key))
    del left, right
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="EqM-B/2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    args = parser.parse_args()
    model = EqM_models[args.model](input_size=args.image_size // 8,
                                  num_classes=args.num_classes,
                                  uncond=True, ebm="direct")
    parameter_names = [name for name, _ in model.named_parameters()]
    del model
    results = [compare(a, b, parameter_names) for a, b in zip(args.ckpt, args.ckpt[1:])]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
