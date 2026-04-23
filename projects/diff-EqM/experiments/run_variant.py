#!/usr/bin/env python3
"""Dispatcher for DG-ANM variant trainers.

Usage:
    python projects/diff-EqM/experiments/run_variant.py \
        --config projects/diff-EqM/configs/variants/v01_current.json
    # or override variant explicitly:
    python projects/diff-EqM/experiments/run_variant.py \
        --variant v03_noised_negatives --output-dir ... --seed 0

The variant is chosen via --variant (or the `variant` field in --config)
and resolved to projects/diff-EqM/experiments/dganm_variants/<variant>.py
(which must expose a `train(args)` function and a `TrainArgs`-compatible
config).

Prints one grep-friendly final line:
    cifar10_variant_fid[<variant>]: <value>
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

# Let the package import succeed when run as a script.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dganm_variants._common import TrainArgs  # noqa: E402


def _load_args(cli) -> TrainArgs:
    cfg = {}
    if cli.config:
        cfg = json.loads(Path(cli.config).read_text())
    # CLI overrides config.
    for k, v in vars(cli).items():
        if k in ("config",):
            continue
        if v is not None:
            cfg[k] = v
    # Split extras (any unknown keys) into the TrainArgs.extras slot.
    known = set(TrainArgs.__dataclass_fields__.keys())
    extras = {k: cfg.pop(k) for k in list(cfg.keys()) if k not in known}
    if extras:
        cfg["extras"] = {**(cfg.get("extras") or {}), **extras}
    if "variant" not in cfg:
        raise SystemExit("must provide --variant or `variant` in config JSON")
    if "output_dir" not in cfg:
        raise SystemExit("must provide --output-dir or `output_dir` in config JSON")
    return TrainArgs(**cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-every-epochs", dest="eval_every_epochs", type=int, default=None)
    ap.add_argument("--eval-fid-samples", dest="eval_fid_samples", type=int, default=None)
    ap.add_argument("--final-fid-samples", dest="final_fid_samples", type=int, default=None)
    ap.add_argument("--euler-num-steps", dest="euler_num_steps", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None)
    cli = ap.parse_args()
    args = _load_args(cli)

    mod_name = f"dganm_variants.{args.variant}"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as err:
        raise SystemExit(f"Unknown variant `{args.variant}`: {err}")
    if not hasattr(mod, "train"):
        raise SystemExit(f"variant `{args.variant}` missing `train(args)`")
    fid = mod.train(args)
    print(f"DONE variant={args.variant} fid={fid:.4f}", flush=True)


if __name__ == "__main__":
    main()
