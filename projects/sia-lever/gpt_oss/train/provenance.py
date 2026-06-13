"""Provenance logging for every adapter (guardrail #4: hash and log everything)."""

import hashlib
import json
import os
import subprocess


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "(unknown)"


_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The harness (H lever) and evaluator (scorer) are load-bearing for any reported metric, so their
# exact bytes are pinned alongside the adapter. Guardrail #4: hash and log everything.
_HARNESS_PATH = os.path.join(_PROJ, "harness", "verifier.py")
_EVALUATOR_PATH = os.path.join(_PROJ, "sia_task", "data", "public", "evaluate.py")


def _sha256_maybe(path):
    return _sha256_file(path) if path and os.path.exists(path) else "(missing)"


def _adapter_hash(out_dir):
    """Hash the produced PEFT adapter weights so a reported result is tied to exact weights.
    Hashes the first existing adapter weight file (safetensors/bin)."""
    for fn in ("adapter_model.safetensors", "adapter_model.bin"):
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            return fn, _sha256_file(p)
    return "(missing)", "(missing)"


def gpu_info():
    info = {"n_gpu": 0, "gpus": []}
    try:
        import torch
        if torch.cuda.is_available():
            info["n_gpu"] = torch.cuda.device_count()
            info["bf16"] = bool(torch.cuda.is_bf16_supported())
            for i in range(info["n_gpu"]):
                p = torch.cuda.get_device_properties(i)
                info["gpus"].append({"name": p.name, "mem_gb": round(p.total_memory / 1e9, 1)})
    except Exception as e:
        info["error"] = str(e)
    return info


def write_provenance(out_dir, base_model, dataset_path, train_config, eval_results=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "base_model.txt"), "w") as f:
        f.write(str(base_model) + "\n")
    with open(os.path.join(out_dir, "git_commit.txt"), "w") as f:
        f.write(_git_commit() + "\n")
    ds_hash = _sha256_file(dataset_path) if dataset_path and os.path.exists(dataset_path) else "(missing)"
    with open(os.path.join(out_dir, "dataset_hash.txt"), "w") as f:
        f.write(f"{dataset_path}\nsha256 {ds_hash}\n")
    with open(os.path.join(out_dir, "gpu_info.json"), "w") as f:
        json.dump(gpu_info(), f, indent=2)
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)
    if eval_results is not None:
        with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)

    # Consolidated provenance record: ties this adapter to the exact harness, evaluator, dataset,
    # base model, GPU and git commit that produced it (all goal-required fields in one place).
    adapter_file, adapter_hash = _adapter_hash(out_dir)
    record = {
        "base_model": str(base_model),
        "git_commit": _git_commit(),
        "dataset_path": dataset_path,
        "dataset_sha256": ds_hash,
        "harness_path": _HARNESS_PATH,
        "harness_sha256": _sha256_maybe(_HARNESS_PATH),
        "evaluator_path": _EVALUATOR_PATH,
        "evaluator_sha256": _sha256_maybe(_EVALUATOR_PATH),
        "adapter_file": adapter_file,
        "adapter_sha256": adapter_hash,
        "gpu_info": gpu_info(),
    }
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(record, f, indent=2)
    return ds_hash
