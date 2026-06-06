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
    return ds_hash
