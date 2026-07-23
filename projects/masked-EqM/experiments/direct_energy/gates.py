"""Pre-registered gates for the direct-energy validation campaign."""
from __future__ import annotations

import statistics
from typing import Any


def fixed_batch_gate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate fixed-batch learnability without looking at a single endpoint.

    The first and final fifth of finite observations are compared.  This is
    deliberately a viability gate, not a baseline-beating claim.
    """
    finite = [m for m in metrics if m.get("finite", False)]
    if len(finite) < 20:
        return {"pass": False, "reason": "fewer than 20 finite observations"}
    window = max(1, len(finite) // 5)
    first, last = finite[:window], finite[-window:]
    mean = lambda xs, key: statistics.fmean(float(x[key]) for x in xs)
    initial_loss, final_loss = mean(first, "loss"), mean(last, "loss")
    initial_cos, final_cos = mean(first, "field_target_cosine"), mean(last, "field_target_cosine")
    max_memory = max(float(m["peak_memory_mb"]) for m in finite)
    min_ratio = min(float(m["field_target_norm_ratio"]) for m in finite)
    max_ratio = max(float(m["field_target_norm_ratio"]) for m in finite)
    head_grad = max(float(m["head_grad_norm"]) for m in finite)
    backbone_grad = max(float(m["backbone_grad_norm"]) for m in finite)
    result = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_cosine": initial_cos,
        "final_cosine": final_cos,
        "loss_reduction_fraction": 1 - final_loss / max(initial_loss, 1e-12),
        "cosine_gain": final_cos - initial_cos,
        "peak_memory_mb": max_memory,
        "norm_ratio_range": [min_ratio, max_ratio],
        "head_grad_max": head_grad,
        "backbone_grad_max": backbone_grad,
    }
    checks = {
        "numerical_stability": len(finite) == len(metrics),
        "learnability": result["loss_reduction_fraction"] >= 0.10,
        "alignment": result["cosine_gain"] >= 0.05,
        "gradient_flow": head_grad > 0 and backbone_grad > 0,
        "noncollapsed_field": max_ratio > 1e-3,
        "nonexploding_field": max_ratio < 100,
    }
    result["checks"] = checks
    result["pass"] = all(checks.values())
    return result
