"""Isolated candidate-kernel runner. Execs a kernel SOURCE STRING (agent- or seed-authored) in a
fresh subprocess, measures correctness + latency, prints one JSON line. Subprocess isolation matters:
LLM/Triton kernels can hang, OOM, or segfault — the parent (harness.py) runs this with a hard
timeout and treats a crash/timeout as "does not compile/run", which is itself a valid trace signal.

Reports RAW signals; the H lever (weak vs strong verifier) decides which to enforce:
  passes_fixed   correctness on the ONE fixed input  (weak verifier checks this)
  passes_random  correctness on N random inputs       (strong verifier / honest negative control)
  latency_ms     timed on the fixed input (CUDA events on GPU, perf_counter on CPU)

CLI (used by harness via subprocess):
  python kernel_task/runner.py --src <kernel.py> --spec <spec.json>
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from reference import make_inputs, reference  # noqa: E402


def _time_call(fn, a, b, reps, device):
    fn(a, b)  # warm
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn(a, b)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps          # ms
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(a, b)
    return (time.perf_counter() - t0) / reps * 1e3      # ms


def run(src, spec):
    shape = tuple(spec.get("shape", [32, 32, 8]))
    device = spec.get("device", "cpu")
    fixed_seed = spec.get("fixed_seed", 0)
    n_random = spec.get("n_random", 8)
    tol = spec.get("tol", 1e-4)
    reps = spec.get("reps", 50)
    res = {"compiles": False, "error": None, "passes_fixed": False, "passes_random": False,
           "max_abs_err_random": None, "latency_ms": None, "out_shape": None, "device": device}

    ns = {}
    try:
        exec(compile(src, "<candidate>", "exec"), ns)     # noqa: S102 (sandboxed via subprocess)
        if "kernel" not in ns or not callable(ns["kernel"]):
            res["error"] = "no callable 'kernel' defined"
            return res
        fn = ns["kernel"]
    except Exception as e:                                # noqa: BLE001
        res["error"] = f"compile: {type(e).__name__}: {e}"
        return res

    try:
        a0, b0 = make_inputs(fixed_seed, shape, device)
        out = fn(a0, b0)
        ref0 = reference(a0, b0)
        res["compiles"] = True
        res["out_shape"] = list(out.shape)
        res["passes_fixed"] = bool(out.shape == ref0.shape
                                   and torch.allclose(out, ref0, atol=tol, rtol=tol))
    except Exception as e:                                # noqa: BLE001
        res["error"] = f"run_fixed: {type(e).__name__}: {e}"
        return res

    # honest negative control: correctness on inputs the candidate could not have memorized
    try:
        max_err, ok = 0.0, True
        for r in range(1, n_random + 1):
            a, b = make_inputs(1000 + r, shape, device)
            out = fn(a, b)
            if out.shape != reference(a, b).shape:
                ok = False
                break
            err = (out - reference(a, b)).abs().max().item()
            max_err = max(max_err, err)
            if err >= tol:
                ok = False
        res["passes_random"] = bool(ok)
        res["max_abs_err_random"] = max_err
    except Exception as e:                                # noqa: BLE001
        res["error"] = f"run_random: {type(e).__name__}: {e}"

    try:
        a0, b0 = make_inputs(fixed_seed, shape, device)
        res["latency_ms"] = _time_call(fn, a0, b0, reps, device)
    except Exception as e:                                # noqa: BLE001
        res["error"] = (res["error"] or "") + f" | timing: {type(e).__name__}: {e}"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    src = open(args.src).read()
    spec = json.loads(open(args.spec).read())
    print(json.dumps(run(src, spec)))


if __name__ == "__main__":
    main()
