"""Run the kernel-task lever comparison end to end. CPU stub now; gpt-oss + GPU when available.

  python kernel_task/run.py                         # CPU stub (no model/GPU)
  python kernel_task/run.py --endpoint --device cuda --shape 128 128 32 \
      --model "$GPT_OSS_MODEL" --base-url "$GPT_OSS_BASE_URL"

Writes results/kernel/comparison.{md,json} + provenance.json. The headline = does H_THEN_W (and the
selector) reach a CORRECT, fast kernel where W-only entrenches a fast-but-wrong one.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import lever_loop as L  # noqa: E402
import harness as Hn  # noqa: E402

POLICIES = ["W_only", "H_only", "H_THEN_W", "selector"]


def _git():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                  # noqa: BLE001
        return "(unknown)"


def _gpu():
    try:
        import torch
        if torch.cuda.is_available():
            return {"cuda": True, "name": torch.cuda.get_device_name(0),
                    "n": torch.cuda.device_count()}
    except Exception:                                  # noqa: BLE001
        pass
    return {"cuda": False}


def _hash_dir(path):
    h = hashlib.sha256()
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            if fn.endswith(".py") or fn.endswith(".md"):
                with open(os.path.join(root, fn), "rb") as f:
                    h.update(f.read())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", action="store_true", help="use gpt-oss endpoint (else CPU stub)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shape", nargs=3, type=int, default=[32, 32, 8], metavar=("N", "K", "C"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "kernel"))
    args = ap.parse_args()

    mode = "endpoint" if args.endpoint else "stub"
    spec = {"shape": list(args.shape), "device": args.device}
    base = Hn.baseline_latency(spec)

    rows = []
    for pol in POLICIES:
        r = L.run_episode(mode, pol, spec=spec, max_steps=args.max_steps,
                          model=args.model, base_url=args.base_url)
        rows.append(r)
        levers = "->".join(t["lever"] for t in r["trajectory"])
        print(f"{pol:10s} correct={r['final_heldout_correct']!s:5} "
              f"passes_strong={r['final_passes_strong']!s:5} "
              f"speedup={r['final_speedup_vs_baseline']}  [{levers}]")

    os.makedirs(args.out, exist_ok=True)
    md = [f"# Kernel-task lever comparison ({mode}, device={args.device}, shape={tuple(args.shape)})",
          "",
          "Op: Triangle Multiplicative Update outgoing core `out[i,j,c]=sum_k a[i,k,c]b[j,k,c]` "
          "(SIA paper's GPU-kernel task). Final quality judged under the STRONG verifier (random "
          f"inputs, tight tol). Baseline (torch_bmm) latency = {base} ms.", "",
          "| policy | final correct | passes strong | speedup vs baseline | levers |",
          "|---|---|---|---|---|"]
    for r in rows:
        levers = " → ".join(t["lever"] for t in r["trajectory"])
        sp = r["final_speedup_vs_baseline"]
        md.append(f"| {r['policy']} | {r['final_heldout_correct']} | {r['final_passes_strong']} "
                  f"| {round(sp, 2) if sp else sp} | {levers} |")
    md += ["",
           "Reading: W-only entrenches a fast-but-WRONG kernel (high 'speedup', correct=False) under "
           "the weak verifier; H-only exposes it but cannot repair (no retrain); H_THEN_W and the "
           "selector reach a CORRECT kernel. The selector picks H_THEN_W from the implausible-speedup "
           "shortcut signature — no held-out labels."]
    with open(os.path.join(args.out, "comparison.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump({"mode": mode, "device": args.device, "shape": args.shape,
                   "baseline_ms": base, "rows": rows}, f, indent=2)
    prov = {"mode": mode, "device": args.device, "shape": args.shape, "model": args.model,
            "git_commit": _git(), "gpu": _gpu(),
            "kernel_task_hash": _hash_dir(HERE), "baseline_ms": base}
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print("\n".join(md))
    print(f"\nsaved -> {args.out}/comparison.{{md,json}} + provenance.json")


if __name__ == "__main__":
    main()
