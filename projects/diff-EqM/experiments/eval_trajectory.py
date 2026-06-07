# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Capability ladder v2 — Rungs C / E / F: trajectory-level behavioral probes.

All FROZEN checkpoints, identical GD sampler, matched noise + labels.
Reuses the GD field/decoder helpers from eval_capabilities.py (same dir).

Generation: EqM GD walks t from ~0 (pure noise) up to 1 (data), step xt += f(xt,t,y)*dt.
A "schedule" picks which model and which class label is active in each t-segment, so
we can splice vanilla/v10 fields (Rung E) or switch the class condition (Rung F)
partway, or branch both arms from a shared intermediate state (Rung C).

Behavioral metrics use a pretrained ImageNet classifier (resnet50) on decoded 256px:
  - top1 acc / top5 hit vs intended label
  - mean top1 confidence (softmax max)
  - for edits: target-class success + source-class retention + LPIPS source-preservation

Modes:
  rescue  (C): gen vanilla full; flag bottom-conf samples; from their matched
               mid-trajectory latent, continue vanilla vs v10; test v10 rescue.
  swap    (E): vanilla-early->v10-late and v10-early->vanilla-late at switch t in
               {.25,.5,.75}, plus pure arms; locate where v10 contributes.
  edit    (F): gen class A to switch t, switch label to B, continue; measure
               target-B success, source-A retention, artifact, confidence.

Usage:
  python eval_trajectory.py --mode swap --vanilla-ckpt V.pt --anm-ckpt A.pt \
      --out-dir DIR --num-images 64 --model EqM-B/2
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from eval_capabilities import load_eqm, decode, eqm_field  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402


# --------------------------------------------------------------------------- #
# Scheduled GD sampler: walk t from t0->1; (model,label) chosen per step.
# --------------------------------------------------------------------------- #
def gd_schedule(pick_fn, x_init, t0, stepsize, n_steps, device,
                snapshot_at=None):
    """pick_fn(i, t) -> (model, y_tensor). Returns (x_final, {step: latent}).

    snapshot_at: set/list of step indices to record the latent (cloned, detached).
    """
    snapshot_at = set(snapshot_at or [])
    xt = x_init.clone()
    B = xt.shape[0]
    t = torch.full((B,), float(t0), device=device)
    snaps = {}
    for i in range(n_steps):
        model, y = pick_fn(i, float(t[0].item()))
        out = eqm_field(model, xt, t, y)
        xt = xt + out * stepsize
        t = t + stepsize
        if i in snapshot_at:
            snaps[i] = xt.clone().detach()
    return xt, snaps


def make_classifier(device):
    from torchvision.models import resnet50, ResNet50_Weights
    clf = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return clf, norm


def classify(clf, norm, px01):
    """px01 in [0,1] (N,3,H,W). Returns (top1_idx, top1_conf, top5_idx)."""
    with torch.no_grad():
        logits = clf(norm(px01))
        prob = logits.softmax(dim=1)
        conf, top1 = prob.max(dim=1)
        top5 = logits.topk(5, dim=1).indices
    return top1.tolist(), conf.tolist(), top5.tolist()


def balanced_labels(n, num_classes, seed, device):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.integers(0, num_classes, size=n), device=device)


def init_noise(n, image_size, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    ls = image_size // 8
    return torch.randn(n, 4, ls, ls, generator=g, device=device)


def mean(x):
    return float(np.mean(x)) if len(x) else float("nan")


# --------------------------------------------------------------------------- #
# Rung E — trajectory swap
# --------------------------------------------------------------------------- #
def run_swap(vanilla, anm, vae, clf, norm, device, args):
    n = args.num_images
    y = balanced_labels(n, args.num_classes, args.seed, device)
    x0 = init_noise(n, args.image_size, args.seed, device)
    n_steps = args.num_sampling_steps
    t0 = args.t0
    switches = [float(s) for s in args.switches.split(",")]

    def const(model):
        return lambda i, t: (model, y)

    def spliced(early, late, ts):
        return lambda i, t: (early if t < ts else late, y)

    configs = {"pure_vanilla": const(vanilla), "pure_v10": const(anm)}
    for ts in switches:
        configs[f"van_then_v10@{ts}"] = spliced(vanilla, anm, ts)
        configs[f"v10_then_van@{ts}"] = spliced(anm, vanilla, ts)

    metrics = {"mode": "swap", "t0": t0, "n_steps": n_steps,
               "switches": switches, "configs": {}}
    grids = []
    for name, pick in configs.items():
        xf, _ = gd_schedule(pick, x0, t0, args.stepsize, n_steps, device)
        px = decode(xf, vae)
        top1, conf, top5 = classify(clf, norm, px)
        acc1 = mean([int(top1[i] == int(y[i])) for i in range(n)])
        acc5 = mean([int(int(y[i]) in top5[i]) for i in range(n)])
        metrics["configs"][name] = {"top1_acc": acc1, "top5_acc": acc5,
                                    "mean_conf": mean(conf)}
        print(f"[swap {name}] top1={acc1:.3f} top5={acc5:.3f} conf={mean(conf):.3f}")
        grids.append(px[:8].cpu())
    save_image(make_grid(torch.cat(grids, 0), nrow=8, padding=2),
               str(Path(args.out_dir) / "swap_grid.png"))
    return metrics


# --------------------------------------------------------------------------- #
# Rung F — class-guided counterfactual edit
# --------------------------------------------------------------------------- #
def run_edit(vanilla, anm, vae, clf, norm, device, args):
    n = args.num_images
    yA = balanced_labels(n, args.num_classes, args.seed, device)
    yB = balanced_labels(n, args.num_classes, args.seed + 777, device)
    # ensure A != B
    same = (yA == yB)
    yB = torch.where(same, (yB + 1) % args.num_classes, yB)
    x0 = init_noise(n, args.image_size, args.seed, device)
    n_steps = args.num_sampling_steps
    t0 = args.t0
    switches = [float(s) for s in args.switches.split(",")]

    metrics = {"mode": "edit", "switches": switches, "arms": {}}
    for arm_name, model in [("vanilla", vanilla), ("v10", anm)]:
        # reference: pure source-A generation (no switch), for source preservation
        ref_A, _ = gd_schedule(lambda i, t: (model, yA), x0, t0,
                               args.stepsize, n_steps, device)
        refA_px = decode(ref_A, vae)
        arm = {}
        for ts in switches:
            pick = lambda i, t, ts=ts: (model, yA if t < ts else yB)
            xf, _ = gd_schedule(pick, x0, t0, args.stepsize, n_steps, device)
            px = decode(xf, vae)
            top1, conf, top5 = classify(clf, norm, px)
            tgt = mean([int(int(yB[i]) in top5[i]) for i in range(n)])      # target B success
            src = mean([int(int(yA[i]) in top5[i]) for i in range(n)])      # source A leftover
            # source-content preservation: LPIPS to pure-A image (lower=more preserved)
            try:
                import lpips
                lp = lpips.LPIPS(net="alex").to(device)
                with torch.no_grad():
                    pres = lp(px * 2 - 1, refA_px * 2 - 1).flatten().tolist()
                pres_m = mean(pres)
            except Exception:
                pres_m = None
            arm[f"switch@{ts}"] = {"target_B_top5": tgt, "source_A_top5": src,
                                   "mean_conf": mean(conf),
                                   "lpips_to_pureA": pres_m}
            print(f"[edit {arm_name} @{ts}] B={tgt:.3f} A_left={src:.3f} "
                  f"conf={mean(conf):.3f} presLPIPS={pres_m}")
        metrics["arms"][arm_name] = arm
    return metrics


# --------------------------------------------------------------------------- #
# Rung C — failed-generation rescue
# --------------------------------------------------------------------------- #
def run_rescue(vanilla, anm, vae, clf, norm, device, args):
    n = args.num_images
    y = balanced_labels(n, args.num_classes, args.seed, device)
    x0 = init_noise(n, args.image_size, args.seed, device)
    n_steps = args.num_sampling_steps
    t0 = args.t0
    switch_step = int(args.rescue_switch * n_steps)

    # 1) full vanilla generation, snapshot the mid-trajectory latent
    xf_van, snaps = gd_schedule(lambda i, t: (vanilla, y), x0, t0,
                                args.stepsize, n_steps, device,
                                snapshot_at=[switch_step])
    px_van = decode(xf_van, vae)
    top1, conf, top5 = classify(clf, norm, px_van)
    conf = np.array(conf)
    # 2) flag "bad" = bottom-quartile vanilla confidence
    thr = np.quantile(conf, 0.25)
    bad = np.where(conf <= thr)[0]
    metrics = {"mode": "rescue", "switch_step": switch_step,
               "n_bad": int(len(bad)), "conf_thr": float(thr),
               "vanilla_full": {"mean_conf": mean(conf.tolist()),
                                "bad_mean_conf": mean(conf[bad].tolist())}}
    if len(bad) == 0:
        return metrics
    mid = snaps[switch_step][bad]                         # matched bad mid-states
    yb = y[bad]
    t_mid = float(t0 + switch_step * args.stepsize)
    remaining = n_steps - switch_step
    # 3) continue the SAME bad states under vanilla vs v10
    out = {}
    for arm_name, model in [("vanilla_cont", vanilla), ("v10_cont", anm)]:
        xf, _ = gd_schedule(lambda i, t: (model, yb), mid, t_mid,
                            args.stepsize, remaining, device)
        px = decode(xf, vae)
        t1, cf, t5 = classify(clf, norm, px)
        acc5 = mean([int(int(yb[i]) in t5[i]) for i in range(len(bad))])
        out[arm_name] = {"mean_conf": mean(cf), "top5_acc": acc5}
        print(f"[rescue {arm_name}] conf={mean(cf):.3f} top5={acc5:.3f} (n_bad={len(bad)})")
    metrics["rescue"] = out
    metrics["rescue_conf_gain_v10_minus_vanilla"] = (
        out["v10_cont"]["mean_conf"] - out["vanilla_cont"]["mean_conf"])
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["swap", "edit", "rescue"])
    ap.add_argument("--vanilla-ckpt", required=True)
    ap.add_argument("--anm-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="EqM-B/2")
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-images", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stepsize", type=float, default=0.003)
    ap.add_argument("--num-sampling-steps", type=int, default=250)
    ap.add_argument("--t0", type=float, default=0.003)
    ap.add_argument("--switches", default="0.25,0.5,0.75",
                    help="swap/edit: t fractions to switch model/label")
    ap.add_argument("--rescue-switch", type=float, default=0.5,
                    help="rescue: trajectory fraction to branch from")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)

    vanilla = load_eqm(args.vanilla_ckpt, args.model, args.num_classes,
                       args.image_size, device)
    anm = load_eqm(args.anm_ckpt, args.model, args.num_classes,
                   args.image_size, device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    clf, norm = make_classifier(device)

    if args.mode == "swap":
        m = run_swap(vanilla, anm, vae, clf, norm, device, args)
    elif args.mode == "edit":
        m = run_edit(vanilla, anm, vae, clf, norm, device, args)
    else:
        m = run_rescue(vanilla, anm, vae, clf, norm, device, args)

    p = Path(args.out_dir) / f"metrics_{args.mode}.json"
    p.write_text(json.dumps(m, indent=2))
    print(f"[metrics] wrote {p}")


if __name__ == "__main__":
    main()
