"""RC-ANM Step 1 — premise probe on a trained EqM checkpoint (GPU, inference).

Loads a trained EqM-B/2 (EMA = frozen teacher = mined model), VAE-encodes IN-1K
batches, runs ANM/PGD endpoint mining at an eps_ball sweep, and measures per
mined endpoint: EqM-native safety scores + the training-gradient impact of
safe vs unsafe endpoints. Tests the pre-registered premise
(preregistration-rcanm-step1.md): does v10 mining actually produce unsafe
endpoints worth certifying, and do they poison the gradient?

Inference-only. No training, no FID. Writes a JSON verdict.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

from models import EqM_models
from transport import create_transport
from diffusers.models import AutoencoderKL


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size),
                                     resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size),
                                 resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[cy:cy + image_size, cx:cx + image_size])


def ct_of(transport, t):
    return transport.get_ct(t)[:, None, None, None]


@torch.no_grad()
def teacher_field(model, xt, t, y):
    out = model(xt, t, y=y)
    if not torch.is_tensor(out):
        out = out[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae", default="ema")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--anm-steps", type=int, default=3)
    ap.add_argument("--anm-step-size", type=float, default=0.15)
    ap.add_argument("--eps-grid", type=str, default="0.25,0.5,1.0,1.5")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)
    latent = args.image_size // 8

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cargs = ckpt.get("args", None)
    model_name = getattr(cargs, "model", "EqM-B/2")
    uncond = getattr(cargs, "uncond", True)
    ebm = getattr(cargs, "ebm", "none")
    path_type = getattr(cargs, "path_type", "Linear")
    prediction = getattr(cargs, "prediction", "velocity")
    loss_weight = getattr(cargs, "loss_weight", None)
    if loss_weight in ("None", "none", ""):
        loss_weight = None
    train_eps = getattr(cargs, "train_eps", 0)
    sample_eps = getattr(cargs, "sample_eps", 0)

    model = EqM_models[model_name](input_size=latent, num_classes=args.num_classes,
                                  uncond=uncond, ebm=ebm).to(device)
    sd = ckpt["ema"] if "ema" in ckpt else ckpt.get("model", ckpt)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    transport = create_transport(path_type, prediction, loss_weight,
                                 train_eps, sample_eps)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    tf = transforms.Compose([
        transforms.Lambda(lambda im: center_crop_arr(im, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)])
    ds = ImageFolder(args.data_path, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True)

    eps_grid = [float(x) for x in args.eps_grid.split(",")]

    def flat_norm(z):
        return z.flatten(1).norm(dim=1).view(-1, *([1] * (z.ndim - 1)))

    def residual(xt, t, y, target):
        out = teacher_field(model, xt, t, y)
        return ((out - target) ** 2).flatten(1).mean(1)

    def mine(x1, t, y, eps0, eps_ball):
        """PGD ascent on the EqM residual w.r.t. the endpoint (frozen teacher)."""
        ct = ct_of(transport, t)
        norm0 = flat_norm(eps0).clamp_min(1e-12)
        delta = torch.zeros_like(eps0)
        for _ in range(args.anm_steps):
            ep = (eps0 + delta).detach().requires_grad_(True)
            xt = t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * ep
            target = (x1 - ep) * ct
            with torch.enable_grad():
                out = model(xt, t, y=y)        # model frozen; grad flows to ep
                if not torch.is_tensor(out):
                    out = out[0]
                res = ((out - target) ** 2).sum()
            g = torch.autograd.grad(res, ep)[0]
            g = g / flat_norm(g).clamp_min(1e-12)
            delta = (delta + args.anm_step_size * norm0 * g).detach()
            dn = flat_norm(delta).clamp_min(1e-12)
            delta = delta * (eps_ball * norm0 / dn).clamp(max=1.0)
        return (eps0 + delta).detach()

    @torch.no_grad()
    def short_rollout_dist(x_start, t, y, steps=10, eta=0.05):
        """How far a short teacher-field descent moves the point (contraction
        proxy): smaller final ||f|| = closer to a fixed point."""
        x = x_start.clone()
        for _ in range(steps):
            x = x + eta * teacher_field(model, x, t, y)
        return teacher_field(model, x, t, y).flatten(1).norm(dim=1)

    results = {str(e): {"r_field": [], "r_target": [], "r_inflate": [],
                        "r_return": [], "disp": []} for e in eps_grid}
    grad_rows = []   # (eps, r_field, grad_cos_to_clean, grad_mag)
    it = iter(loader)
    for bi in range(args.batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device)
        with torch.no_grad():
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
        n = x1.shape[0]
        t = torch.rand(n, device=device)
        y = torch.randint(args.num_classes, (n,), device=device)
        eps0 = torch.randn_like(x1)
        ct = ct_of(transport, t)
        tt = t[:, None, None, None]
        xt_orig = tt * x1 + (1 - tt) * eps0
        tgt_orig = (x1 - eps0) * ct
        with torch.no_grad():
            f_orig = teacher_field(model, xt_orig, t, y)
            res_orig = residual(xt_orig, t, y, tgt_orig)
            ret_orig = short_rollout_dist(xt_orig, t, y)

        # clean-batch mean gradient (for M2 alignment) on a small param subset
        for e in eps_grid:
            eps_adv = mine(x1, t, y, eps0, e)
            xt_adv = tt * x1 + (1 - tt) * eps_adv
            tgt_adv = (x1 - eps_adv) * ct
            with torch.no_grad():
                f_adv = teacher_field(model, xt_adv, t, y)
                res_adv = residual(xt_adv, t, y, tgt_adv)
                ret_adv = short_rollout_dist(xt_adv, t, y)
            r_field = (1 - F.cosine_similarity(f_adv.flatten(1),
                                               f_orig.flatten(1), dim=1)) / 2
            r_target = (1 - F.cosine_similarity((x1 - eps_adv).flatten(1),
                                                (x1 - eps0).flatten(1), dim=1)) / 2
            r_inflate = (res_adv / res_orig.clamp_min(1e-9)).clamp(0, 50)
            r_return = ((ret_adv - ret_orig) / (ret_orig.abs() + 1e-9)).clamp(-5, 5)
            disp = (flat_norm(eps_adv - eps0) / flat_norm(eps0).clamp_min(1e-12)
                    ).flatten()
            R = results[str(e)]
            R["r_field"] += r_field.cpu().tolist()
            R["r_target"] += r_target.cpu().tolist()
            R["r_inflate"] += r_inflate.cpu().tolist()
            R["r_return"] += r_return.cpu().tolist()
            R["disp"] += disp.cpu().tolist()

        # M2 gradient impact at the v10 eps (0.5 if present else first): per-sample
        # EqM-loss gradient cosine to the clean-batch-mean gradient.
        if bi < 8:
            e_v10 = 0.5 if 0.5 in eps_grid else eps_grid[0]
            eps_adv = mine(x1, t, y, eps0, e_v10)
            xt_adv = tt * x1 + (1 - tt) * eps_adv
            tgt_adv = (x1 - eps_adv) * ct
            with torch.no_grad():
                f_adv = teacher_field(model, xt_adv, t, y)
                rfield = ((1 - F.cosine_similarity(f_adv.flatten(1),
                          f_orig.flatten(1), dim=1)) / 2).cpu().numpy()
            # clean-batch mean grad (last conv/linear param of the model)
            params = [p for p in model.parameters()]
            target_p = params[-2]
            target_p.requires_grad_(True)
            def loss_for(idx_xt, idx_tgt, idxs):
                out = model(idx_xt[idxs], t[idxs], y=y[idxs])
                if not torch.is_tensor(out):
                    out = out[0]
                return ((out - idx_tgt[idxs]) ** 2).mean()
            gclean = torch.autograd.grad(
                loss_for(xt_orig, tgt_orig, torch.arange(n, device=device)),
                target_p, retain_graph=False)[0].flatten()
            for i in range(n):
                gi = torch.autograd.grad(
                    loss_for(xt_adv, tgt_adv, torch.tensor([i], device=device)),
                    target_p, retain_graph=False)[0].flatten()
                cos = F.cosine_similarity(gi, gclean, dim=0).item()
                grad_rows.append((float(rfield[i]), cos, float(gi.norm())))
            target_p.requires_grad_(False)
        print(f"batch {bi+1}/{args.batches} done", flush=True)

    # ---- aggregate + pre-registered gates ----
    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    summary = {}
    null_field = np.array(results[str(eps_grid[0])]["r_field"])  # weakest eps null
    for e in eps_grid:
        R = results[str(e)]
        rf = np.array(R["r_field"]); rt = np.array(R["r_target"])
        # unsafe = above 95th pct of the WEAKEST-eps r_field null
        thr_f = pct(null_field, 95)
        summary[str(e)] = dict(
            disp_mean=float(np.mean(R["disp"])),
            r_field_mean=float(rf.mean()), r_field_p95=pct(rf, 95),
            r_target_mean=float(rt.mean()),
            r_inflate_mean=float(np.mean(R["r_inflate"])),
            r_return_mean=float(np.mean(R["r_return"])),
            unsafe_frac_field=float((rf > thr_f).mean()),
            unsafe_frac_target=float((rt > pct(np.array(
                results[str(eps_grid[0])]["r_target"]), 95)).mean()))

    # M2: gradient alignment, unsafe (high r_field) vs safe
    verdict = {"checkpoint": args.ckpt, "model": model_name, "uncond": uncond,
               "eps_grid": eps_grid, "per_eps": summary}
    if grad_rows:
        gr = np.array(grad_rows)
        med_r = np.median(gr[:, 0])
        unsafe = gr[gr[:, 0] > med_r]
        safe = gr[gr[:, 0] <= med_r]
        from scipy import stats
        tt_ = stats.ttest_ind(unsafe[:, 1], safe[:, 1], equal_var=False)
        verdict["gradient_impact"] = dict(
            n=len(gr), median_rfield=float(med_r),
            grad_cos_unsafe=float(unsafe[:, 1].mean()),
            grad_cos_safe=float(safe[:, 1].mean()),
            welch_p=float(tt_.pvalue),
            unsafe_worse=bool(unsafe[:, 1].mean() < safe[:, 1].mean()
                              and tt_.pvalue < 0.05))

    # gates
    e_v10 = "0.5" if "0.5" in summary else str(eps_grid[0])
    s1_premise = summary[e_v10]["unsafe_frac_field"] >= 0.05 or \
        summary[e_v10]["unsafe_frac_target"] >= 0.05
    any_eps_unsafe = any(summary[str(e)]["unsafe_frac_field"] >= 0.05
                         for e in eps_grid)
    s1_impact = verdict.get("gradient_impact", {}).get("unsafe_worse", False)
    if s1_premise and s1_impact:
        branch = "P1 PREMISE-HOLDS -> Step 2 CIFAR-mini RC-ANM ladder"
    elif any_eps_unsafe and s1_impact:
        branch = "P2 SAFE-AGGRESSIVE -> Step 2 with aggressive arm as treatment"
    else:
        branch = "P3 NON-PROBLEM -> certification bounds nothing at scale; STOP"
    verdict["gates"] = dict(s1_premise=bool(s1_premise),
                            s1_impact=bool(s1_impact),
                            any_eps_unsafe=bool(any_eps_unsafe))
    verdict["branch"] = branch
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps({"gates": verdict["gates"], "branch": branch,
                      "grad": verdict.get("gradient_impact")}, indent=2))


if __name__ == "__main__":
    main()
