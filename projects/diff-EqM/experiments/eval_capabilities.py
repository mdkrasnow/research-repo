# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Qualitative + quantitative capability eval for EqM checkpoints.

Tests the EqM energy-landscape downstream tasks (partially-noised denoising,
inpainting, image composition) on REAL ImageNet validation images. Built to
compare vanilla EqM vs adversarial-hard-example-mining (v10) checkpoints on
landscape-dependent tasks where the FID metric is expected to understate the
benefit of mining (mining sharpens the field OFF the data manifold, which is
exactly what these tasks exercise).

Single-GPU (no DDP); designed for a small batch (~8 images) so output grids
are human-inspectable. Forks the gradient-descent sampler loop from
eqm-upstream/sample_gd.py verbatim (same loop that produced FID 29.01).

Modes:
  denoise  : encode real val image -> add gamma-fraction noise -> GD restore.
             Output grid: [original | noised | restored] per gamma level.
  inpaint  : encode real val image -> mask region -> GD generate with known
             region clamped each step. Grid: [original | masked | filled].
  compose  : GD from noise with the field summed over two class labels.
             Grid: per (c1,c2) pair.

Each mode writes:
  - a PNG grid (decoded 256x256 images) for qualitative assessment
  - metrics.json (LPIPS + PSNR where applicable; classifier hits for compose)

Usage (single GPU):
  python eval_capabilities.py --mode denoise --ckpt <path> --tag v10 \\
      --out-dir <dir> --num-images 8 --val-path <imagenet/val>
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

# eqm-upstream on path (sibling dir); same import style as train_imagenet.py
UPSTREAM = str(Path(__file__).resolve().parent.parent / "eqm-upstream")
if UPSTREAM not in sys.path:
    sys.path.insert(0, UPSTREAM)

from models import EqM_models           # noqa: E402
from download import find_model         # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402

VAE_SCALE = 0.18215


# --------------------------------------------------------------------------- #
# Model / VAE helpers
# --------------------------------------------------------------------------- #
def load_eqm(ckpt_path, model_name, num_classes, image_size, device):
    latent_size = image_size // 8
    model = EqM_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
        uncond=True,    # EqM disables time conditioning (matches sample_gd.py / our trains)
        ebm="none",
    ).to(device)
    state_dict = find_model(ckpt_path)
    # prefer EMA weights (what FID eval used)
    if "ema" in state_dict:
        model.load_state_dict(state_dict["ema"])
    elif "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode(img, vae):
    """pixel tensor in [-1,1], (N,3,H,W) -> latent (N,4,h,w)."""
    with torch.no_grad():
        return vae.encode(img).latent_dist.sample() * VAE_SCALE


def decode(latent, vae):
    """latent (N,4,h,w) -> pixel tensor in [0,1] (N,3,H,W)."""
    with torch.no_grad():
        img = vae.decode(latent / VAE_SCALE).sample
    return torch.clamp(img * 0.5 + 0.5, 0.0, 1.0)


def eqm_field(model, x, t, y):
    """One EqM forward. Returns the gradient-field/velocity tensor.

    model.forward sets x.requires_grad_(True) internally; with ebm='none'
    no autograd.grad is taken, so we can run inside inference cleanly by
    detaching the output.
    """
    out = model(x, t, y)
    if not torch.is_tensor(out):
        out = out[0]
    return out.detach()


# --------------------------------------------------------------------------- #
# GD sampler core (mirrors sample_gd.py: xt <- xt + f(xt,t,y)*stepsize)
# --------------------------------------------------------------------------- #
def gd_restore(model, x_init, y, t_init, stepsize, num_steps, device,
               clamp_fn=None, clamp_start_frac=0.0):
    """Run EqM gradient descent from x_init at t_init up toward t=1.

    clamp_fn(xt, t) -> xt is applied AFTER each GD step once the loop has
    passed clamp_start_frac of its iterations. Delaying the clamp lets the
    global structure form before the known region is locked (RePaint-style);
    clamping from step 0 leaves the hole as garbage because the model never
    sees a half-clean/half-noise input it was trained on.
    """
    xt = x_init.clone()
    t = torch.full((xt.shape[0],), float(t_init), device=device)
    n_iter = max(1, int(round((1.0 - t_init) / stepsize)) if num_steps is None
                 else num_steps)
    clamp_after = int(clamp_start_frac * n_iter)
    for i in range(n_iter):
        out = eqm_field(model, xt, t, y)
        xt = xt + out * stepsize
        t = t + stepsize
        if clamp_fn is not None and i >= clamp_after:
            xt = clamp_fn(xt, float(t[0].item()))
    return xt


def gd_compose(model, x_init, y1, y2, stepsize, num_steps, device):
    """GD from noise with field = f(x,y1) + f(x,y2) (compositional EBM sum)."""
    xt = x_init.clone()
    t = torch.zeros((xt.shape[0],), device=device)
    for _ in range(num_steps):
        out = eqm_field(model, xt, t, y1) + eqm_field(model, xt, t, y2)
        xt = xt + out * stepsize
        t = t + stepsize
    return xt


# --------------------------------------------------------------------------- #
# Val data
# --------------------------------------------------------------------------- #
def load_val_images(val_path, n, image_size, device, seed=0):
    """Return (img_pixels[-1,1], class_idx) for n val images, deterministic."""
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1,1]
    ])
    # Manual fast loader: avoid ImageFolder's full 50K-file tree scan (slow on
    # holylabs NFS). Class idx = sorted-synset order, matching training's
    # ImageFolder convention exactly (val + train share synset dir names).
    synsets = sorted(d for d in os.listdir(val_path)
                     if os.path.isdir(os.path.join(val_path, d)))
    rng = np.random.default_rng(seed)
    # pick n distinct classes deterministically, one image each
    chosen = rng.choice(len(synsets), size=n, replace=(n > len(synsets)))
    imgs, labels = [], []
    for ci in chosen:
        cls_dir = os.path.join(val_path, synsets[int(ci)])
        files = sorted(os.listdir(cls_dir))
        fpath = os.path.join(cls_dir, files[0])
        img = Image.open(fpath).convert("RGB")
        imgs.append(tf(img))
        labels.append(int(ci))
    return torch.stack(imgs).to(device), torch.tensor(labels, device=device)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def psnr(a, b):
    """a,b in [0,1], (N,3,H,W). Returns per-image PSNR list."""
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3))
    return (10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))).tolist()


def try_lpips(device):
    try:
        import lpips
        return lpips.LPIPS(net="alex").to(device)
    except Exception as e:
        print(f"[warn] lpips unavailable ({e}); skipping LPIPS", file=sys.stderr)
        return None


def lpips_score(metric, a, b):
    """a,b in [0,1]; lpips wants [-1,1]."""
    if metric is None:
        return None
    with torch.no_grad():
        d = metric(a * 2 - 1, b * 2 - 1)
    return d.flatten().tolist()


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_denoise(model, vae, device, args):
    imgs, labels = load_val_images(args.val_path, args.num_images,
                                   args.image_size, device, args.seed)
    z0 = encode(imgs, vae)
    orig_px = torch.clamp(imgs * 0.5 + 0.5, 0, 1)

    gammas = [float(g) for g in args.gammas.split(",")]
    lp = try_lpips(device)
    rows = [orig_px.cpu()]               # row 0: originals
    metrics = {"mode": "denoise", "tag": args.tag, "gammas": {}}

    for gamma in gammas:
        torch.manual_seed(args.seed)
        noise = torch.randn_like(z0)
        x_g = (1.0 - gamma) * z0 + gamma * noise
        noised_px = decode(x_g, vae)
        restored = gd_restore(model, x_g, labels, t_init=(1.0 - gamma),
                              stepsize=args.stepsize, num_steps=None,
                              device=device)
        restored_px = decode(restored, vae)
        rows.append(noised_px.cpu())
        rows.append(restored_px.cpu())
        metrics["gammas"][f"{gamma}"] = {
            "psnr_restored_vs_orig": psnr(restored_px, orig_px),
            "lpips_restored_vs_orig": lpips_score(lp, restored_px, orig_px),
        }
        print(f"[denoise gamma={gamma}] mean PSNR="
              f"{np.mean(metrics['gammas'][f'{gamma}']['psnr_restored_vs_orig']):.2f}")

    # grid: column = image, row group = [orig, (noised,restored) per gamma]
    grid_tensor = torch.cat(rows, dim=0)
    save_grid(grid_tensor, args.num_images,
              Path(args.out_dir) / f"denoise_{args.tag}.png")
    return metrics


def run_inpaint(model, vae, device, args):
    imgs, labels = load_val_images(args.val_path, args.num_images,
                                   args.image_size, device, args.seed)
    z0 = encode(imgs, vae)
    orig_px = torch.clamp(imgs * 0.5 + 0.5, 0, 1)
    n, _, h, w = z0.shape

    # latent-space mask: 1 = known, 0 = hole. Center box covering ~middle 50%.
    mask = torch.ones((n, 1, h, w), device=device)
    if args.mask == "box":
        h0, h1 = h // 4, 3 * h // 4
        w0, w1 = w // 4, 3 * w // 4
        mask[:, :, h0:h1, w0:w1] = 0.0
    elif args.mask == "half":
        mask[:, :, :, w // 2:] = 0.0     # right half unknown
    else:
        raise ValueError(f"unknown mask {args.mask}")

    # Fixed noise for the known-region forward diffusion (RePaint-style).
    torch.manual_seed(args.seed)
    known_noise = torch.randn_like(z0)

    # EqM path: x_t = (1-t)*noise + t*data; t=1 is data, t=0 is noise.
    # Clamp the KNOWN region to its correct value at the current t so the
    # model always sees an on-path input there, and let the hole evolve freely.
    def clamp_fn(xt, t):
        known_at_t = (1.0 - t) * known_noise + t * z0
        return mask * known_at_t + (1.0 - mask) * xt

    # masked preview (known region clean, hole grey) for the grid
    masked_px = decode(mask * z0, vae)

    x0 = torch.randn_like(z0)
    # Clamp only after 50% of steps: global structure forms first, then the
    # known region locks. Clamping from step 0 leaves the hole as noise mosaic.
    filled = gd_restore(model, x0, labels, t_init=0.0,
                        stepsize=args.stepsize, num_steps=args.num_sampling_steps,
                        device=device, clamp_fn=clamp_fn,
                        clamp_start_frac=args.inpaint_clamp_start)
    filled_px = decode(filled, vae)

    lp = try_lpips(device)
    # Hole-only PSNR: upsample latent hole-mask (1-mask) to pixel resolution.
    import torch.nn.functional as Fnn
    hole_px = Fnn.interpolate(1.0 - mask, size=orig_px.shape[-2:],
                              mode="nearest")          # (n,1,H,W), 1 in hole
    hole_px = hole_px.expand(-1, 3, -1, -1)
    def psnr_masked(a, b, m):
        se = ((a - b) ** 2) * m
        mse = se.sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp_min(1)
        return (10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))).tolist()
    metrics = {
        "mode": "inpaint", "tag": args.tag, "mask": args.mask,
        "clamp_start_frac": args.inpaint_clamp_start,
        "psnr_filled_vs_orig_wholeimg": psnr(filled_px, orig_px),
        "psnr_hole_only": psnr_masked(filled_px, orig_px, hole_px),
        "lpips_filled_vs_orig": lpips_score(lp, filled_px, orig_px),
    }
    print(f"[inpaint {args.mask}] hole-only PSNR="
          f"{np.mean(metrics['psnr_hole_only']):.2f} "
          f"whole={np.mean(metrics['psnr_filled_vs_orig_wholeimg']):.2f}")

    rows = [orig_px.cpu(), masked_px.cpu(), filled_px.cpu()]
    save_grid(torch.cat(rows, dim=0), args.num_images,
              Path(args.out_dir) / f"inpaint_{args.mask}_{args.tag}.png")
    return metrics


def run_compose(model, vae, device, args):
    # N random distinct class pairs (deterministic by seed) for a real sample
    # size, instead of a handful of hand-picked pairs.
    if args.class_pairs:
        pairs = [tuple(int(x) for x in p.split(":"))
                 for p in args.class_pairs.split(",")]
    else:
        rng = np.random.default_rng(args.seed)
        pairs = []
        while len(pairs) < args.num_pairs:
            a, b = int(rng.integers(args.num_classes)), int(rng.integers(args.num_classes))
            if a != b:
                pairs.append((a, b))
    n = len(pairs)
    latent_size = args.image_size // 8
    torch.manual_seed(args.seed)

    # batch the GD over all pairs (chunk to fit memory)
    chunk = args.compose_batch
    outs = []
    for s in range(0, n, chunk):
        pc = pairs[s:s + chunk]
        x0 = torch.randn(len(pc), 4, latent_size, latent_size, device=device)
        y1 = torch.tensor([p[0] for p in pc], device=device)
        y2 = torch.tensor([p[1] for p in pc], device=device)
        oc = gd_compose(model, x0, y1, y2, stepsize=args.stepsize,
                        num_steps=args.num_sampling_steps, device=device)
        outs.append(oc)
    out = torch.cat(outs, dim=0)
    out_px = decode(out, vae)

    metrics = {"mode": "compose", "tag": args.tag,
               "pairs": [f"{a}+{b}" for a, b in pairs]}
    # optional: pretrained classifier joint-hit
    try:
        from torchvision.models import resnet50, ResNet50_Weights
        clf = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
        norm = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        with torch.no_grad():
            logits = clf(norm(out_px))
            top5 = logits.topk(5, dim=1).indices.tolist()
        hits = []
        for i, (a, b) in enumerate(pairs):
            hits.append({"pair": f"{a}+{b}",
                         "c1_in_top5": a in top5[i],
                         "c2_in_top5": b in top5[i],
                         "both": a in top5[i] and b in top5[i]})
        metrics["classifier_hits"] = hits
        metrics["both_rate"] = float(np.mean([h["both"] for h in hits]))
        print(f"[compose] both-class top5 rate={metrics['both_rate']:.2f}")
    except Exception as e:
        print(f"[warn] classifier eval skipped ({e})", file=sys.stderr)

    save_grid(out_px.cpu(), min(10, n),
              Path(args.out_dir) / f"compose_{args.tag}.png")
    return metrics


# --------------------------------------------------------------------------- #
# Grid output
# --------------------------------------------------------------------------- #
def save_grid(tensor, ncol, path):
    """tensor (R*ncol, 3, H, W) laid out row-major. Saves a PNG grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(tensor, nrow=ncol, padding=2)
    save_image(grid, str(path))
    print(f"[grid] wrote {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["denoise", "inpaint", "compose"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="label for output filenames, e.g. vanilla / v10")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-path",
                    default="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val")
    ap.add_argument("--model", default="EqM-B/2")
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stepsize", type=float, default=0.003)   # matches FID eval
    ap.add_argument("--num-sampling-steps", type=int, default=250)
    ap.add_argument("--gammas", default="0.3,0.5,0.7,0.9",
                    help="denoise noise fractions")
    ap.add_argument("--mask", default="box", choices=["box", "half"])
    ap.add_argument("--inpaint-clamp-start", type=float, default=0.5,
                    help="fraction of steps before clamping known region (RePaint-style)")
    ap.add_argument("--class-pairs", default="",
                    help="compose: comma list of c1:c2; empty = random --num-pairs")
    ap.add_argument("--num-pairs", type=int, default=50,
                    help="compose: number of random class pairs when --class-pairs empty")
    ap.add_argument("--compose-batch", type=int, default=25,
                    help="compose: GD batch size (memory; 2 fwd/step)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = load_eqm(args.ckpt, args.model, args.num_classes,
                     args.image_size, device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "denoise":
        metrics = run_denoise(model, vae, device, args)
    elif args.mode == "inpaint":
        metrics = run_inpaint(model, vae, device, args)
    else:
        metrics = run_compose(model, vae, device, args)

    mpath = Path(args.out_dir) / f"metrics_{args.mode}_{args.tag}.json"
    mpath.write_text(json.dumps(metrics, indent=2))
    print(f"[metrics] wrote {mpath}")


if __name__ == "__main__":
    main()
