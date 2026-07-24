"""Frozen-field, observation-constrained EqM reconstruction evaluation.

This deliberately reuses the update convention from eval_masked_recovery.py:
the learned vector field is an update direction and is never called an energy.
"""
import argparse, hashlib, json, math, os, time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

# Keep deterministic-mask/projection tests runnable on developer machines that
# do not carry the cluster's diffusion stack.
try:
    from diffusers.models import AutoencoderKL
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from download import find_model
    from models import EqM_models
except ImportError:
    AutoencoderKL = transforms = ImageFolder = find_model = EqM_models = None

try:
    import lpips
except ImportError:
    lpips = None

LATENT_SCALE = 0.18215


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16) % (2**63 - 1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def block_mask(h, w, missing_fraction, seed):
    """Binary V mask with an aspect-ratio-log-uniform rectangular hole."""
    g = torch.Generator().manual_seed(seed)
    ratio = float(torch.exp(torch.empty(1).uniform_(math.log(.5), math.log(2), generator=g)).item())
    area = max(1, min(h * w, round(h * w * missing_fraction)))
    candidates=[(abs(a*b-area),a,b) for a in range(1,h+1) for b in range(1,w+1)]
    _, bh, bw = min(candidates, key=lambda q:(q[0],abs(math.log((q[2]/q[1])/ratio))))
    top = int(torch.randint(0, h - bh + 1, (1,), generator=g))
    left = int(torch.randint(0, w - bw + 1, (1,), generator=g))
    v = torch.ones(1, h, w)
    v[:, top:top + bh, left:left + bw] = 0
    return v, {"block_area_fraction": bh * bw / (h * w), "top": top, "left": left, "height": bh, "width": bw, "aspect_ratio": bw / bh}


def fixed_visible(h, w, fraction, generator, available=None):
    """Uniform subset conditional on an exact requested visible cardinality."""
    available = torch.ones(h*w, dtype=torch.bool) if available is None else available.flatten().bool()
    locations = available.nonzero().flatten(); keep = round(h*w*fraction)
    if keep > len(locations): raise ValueError("target visible fraction exceeds available coordinates")
    chosen = locations[torch.randperm(len(locations), generator=generator)[:keep]]
    out=torch.zeros(h*w); out[chosen]=1
    return out.reshape(1,h,w)


def stroke_mask(h, w, target_missing, seed):
    """Fast deterministic free-form control: smoothed random field thresholded
    at an exact cardinality, yielding connected irregular regions."""
    g = torch.Generator().manual_seed(seed)
    budget = max(1, round((1 - target_missing) * h * w))
    # Coarse random field upsampled to latent resolution: an existing-style
    # patchy/free-form control without costly per-mask iterative strokes.
    ch,cw=max(2,math.ceil(h/4)),max(2,math.ceil(w/4))
    field=torch.rand(ch,cw,generator=g).repeat_interleave(4,0).repeat_interleave(4,1)[:h,:w]
    field=(field + torch.rand(h,w,generator=g)*1e-4).flatten()
    holes=torch.topk(field,budget).indices
    v=torch.ones(h*w); v[holes]=0; v=v.reshape(1,h,w)
    return v, {"block_area_fraction": 0.0, "stroke_missing_fraction": float((v == 0).float().mean())}


def visibility_mask(family, h, w, target_visible, seed, block_missing=.20):
    """Returns V=1 visible, exactly deterministic from manifest seed."""
    g = torch.Generator().manual_seed(seed)
    if family == "bernoulli":
        v = fixed_visible(h,w,target_visible,g); meta = {"block_area_fraction": 0.0}
    elif family == "block":
        v, meta = block_mask(h, w, 1 - target_visible, seed)
    elif family == "combined":
        if not 0 <= target_visible / (1 - block_missing) <= 1: raise ValueError("invalid combined visible fraction")
        b, meta = block_mask(h, w, block_missing, seed)
        v = fixed_visible(h,w,target_visible,g,b)
    elif family == "irregular":
        v, meta = stroke_mask(h, w, target_visible, seed)
    else: raise ValueError(f"unknown family {family}")
    meta.update({"actual_visible_fraction": float(v.mean()), "mask_hash": hashlib.sha256(v.numpy().tobytes()).hexdigest()})
    return v, meta


def make_manifest(path, dataset_length, num_samples, split, families, visible_fracs, latent_hw=32, seed=20260724, index_offset=0):
    # One permutation with non-overlapping offsets is the split contract.
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_length, generator=g)[index_offset:index_offset + num_samples].tolist()
    if len(indices) != num_samples: raise ValueError("manifest exceeds dataset")
    rows = []
    for index in indices:
        for family in families:
            for fraction in visible_fracs:
                mseed = stable_seed(seed, split, index, family, fraction, "mask")
                _, meta = visibility_mask(family, latent_hw, latent_hw, fraction, mseed)
                rows.append({"dataset_index": index, "sample_id": f"{split}:{index}", "split": split, "mask_family": family,
                             "requested_visible_fraction": fraction, "mask_seed": mseed, "corruption_seed": stable_seed(seed,index,"fill"),
                             "initialization_seed": stable_seed(seed,index,"init"), "encode_seed": stable_seed(seed,index,"encode"), **meta})
    payload = {"schema_version": 1, "seed": seed, "rows": rows}
    payload["manifest_hash"] = digest(payload)
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(payload, indent=2))
    return payload


def project(proposed, clean, v, mode, rho=1.):
    if mode == "none": return proposed
    if mode == "hard": return v * clean + (1-v) * proposed
    if mode == "soft": return proposed + rho * v * (clean - proposed)
    raise ValueError(mode)


def shard_rows(rows, shard_id, num_shards):
    """Stable disjoint row assignment; used both by the evaluator and tests."""
    if not 0 <= shard_id < num_shards: raise ValueError("invalid shard")
    return [row for i, row in enumerate(rows) if i % num_shards == shard_id]


def sampler_recover(model, state, clean, v, y, steps, stepsize, sampler, mu, mode, rho):
    """Exact GD/NAG update order from eval_masked_recovery, then projection."""
    t = torch.ones((state.shape[0],), device=state.device, dtype=state.dtype); previous_field = torch.zeros_like(state)
    observed_max = []
    for _ in range(steps - 1):
        model_input = state if sampler == "gd" else state + stepsize * previous_field * mu
        out = model(model_input, t, y)
        if not torch.is_tensor(out): out = out[0]
        out = out.detach()
        if sampler != "gd": previous_field = out
        proposed = (state + out * stepsize).detach()
        state = project(proposed, clean, v, mode, rho).detach()
        observed_max.append(float((v * (state-clean)).abs().max().cpu()))
        t = t + stepsize
    return state, observed_max


def load_ema(path, device):
    if EqM_models is None: raise RuntimeError("cluster evaluation dependencies are unavailable")
    model = EqM_models["EqM-B/2"](input_size=32, num_classes=1000, uncond=True, ebm="none").to(device)
    ema = deepcopy(model).to(device); state = find_model(path)
    ema.load_state_dict(state.get("ema", state)); ema.eval()
    for p in ema.parameters(): p.requires_grad_(False)
    return ema


def regional_metrics(pred_latent, clean_latent, decoded, clean_img, v, lpips_fn):
    missing = 1-v; denom = missing.sum((1,2,3)).clamp_min(1) * clean_latent.shape[1]
    miss_mse = (((pred_latent-clean_latent).square()*missing).sum((1,2,3))/denom)
    obs_mse = (((pred_latent-clean_latent).square()*v).sum((1,2,3))/(v.sum((1,2,3)).clamp_min(1)*clean_latent.shape[1]))
    px_v = F.interpolate(v, size=clean_img.shape[-2:], mode="nearest"); px_m = 1-px_v
    px_denom = (px_m.sum((1,2,3)).clamp_min(1)*clean_img.shape[1])
    px_mse = ((decoded-clean_img).square()*px_v).sum((1,2,3))/(px_v.sum((1,2,3)).clamp_min(1)*clean_img.shape[1])
    composite = px_m*decoded + px_v*clean_img
    lp_full = lpips_fn(decoded.clamp(-1,1),clean_img.clamp(-1,1)).flatten() if lpips_fn else torch.full_like(miss_mse, float('nan'))
    lp_hole = lpips_fn(composite.clamp(-1,1),clean_img.clamp(-1,1)).flatten() if lpips_fn else torch.full_like(miss_mse, float('nan'))
    return miss_mse, obs_mse, ((pred_latent-clean_latent).square().mean((1,2,3))), lp_full, lp_hole, px_mse


def run(args):
    if AutoencoderKL is None: raise RuntimeError("cluster evaluation dependencies are unavailable")
    manifest = json.loads(Path(args.sample_manifest).read_text()) if args.sample_manifest else make_manifest(args.create_manifest, len(ImageFolder(args.data_path)), args.num_samples, args.manifest_split, args.mask_families.split(','), [float(x) for x in args.target_visible_fracs.split(',')], seed=args.seed)
    rows = [r for r in manifest['rows'] if r['split'] == args.manifest_split and (args.mask_family == 'all' or r['mask_family'] == args.mask_family)]
    rows = shard_rows(rows, args.shard_id, args.num_shards)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = load_ema(args.checkpoint, device); vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema').to(device).eval()
    lpfn = lpips.LPIPS(net='alex').to(device).eval() if lpips and not args.no_lpips else None
    transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    data = ImageFolder(args.data_path, transform=transform); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    outfile = output / f"{args.checkpoint_id}_{args.projection_mode}_shard{args.shard_id}.jsonl"
    done = set()
    if args.resume and outfile.exists(): done = {json.loads(x)['record_id'] for x in outfile.read_text().splitlines() if x}
    resolved = vars(args) | {'manifest_hash':manifest['manifest_hash'], 'rows':len(rows)}; print(json.dumps(resolved,indent=2,sort_keys=True))
    with outfile.open('a') as f, torch.no_grad():
        for row in rows:
            rid = digest([args.checkpoint_id,args.projection_mode,args.projection_strength,row]);
            if rid in done: continue
            start = time.time()
            x, label = data[row['dataset_index']]; x=x.unsqueeze(0).to(device); y=torch.tensor([label],device=device)
            torch.manual_seed(row['encode_seed']); clean=vae.encode(x).latent_dist.sample()*LATENT_SCALE
            v,_ = visibility_mask(row['mask_family'], clean.shape[-2], clean.shape[-1], row['requested_visible_fraction'], row['mask_seed']); v=v.to(device)
            torch.manual_seed(row['corruption_seed']); fill=torch.randn_like(clean); observed=v*clean+(1-v)*fill
            state=observed.clone(); reconstructed,invariants=sampler_recover(model,state,clean,v,y,args.steps,args.step_size,args.sampler,args.momentum,args.projection_mode,args.projection_strength)
            decoded=vae.decode(reconstructed/LATENT_SCALE).sample
            metrics=regional_metrics(reconstructed,clean,decoded,x,v,lpfn)
            rec={"record_id":rid,"experiment_id":"frozen_prior_constraint","checkpoint_id":args.checkpoint_id,"checkpoint_path":args.checkpoint,"git_commit":args.git_commit,"config_hash":digest(resolved),"class_id":label,"projection_mode":args.projection_mode,"projection_strength":args.projection_strength,"sampler":args.sampler,"sampler_steps":args.steps,"step_size":args.step_size,"momentum":args.momentum,"missing_model_mse":float(metrics[0]),"observed_model_mse":float(metrics[1]),"full_model_mse":float(metrics[2]),"lpips_full":float(metrics[3]),"lpips_missing_composite":float(metrics[4]),"observed_pixel_mse":float(metrics[5]),"hard_projection_max_abs":max(invariants,default=0.),"runtime_seconds":time.time()-start,"completion_status":"ok",**row}
            if args.strict and args.projection_mode=='hard' and rec['hard_projection_max_abs']>1e-6: raise RuntimeError(rec)
            f.write(json.dumps(rec)+"\n"); f.flush()
    print(outfile)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--checkpoint-id',required=True); p.add_argument('--data-path',required=True); p.add_argument('--sample-manifest'); p.add_argument('--create-manifest'); p.add_argument('--manifest-split',default='pilot',choices=['pilot','final']); p.add_argument('--num-samples',type=int,default=256); p.add_argument('--mask-families',default='bernoulli,block,combined,irregular'); p.add_argument('--mask-family',default='all'); p.add_argument('--target-visible-fracs',default='.5'); p.add_argument('--projection-mode',default='hard',choices=['none','hard','soft']); p.add_argument('--projection-strength',type=float,default=1.); p.add_argument('--sampler',default='gd',choices=['gd','ngd']); p.add_argument('--steps',type=int,default=250); p.add_argument('--step-size',type=float,default=.0017); p.add_argument('--momentum',type=float,default=.3); p.add_argument('--shard-id',type=int,default=0); p.add_argument('--num-shards',type=int,default=1); p.add_argument('--output-dir',required=True); p.add_argument('--seed',type=int,default=20260724); p.add_argument('--device'); p.add_argument('--git-commit',default='unknown'); p.add_argument('--resume',action='store_true'); p.add_argument('--strict',action='store_true'); p.add_argument('--no-lpips',action='store_true'); run(p.parse_args())
