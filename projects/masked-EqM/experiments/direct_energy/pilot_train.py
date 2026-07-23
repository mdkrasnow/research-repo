"""Matched short pilot for none, dot, and direct EqM fields."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from models import EqM_models
from experiments.direct_energy.metrics import append_jsonl


def grad_norm(params):
    terms = [p.grad.detach().norm().square() for p in params if p.grad is not None]
    return torch.stack(terms).sum().sqrt().item() if terms else 0.0


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    if args.ebm != "none":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    metrics = output / "metrics.jsonl"; metrics.unlink(missing_ok=True)
    transform = transforms.Compose([transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size), transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3)])
    dataset = ImageFolder(args.data_path, transform=transform)
    images, labels = zip(*(dataset[i] for i in range(args.batch_size)))
    with torch.no_grad():
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        x1 = vae.encode(torch.stack(images).to(device)).latent_dist.sample().mul_(.18215)
    del vae
    model = EqM_models[args.model](input_size=args.image_size//8, num_classes=args.num_classes, uncond=True, ebm=args.ebm).to(device)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    eps = torch.randn_like(x1); t = torch.linspace(.1,.9,args.batch_size,device=device); y=torch.tensor(labels,device=device)
    xt=(t[:,None,None,None]*x1+(1-t[:,None,None,None])*eps).detach(); c=torch.minimum(torch.ones_like(t),5-5*t)*4; target=(x1-eps)*c[:,None,None,None]
    checkpoints={max(1,round(args.steps*f/100)) for f in (1,5,10,25,50,100)}
    torch.cuda.reset_peak_memory_stats(device); started=time.time()
    context=torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if args.ebm!="none" else torch.no_grad()
    # no_grad context would suppress vanilla training; use null context for none.
    from contextlib import nullcontext
    context=torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if args.ebm!="none" else nullcontext()
    with context:
      for step in range(1,args.steps+1):
        now=time.time(); opt.zero_grad(set_to_none=True)
        if args.ebm=="none": field=model(xt,t,y,train=False); energy=None
        else: field,energy=model(xt,t,y,get_energy=True,train=True)
        loss=(field-target).square().mean(); finite=bool(torch.isfinite(loss) and torch.isfinite(field).all())
        if not finite: raise RuntimeError("nonfinite pilot metric")
        loss.backward(); head=grad_norm(model.energy_head.parameters()) if args.ebm=="direct" else 0.; back=grad_norm(model.x_embedder.parameters()); opt.step()
        fn=field.detach().flatten(1).norm(dim=1); tn=target.flatten(1).norm(dim=1)
        rec={"step":step,"loss":loss.item(),"finite":finite,"field_target_cosine":F.cosine_similarity(field.detach().flatten(1),target.flatten(1)).mean().item(),"field_target_norm_ratio":(fn/tn.clamp_min(1e-12)).mean().item(),"field_norm":fn.mean().item(),"head_grad_norm":head,"backbone_grad_norm":back,"peak_memory_mb":torch.cuda.max_memory_allocated()/2**20,"steps_per_sec":1/(time.time()-now),"energy_mean":energy.detach().mean().item() if energy is not None else None}
        append_jsonl(metrics,rec)
        if step in checkpoints: torch.save({"model":model.state_dict(),"args":vars(args),"step":step},output/f"step{step:04d}.pt")
    summary={"ebm":args.ebm,"seed":args.seed,"steps":args.steps,"runtime_seconds":time.time()-started,"peak_memory_mb":torch.cuda.max_memory_allocated()/2**20}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--data-path",required=True);p.add_argument("--output",required=True);p.add_argument("--ebm",choices=["none","dot","direct"],required=True);p.add_argument("--model",default="EqM-S/2",choices=list(EqM_models.keys()));p.add_argument("--image-size",type=int,default=256);p.add_argument("--num-classes",type=int,default=1000);p.add_argument("--vae",default="ema");p.add_argument("--batch-size",type=int,default=2);p.add_argument("--steps",type=int,default=1000);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--seed",type=int,default=0);main(p.parse_args())
