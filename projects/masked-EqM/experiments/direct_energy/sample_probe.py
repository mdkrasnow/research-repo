"""Cheap matched fixed-seed sampler probe for pilot checkpoints."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
PROJECT_ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT_ROOT))
from models import EqM_models

def main(a):
 torch.manual_seed(a.seed); device=torch.device("cuda"); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if a.ebm!="none": torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 model=EqM_models[a.model](input_size=a.image_size//8,num_classes=a.num_classes,uncond=True,ebm=a.ebm).to(device)
 state=torch.load(a.ckpt,map_location=device);model.load_state_dict(state["model"]);model.eval()
 for p in model.parameters():p.requires_grad_(False)
 z=torch.randn(a.num_samples,4,a.image_size//8,a.image_size//8,device=device); y=torch.arange(a.num_samples,device=device)%a.num_classes; t=torch.ones(a.num_samples,device=device); trajectory=[];start=time.time()
 ctx=torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if a.ebm!="none" else __import__('contextlib').nullcontext()
 with ctx:
  for step in range(a.steps-1):
   with torch.set_grad_enabled(a.ebm!="none"):
    field,energy=model(z,t,y,get_energy=True,train=False)
   field=field.detach(); z=(z+a.stepsize*field).detach(); t=t+a.stepsize
   trajectory.append({"step":step+1,"latent_norm":z.flatten(1).norm(dim=1).mean().item(),"field_norm":field.flatten(1).norm(dim=1).mean().item(),"energy_mean":energy.detach().mean().item() if torch.is_tensor(energy) else None,"finite":bool(torch.isfinite(z).all())})
 with torch.no_grad():
  vae=AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{a.vae}").to(device);images=vae.decode(z/.18215).sample
 for i,img in enumerate(images):save_image(img,out/f"{i:03d}.png",normalize=True,value_range=(-1,1))
 (out/"trajectory.json").write_text(json.dumps({"ebm":a.ebm,"runtime_seconds":time.time()-start,"trajectory":trajectory},indent=2)+"\n")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--ckpt",required=True);p.add_argument("--output",required=True);p.add_argument("--ebm",choices=["none","dot","direct"],required=True);p.add_argument("--model",default="EqM-S/2",choices=list(EqM_models.keys()));p.add_argument("--image-size",type=int,default=256);p.add_argument("--num-classes",type=int,default=1000);p.add_argument("--vae",default="ema");p.add_argument("--num-samples",type=int,default=16);p.add_argument("--steps",type=int,default=50);p.add_argument("--stepsize",type=float,default=.003);p.add_argument("--seed",type=int,default=123);main(p.parse_args())
