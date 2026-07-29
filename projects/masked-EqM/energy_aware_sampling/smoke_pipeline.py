"""Tiny end-to-end smoke: sampling arms, VAE decode, FID, and cost records."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from diffusers.models import AutoencoderKL
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.utils import save_image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from models import EqM_models
from energy_aware_sampling.core import fixed_sample, armijo_sample, replay_sample

def model(path, ebm, device):
 m=EqM_models['EqM-B/2'](input_size=32,num_classes=1000,uncond=True,ebm=ebm).to(device); m.load_state_dict(torch.load(path,map_location=device)['ema']);m.eval()
 for p in m.parameters():p.requires_grad_(False)
 return m
def sample_batches(fn,m,x,y,steps,step,batch,schedule=None):
 out=[]; cost={'gradient_evaluations':0,'energy_forwards':0,'max_backtrack_samples':0};schedule=None
 for i in range(0,len(x),batch):
  z,s=fn(m,x[i:i+batch],y[i:i+batch],steps,step) if fn is not replay_sample else fn(m,x[i:i+batch],y[i:i+batch],schedule)
  if schedule is None and 'accepted_step_median_by_iteration' in s:schedule=s['accepted_step_median_by_iteration']
  out.append(z.cpu());
  for k in cost:cost[k]+=s.get(k,0)
 return torch.cat(out),cost,schedule
def main(a):
 d=torch.device('cuda');torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
 g=torch.Generator().manual_seed(a.seed);x=torch.randn(a.samples,4,32,32,generator=g);y=torch.arange(a.samples)%1000;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 ds=ImageFolder(a.real,transform=Compose([Resize(256),CenterCrop(256)]));real=out/'real';real.mkdir(exist_ok=True)
 for i in range(a.samples): ds[i][0].save(real/f'{i:06d}.png')
 vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ema').to(d).eval();[p.requires_grad_(False) for p in vae.parameters()]
 arms=[('direct_fixed','direct',fixed_sample),('direct_armijo','direct',armijo_sample),('direct_replay','direct',replay_sample),('dot_fixed','dot',fixed_sample),('dot_armijo','dot',armijo_sample)]
 records=[]; direct_schedule=None
 for name,ebm,fn in arms:
  m=model(a.direct if ebm=='direct' else a.dot,ebm,d);start=time.time()
  if fn is replay_sample:
   if direct_schedule is None: raise RuntimeError('missing direct Armijo schedule')
   z,c,_=sample_batches(fn,m,x.to(d),y.to(d),a.steps,a.step,a.batch,direct_schedule)
  else:z,c,s=sample_batches(fn,m,x.to(d),y.to(d),a.steps,a.step,a.batch);direct_schedule=s if name=='direct_armijo' else direct_schedule
  folder=out/name;folder.mkdir(exist_ok=True)
  with torch.no_grad(): imgs=vae.decode(z.to(d)/.18215).sample
  for i,img in enumerate(imgs):save_image(img,folder/f'{i:06d}.png',normalize=True,value_range=(-1,1))
  fid=calculate_fid_given_paths([str(real),str(folder)],batch_size=a.fid_batch,device=d,dims=2048);records.append({'arm':name,'fid_smoke':fid,'wall_seconds':time.time()-start,**c});del m;torch.cuda.empty_cache()
 (out/'metrics.json').write_text(json.dumps({'samples':a.samples,'steps':a.steps,'records':records},indent=2)+'\n');print(json.dumps(records,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--direct',required=True);p.add_argument('--dot',required=True);p.add_argument('--real',required=True);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=32);p.add_argument('--steps',type=int,default=10);p.add_argument('--step',type=float,default=.0017);p.add_argument('--batch',type=int,default=4);p.add_argument('--fid-batch',type=int,default=16);p.add_argument('--seed',type=int,default=20260729);main(p.parse_args())
