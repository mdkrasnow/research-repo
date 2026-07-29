"""Create one immutable, disjoint held-out bank for the preliminary run."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, ToTensor, Normalize
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from train import center_crop_arr

def main():
 p=argparse.ArgumentParser(); p.add_argument('--val',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--pairs',type=int,default=64); p.add_argument('--calibration',type=int,default=256); p.add_argument('--reference',type=int,default=512); p.add_argument('--seed',type=int,default=20260729); a=p.parse_args()
 if a.out.exists(): raise SystemExit(f'refuse overwrite: {a.out}')
 g=torch.Generator().manual_seed(a.seed)
 ds=ImageFolder(a.val,transform=Compose([Lambda(lambda x:center_crop_arr(x,256)),ToTensor(),Normalize([.5]*3,[.5]*3)]))
 by={};
 for i,(_,y) in enumerate(ds.samples): by.setdefault(y,[]).append(i)
 classes=torch.randperm(len(by),generator=g)[:a.pairs].tolist(); endpoints=[]; used=set()
 # Candidate endpoints are 10 held-out examples/class; select a median DINO-distance pair.
 hub=torch.hub.load('facebookresearch/dinov2','dinov2_vits14').cuda().eval()
 for y in classes:
  cand=torch.tensor(by[y])[torch.randperm(len(by[y]),generator=g)[:10]].tolist()
  imgs=torch.stack([ds[i][0] for i in cand]).cuda()
  with torch.no_grad(): f=hub(torch.nn.functional.interpolate((imgs+1)/2,(224,224),mode='bilinear',align_corners=False)).float()
  d=torch.cdist(f,f); tri=torch.triu_indices(10,10,1); vals=d[tri[0],tri[1]]; order=vals.argsort(); q=order[len(order)*7//10]
  i,j=int(tri[0,q]),int(tri[1,q]); endpoints += [cand[i],cand[j]]; used.update((cand[i],cand[j]))
 remaining=[i for i in range(len(ds)) if i not in used]; perm=torch.tensor(remaining)[torch.randperm(len(remaining),generator=g)]
 cal=perm[:a.calibration].tolist(); ref=perm[a.calibration:a.calibration+a.reference].tolist()
 from diffusers.models import AutoencoderKL
 vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').cuda().eval()
 def images(ix): return torch.stack([ds[i][0] for i in ix])
 def lat(ix):
  chunks=[]
  with torch.no_grad():
   for s in range(0,len(ix),16): chunks.append(vae.encode(images(ix[s:s+16]).cuda()).latent_dist.mode().mul(.18215).cpu())
  return torch.cat(chunks)
 ep=images(endpoints); bank={'calibration_latents':lat(cal),'calibration_labels':torch.tensor([ds[i][1] for i in cal]),'reference_images':images(ref),'endpoint_latents':lat(endpoints),'endpoint_labels':torch.tensor([ds[i][1] for i in endpoints]),'endpoint_images':ep,'pairs':torch.arange(2*a.pairs).reshape(a.pairs,2),'metadata':{'seed':a.seed,'val':str(a.val),'pairs':a.pairs,'calibration':a.calibration,'reference':a.reference,'endpoint_indices':endpoints,'calibration_indices':cal,'reference_indices':ref,'pair_selection':'within-class DINOv2 candidate pair at 70th percentile'}}
 a.out.parent.mkdir(parents=True,exist_ok=False); torch.save(bank,a.out); print(json.dumps(bank['metadata']))
if __name__=='__main__': main()
