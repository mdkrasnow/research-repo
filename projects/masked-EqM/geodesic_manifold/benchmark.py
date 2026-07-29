"""Run the single-seed preliminary benchmark.  All checkpoint paths are explicit.

This intentionally avoids a hidden checkpoint-selection loop: one epoch-15 EMA
checkpoint per arm is supplied on the command line and recorded with SHA-256.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energy_monotonicity.evaluate_energy_monotonicity import CheckpointRecord, load_model, sha256_file
from energy_monotonicity.core import get_effective_field
from geodesic_manifold.core import (Calibration, calibrate_linear, lambda_from_energy,
    open_uniform_cubic_basis, path_from_controls, kinetic_objective, kth_neighbor_radii,
    normalized_manifold_metrics, paired_bootstrap)

SIGNS = {"dot": -1, "direct": 1}

def scalar_energy(model, variant, z, labels):
    result = get_effective_field(model, variant, z, labels)
    assert result.scalar_energy is not None
    return SIGNS[variant] * result.scalar_energy

def make_record(path: Path, variant: str) -> CheckpointRecord:
    state = torch.load(path, map_location="cpu", weights_only=False); a = state["args"]
    get = lambda k, d=None: a.get(k, d) if isinstance(a, dict) else getattr(a, k, d)
    if get("ebm") != variant: raise ValueError(f"{path} is not ebm={variant}")
    return CheckpointRecord(variant, int(state["epoch"]), str(path), str(path.parent), None,
      int(state.get("step", -1)), "ema" in state, "ema", get("model", "EqM-B/2"),
      int(get("image_size",256)), int(get("num_classes",1000)), bool(get("uncond",True)),
      get("corruption_mode", "gaussian"), "explicit CLI checkpoint", 0, get("git_sha"), path.parent.name, sha256_file(path))

def encode_images(images, vae, device):
    with torch.no_grad(): return vae.encode(images.to(device)).latent_dist.mode().mul(.18215)

def feature_model(name, device):
    from torchvision.models import inception_v3, Inception_V3_Weights
    if name == "inception":
        weights = Inception_V3_Weights.DEFAULT; model = inception_v3(weights=weights, aux_logits=False)
        # Pool features, not the 1,000-class ImageNet classifier logits.
        model.fc = torch.nn.Identity(); model = model.to(device).eval()
        transform = weights.transforms()
        return model, lambda x: transform(((x + 1) / 2)), 2048
    if name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        return model, lambda x: torch.nn.functional.interpolate((x + 1) / 2, (224,224), mode="bilinear", align_corners=False), 384
    raise ValueError(name)

def features(model, preprocess, images, device, batch=16):
    out=[]
    with torch.no_grad():
      for start in range(0,len(images),batch):
        x=preprocess(images[start:start+batch].to(device)); y=model(x)
        if isinstance(y, tuple): y=y[0]
        if y.ndim > 2: y=y.flatten(1)
        out.append(y.float().cpu())
    return torch.cat(out).numpy()

def optimize(paths, model, variant, labels, calibration, restarts, steps, lr, path_batch=2):
    basis=open_uniform_cubic_basis().to(paths); endpoints=paths[:, [0,-1]].detach(); best=None
    for restart in range(restarts):
        # bounded residual around linear controls; endpoints remain exact.
        raw=torch.zeros((len(paths),8,*paths.shape[2:]),device=paths.device,requires_grad=True)
        if restart: raw.data.normal_(0,.01)
        opt=torch.optim.Adam([raw],lr=lr)
        for _ in range(steps):
            linear=torch.stack([endpoints[:,0]*(1-t)+endpoints[:,1]*t for t in torch.linspace(0,1,10,device=paths.device)],1)
            controls=torch.cat([endpoints[:,:1], linear[:,1:-1]+.5*torch.tanh(raw), endpoints[:,1:]],1)
            path=path_from_controls(controls,basis)
            objectives=[]
            for start in range(0, len(path), path_batch):
                stop=min(start+path_batch,len(path)); part=path[start:stop]; y=labels[start:stop]
                objective,_=kinetic_objective(part,lambda z: lambda_from_energy(scalar_energy(model,variant,z.flatten(0,1),y[:,None].expand(-1,32).reshape(-1)).reshape(z.shape[:2]),calibration))
                objectives.append(objective)
            objective=torch.cat(objectives)
            opt.zero_grad(); objective.mean().backward(); opt.step()
        candidate=(path.detach(),objective.detach())
        best=candidate if best is None else (torch.where((candidate[1]<best[1])[:,None,None,None,None],candidate[0],best[0]), torch.minimum(candidate[1],best[1]))
    return best

def main(argv=None):
  p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); args=p.parse_args(argv)
  c=json.loads(args.config.read_text()); out=Path(c["output"]); out.mkdir(parents=True,exist_ok=True)
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Bank is a deliberate, disjoint tensor artifact created by the companion bank builder.
  bank=torch.load(c["bank"],map_location="cpu",weights_only=True)
  required={"calibration_latents","calibration_labels","reference_images","endpoint_latents","endpoint_labels","endpoint_images","pairs"}
  if required-set(bank): raise ValueError(f"bank missing {required-set(bank)}")
  records={v:make_record(Path(c["checkpoints"][v]),v) for v in ("dot","direct")}
  models={v:load_model(records[v],device,torch.float32) for v in records}
  calibrations={}
  for v in models:
    z=bank["calibration_latents"].to(device); y=bank["calibration_labels"].to(device)
    on=scalar_energy(models[v],v,z,y); off=scalar_energy(models[v],v,(z[::2]+z[1::2])/2,y[::2])
    calibrations[v]=calibrate_linear(on,off)
  pairs=bank["pairs"].long(); lat=bank["endpoint_latents"]; labels=bank["endpoint_labels"]
  endpoints=torch.stack([lat[pairs[:,0]],lat[pairs[:,1]]],1).to(device); pair_labels=labels[pairs[:,0]].to(device)
  initial=torch.lerp(endpoints[:,0:1],endpoints[:,1:2],torch.linspace(0,1,33,device=device)[None,:,None,None,None])
  paths={"linear":initial.detach()}
  # Model-free spherical interpolation is a labelled geometry control, not an energy method.
  flat=endpoints.flatten(2); a,b=flat[:,0],flat[:,1]; cosine=(a*b).sum(1)/(a.norm(dim=1)*b.norm(dim=1)).clamp_min(1e-12)
  theta=cosine.clamp(-.999999,.999999).acos(); t=torch.linspace(0,1,33,device=device)[None,:,None]
  slerp=(torch.sin((1-t)*theta[:,None,None])*a[:,None]+torch.sin(t*theta[:,None,None])*b[:,None])/torch.sin(theta)[:,None,None]
  paths["slerp"]=slerp.reshape_as(initial).detach()
  metric_fallback={}
  for v in models:
    mids=(initial[:,:-1]+initial[:,1:])/2
    initial_energy=scalar_energy(models[v],v,mids.flatten(0,1),pair_labels[:,None].expand(-1,32).reshape(-1)).reshape(mids.shape[:2])
    if (lambda_from_energy(initial_energy,calibrations[v]) <= 0).any():
      # Registered secondary metric: same calibration endpoints, positivity guaranteed.
      calibrations[v]=Calibration(calibrations[v].mean_on,calibrations[v].mean_off,calibrations[v].alpha,calibrations[v].beta,"exp")
      metric_fallback[v]="linear metric non-positive on the fixed initial path; used preregistered exponential secondary"
    paths[v],_=optimize(initial,models[v],v,pair_labels,calibrations[v],c["restarts"],c["steps"],c["lr"])
  # Decode once; fixed feature encoders never take part in restart selection.
  from diffusers.models import AutoencoderKL
  vae=AutoencoderKL.from_pretrained(c["vae"]).to(device).eval()
  decoded={}
  with torch.no_grad():
    for k,v in paths.items():
      decoded[k]=vae.decode(v.reshape(-1,*v.shape[2:])/.18215).sample.cpu().reshape(len(v),33,3,256,256)
  table=[]; full={}
  for feature_name in ("dinov2","inception"):
    fm,pre,_=feature_model(feature_name,device); ref=features(fm,pre,bank["reference_images"],device); radii=kth_neighbor_radii(ref)
    for name,path in decoded.items():
      metric=normalized_manifold_metrics(features(fm,pre,path.flatten(0,1),device).reshape(len(path),33,-1),ref,radii); full[f"{feature_name}_{name}"]=metric
      table.append({"feature":feature_name,"method":name,**{k:float(v.mean()) for k,v in metric.items() if k!="rho"}})
  dot=full["dinov2_dot"]["excess"]; direct=full["dinov2_direct"]["excess"]; boot=paired_bootstrap(dot,direct,c["bootstrap"],c["seed"])
  result={"single_seed_preliminary":True,"checkpoints":{v:records[v].__dict__ for v in records},"calibration":{v:calibrations[v].__dict__ for v in calibrations},"linear_metric_infeasibility":metric_fallback,"rows":table,"direct_dot_relative_improvement":{"mean":float(((dot-direct)/np.maximum(dot,1e-12)).mean()),"ci95":np.quantile(boot,[.025,.975]).tolist()},"warning":"No seed-level or preregistered pass/fail inference: only one matched checkpoint per method; any exponential-metric result is secondary."}
  (out/"summary.json").write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+"\n")

if __name__ == "__main__": main()
