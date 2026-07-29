"""Run the single-seed preliminary benchmark.  All checkpoint paths are explicit.

This intentionally avoids a hidden checkpoint-selection loop: one epoch-15 EMA
checkpoint per arm is supplied on the command line and recorded with SHA-256.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energy_monotonicity.evaluate_energy_monotonicity import CheckpointRecord, load_model, sha256_file
from energy_monotonicity.core import (get_effective_field, _raw_backbone_output,
    _direct_tokens, _one_scalar_per_sample)
from geodesic_manifold.core import (Calibration, calibrate_linear, lambda_from_energy,
    open_uniform_cubic_basis, path_from_controls, kinetic_objective, kth_neighbor_radii,
    normalized_manifold_metrics, paired_bootstrap)

SIGNS = {"dot": -1, "direct": 1}

def scalar_energy(model, variant, z, labels):
    """Detached scalar for calibration/reporting only."""
    result = get_effective_field(model, variant, z, labels)
    assert result.scalar_energy is not None
    return SIGNS[variant] * result.scalar_energy

def differentiable_scalar_energy(model, variant, z, labels):
    """Canonical scalar energy retaining the input graph for geodesic descent.

    ``get_effective_field`` deliberately detaches all diagnostic outputs.  That is
    correct for checkpoint evaluation, but using it here would erase the metric
    gradient and silently turn the solver into Euclidean interpolation.
    """
    gamma = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    if variant == "dot":
        raw = _raw_backbone_output(model, z, gamma, labels)
        scalar = (z * raw).flatten(1).sum(1)
    elif variant == "direct":
        tokens, conditioning = _direct_tokens(model, z, gamma, labels)
        scalar = _one_scalar_per_sample(model.energy_head(tokens, conditioning), z.shape[0])
    else:
        raise ValueError(f"scalar energy unavailable for {variant}")
    return SIGNS[variant] * scalar

def none_gradient_norm(model, z, labels):
    """Secondary control potential h(x)=||f_none(x)||², never an energy."""
    gamma = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    field = model(z, gamma, labels)
    return field.flatten(1).square().sum(1)

def metric_potential(model, variant, z, labels, differentiable=False):
    if variant == "none":
        return none_gradient_norm(model, z, labels)
    return (differentiable_scalar_energy if differentiable else scalar_energy)(model, variant, z, labels)

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

def frechet_distance(real: np.ndarray, generated: np.ndarray) -> float:
    """FID in the frozen Inception feature space, for interior path points."""
    from scipy import linalg
    mean_real, mean_generated = real.mean(0), generated.mean(0)
    cov_real = np.cov(real, rowvar=False); cov_generated = np.cov(generated, rowvar=False)
    root, _ = linalg.sqrtm(cov_real @ cov_generated, disp=False)
    if np.iscomplexobj(root): root = root.real
    return float((mean_real - mean_generated).dot(mean_real - mean_generated) +
                 np.trace(cov_real + cov_generated - 2 * root))

def write_pair_artifacts(out: Path, full: dict[str, dict[str, np.ndarray]]) -> None:
    """Persist the requested paired direct-to-dot detour/excess tradeoff data."""
    direct, dot = full["dinov2_direct"], full["dinov2_dot"]
    with (out / "paired_tradeoff.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair", "method", "detour", "excess", "precision", "d_rmse"])
        writer.writeheader()
        for pair in range(len(direct["excess"])):
            for method, value in (("direct", direct), ("dot", dot)):
                writer.writerow({"pair":pair,"method":method, **{key:float(value[key][pair]) for key in ("detour","excess","precision","d_rmse")}})
    # A CSV is the authoritative plot data; also emit the requested visual artifact.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6, 5))
    for pair in range(len(direct["excess"])):
        axis.plot([dot["detour"][pair], direct["detour"][pair]], [dot["excess"][pair], direct["excess"][pair]], color="0.75", linewidth=.6)
    axis.scatter(dot["detour"], dot["excess"], s=14, label="dot", color="#d95f02")
    axis.scatter(direct["detour"], direct["excess"], s=14, label="direct", color="#1b9e77")
    axis.set(xlabel="DINOv2 feature detour ratio", ylabel="normalized manifold excess")
    axis.legend(); figure.tight_layout(); figure.savefig(out / "paired_tradeoff.png", dpi=180); plt.close(figure)

def optimize(paths, model, variant, labels, calibration, restarts, steps, lr):
    """Minimize the exact discrete objective with bounded activation memory.

    A 64-path batch retains every transformer activation until one joint
    backward call and exceeds even a 140-GB H200.  Each term is separable by
    endpoint pair, so accumulate its (mean-normalized) gradient one path at a
    time before exactly one Adam step.  This does not alter the objective,
    endpoints, restart policy, or number of optimizer steps.
    """
    basis=open_uniform_cubic_basis().to(paths); endpoints=paths[:, [0,-1]].detach(); best=None
    t_controls=torch.linspace(0,1,10,device=paths.device)
    def controls_for(raw, index):
        endpoint=endpoints[index:index+1]
        linear=torch.stack([endpoint[:,0]*(1-t)+endpoint[:,1]*t for t in t_controls],1)
        return torch.cat([endpoint[:,:1], linear[:,1:-1]+.5*torch.tanh(raw[index:index+1]), endpoint[:,1:]],1)
    def objectives_for(raw, differentiable):
        for index in range(len(paths)):
            part=path_from_controls(controls_for(raw,index),basis)
            y=labels[index:index+1]
            objective,_=kinetic_objective(part,lambda z: lambda_from_energy(metric_potential(model,variant,z.flatten(0,1),y[:,None].expand(-1,32).reshape(-1),differentiable=differentiable).reshape(z.shape[:2]),calibration))
            yield objective
    for restart in range(restarts):
        raw=torch.zeros((len(paths),8,*paths.shape[2:]),device=paths.device,requires_grad=True)
        if restart: raw.data.normal_(0,.01)
        opt=torch.optim.Adam([raw],lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            # Backward immediately so each transformer's activation graph frees.
            for objective in objectives_for(raw,differentiable=True):
                (objective / len(paths)).backward()
            opt.step()
        with torch.no_grad():
            scores=torch.cat([value.detach() for value in objectives_for(raw,differentiable=False)])
            controls=torch.cat([controls_for(raw,index) for index in range(len(paths))])
            candidate=path_from_controls(controls,basis).detach()
        best=(candidate, scores) if best is None else (torch.where((scores<best[1])[:,None,None,None,None],candidate,best[0]), torch.minimum(scores,best[1]))
    return best

def main(argv=None):
  p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); args=p.parse_args(argv)
  c=json.loads(args.config.read_text()); out=Path(c["output"]); out.mkdir(parents=True,exist_ok=True)
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Bank is a deliberate, disjoint tensor artifact created by the companion bank builder.
  bank=torch.load(c["bank"],map_location="cpu",weights_only=True)
  required={"calibration_latents","calibration_labels","reference_images","endpoint_latents","endpoint_labels","endpoint_images","pairs"}
  if required-set(bank): raise ValueError(f"bank missing {required-set(bank)}")
  records={v:make_record(Path(c["checkpoints"][v]),v) for v in ("dot","direct","none")}
  models={v:load_model(records[v],device,torch.float32) for v in records}
  calibrations={}; unavailable={}
  for v in models:
    z=bank["calibration_latents"].to(device); y=bank["calibration_labels"].to(device)
    on=metric_potential(models[v],v,z,y); off=metric_potential(models[v],v,(z[::2]+z[1::2])/2,y[::2])
    try:
      calibrations[v]=calibrate_linear(on,off)
    except ValueError as error:
      if v != "none": raise
      # This is a labelled secondary control; it must not block scalar-energy arms.
      unavailable["none_gradient_norm"]=str(error)
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
    if v not in calibrations: continue
    mids=(initial[:,:-1]+initial[:,1:])/2
    initial_parts=[]
    for start in range(0, len(mids), 2):
      stop=min(start+2,len(mids)); part=mids[start:stop]; y=pair_labels[start:stop]
      initial_parts.append(metric_potential(models[v],v,part.flatten(0,1),y[:,None].expand(-1,32).reshape(-1)).reshape(part.shape[:2]))
    initial_energy=torch.cat(initial_parts)
    if (lambda_from_energy(initial_energy,calibrations[v]) <= 0).any():
      # Registered secondary metric: same calibration endpoints, positivity guaranteed.
      calibrations[v]=Calibration(calibrations[v].mean_on,calibrations[v].mean_off,calibrations[v].alpha,calibrations[v].beta,"exp")
      metric_fallback[v]="linear metric non-positive on the fixed initial path; used preregistered exponential secondary"
    paths["none_gradient_norm" if v == "none" else v],_=optimize(initial,models[v],v,pair_labels,calibrations[v],c["restarts"],c["steps"],c["lr"])
  # Decode once; fixed feature encoders never take part in restart selection.
  from diffusers.models import AutoencoderKL
  vae=AutoencoderKL.from_pretrained(c["vae"]).to(device).eval()
  decoded={}
  with torch.no_grad():
    for k,v in paths.items():
      decoded[k]=vae.decode(v.reshape(-1,*v.shape[2:])/.18215).sample.cpu().reshape(len(v),33,3,256,256)
  table=[]; full={}; feature_cache={}
  for feature_name in ("dinov2","inception"):
    fm,pre,_=feature_model(feature_name,device); ref=features(fm,pre,bank["reference_images"],device); radii=kth_neighbor_radii(ref)
    for name,path in decoded.items():
      path_features=features(fm,pre,path.flatten(0,1),device).reshape(len(path),33,-1)
      metric=normalized_manifold_metrics(path_features,ref,radii); full[f"{feature_name}_{name}"]=metric
      feature_cache[f"{feature_name}_{name}"]=path_features
      row={"feature":feature_name,"method":name,"solver_success":1.0,**{k:float(v.mean()) for k,v in metric.items() if k!="rho"}}
      if feature_name == "inception": row["interior_fid"]=frechet_distance(ref,path_features[:,1:32].reshape(-1,path_features.shape[-1]))
      table.append(row)
  dot=full["dinov2_dot"]["excess"]; direct=full["dinov2_direct"]["excess"]; boot=paired_bootstrap(dot,direct,c["bootstrap"],c["seed"])
  write_pair_artifacts(out,full)
  result={"single_seed_preliminary":True,"checkpoints":{v:records[v].__dict__ for v in records},"calibration":{v:calibrations[v].__dict__ for v in calibrations},"linear_metric_infeasibility":metric_fallback,"unavailable_secondary_controls":unavailable,"rows":table,"direct_dot_relative_improvement":{"mean":float(((dot-direct)/np.maximum(dot,1e-12)).mean()),"ci95":np.quantile(boot,[.025,.975]).tolist()},"artifacts":{"paired_tradeoff_csv":"paired_tradeoff.csv","paired_tradeoff_plot":"paired_tradeoff.png"},"warning":"No seed-level or preregistered pass/fail inference: only one matched checkpoint per method; any exponential-metric result is secondary. none_gradient_norm is a labelled gradient-field control, not an energy-value comparison."}
  (out/"summary.json").write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+"\n")

if __name__ == "__main__": main()
