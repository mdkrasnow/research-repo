"""Small fixed-bank candidate-ranking test for direct EqM scalar energy.

Quality is independent of EqM scores: DINOv2 nearest-reference similarity and
ImageNet classifier probability of the supplied class.  Lower scalar energy
and lower base-field norm are hypothesized to be better.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor
from torchvision.models import resnet50, ResNet50_Weights
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from train import center_crop_arr
from eval_masked_recovery import load_ema_model, gd_recover

SCORES=("direct_energy","direct_energy_zero_anchored","dot_energy","dot_energy_zero_anchored","base_field_norm")

def rank(x):
    order=np.argsort(x,kind="stable"); out=np.empty(len(x)); out[order]=np.arange(len(x)); return out/(max(1,len(x)-1))
def boot(fn,n,reps,seed):
    g=np.random.default_rng(seed); vals=np.array([fn(g.integers(0,n,n)) for _ in range(reps)])
    return float(fn(np.arange(n))),[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def cluster_boot(fn,source,reps,seed):
    """Resample complete source-image candidate clusters, not correlated rows."""
    source=np.asarray(source); units=np.unique(source); by=[np.flatnonzero(source==u) for u in units]; g=np.random.default_rng(seed)
    vals=[]
    for _ in range(reps):
        ix=np.concatenate([by[j] for j in g.integers(0,len(by),len(by))]); vals.append(fn(ix))
    return float(fn(np.arange(len(source)))),[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def spearman(score,quality): return float(np.corrcoef(rank(-score),rank(quality))[0,1])
def pairacc(score,quality):
    d=quality[:,None]-quality[None,:]; s=score[:,None]-score[None,:]; mask=np.triu(d!=0,1)
    return float(np.mean((d[mask]*s[mask])<0))
def scalar(model,x,t,y):
    with torch.enable_grad(): return model(x.detach().requires_grad_(True),t,y,energy_only=True,train=False).detach()
def fieldnorm(model,x,t,y):
    with torch.no_grad(): return model(x,t,y,train=False).flatten(1).norm(dim=1)
def save_img(x,path):
    from PIL import Image
    a=((x.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy(); Image.fromarray(a).save(path)

def main(a):
    cfg=json.loads(a.config.read_text()); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    d=torch.device('cuda'); out=a.output; out.mkdir(parents=True,exist_ok=False); (out/'images').mkdir()
    tf=Compose([Lambda(lambda x:center_crop_arr(x,256)),ToTensor(),Normalize([.5]*3,[.5]*3)])
    ds=ImageFolder(a.data_path,transform=tf); g=torch.Generator().manual_seed(cfg['seed'])
    perm=torch.randperm(len(ds),generator=g).tolist(); n=cfg['num_per_group']; primary=perm[:n]; refs=perm[n:n+cfg['reference_images']]
    vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema').to(d).eval()
    def encode(ix):
        xs=torch.stack([ds[i][0] for i in ix]).to(d)
        chunks=[]
        with torch.no_grad():
            for start in range(0,len(xs),8):
                chunks.append(vae.encode(xs[start:start+8]).latent_dist.mode().mul(.18215))
        z=torch.cat(chunks)
        return xs,z,torch.tensor([ds[i][1] for i in ix],device=d)
    real,rz,ry=encode(primary); ref,_,_=encode(refs)
    weights=ResNet50_Weights.IMAGENET1K_V2; cls=resnet50(weights=weights).to(d).eval()
    with torch.no_grad():
        order=cls(weights.transforms()(real.add(1).div(2))).argsort(1,descending=True)
    competing=torch.stack([order[i][order[i]!=ry[i]][:cfg.get('wrong_labels_per_real',1)] for i in range(n)],0)
    models={v:load_ema_model(cfg['checkpoints'][v],'EqM-B/2',32,1000,True,v,d) for v in ('none','dot','direct')}
    t=torch.full((n,),float(cfg['t_eval']),device=d)
    cuda_generator=torch.Generator(device=d).manual_seed(cfg['seed'] + 1)
    noise=torch.randn(rz.shape,generator=cuda_generator,device=d)
    rows=[]
    def add(group,x,z,y,source,severity=0.,wrong=False):
        for i in range(len(x)):
            name=f'{len(rows):04d}_{group}.png'; save_img(x[i],out/'images'/name)
            rows.append({'candidate_id':len(rows),'image_file':f'images/{name}','group':group,'source_id':int(source[i]),'severity':float(severity),'label':int(y[i]),'wrong_label':bool(wrong),'latent':z[i].detach().cpu(),'image':x[i].detach().cpu()})
    add('real',real,rz,ry,primary)
    generated={}
    for v,m in models.items():
        z=gd_recover(m,noise.clone(),ry, cfg['sampler']['steps'],cfg['sampler']['stepsize'],'gd',.3)
        with torch.no_grad(): x=vae.decode(z/.18215).sample
        generated[v]=(x,z); add(f'generated_{v}',x,z,ry,primary)
    for sev in cfg['corruption_severities']:
        z=(1-sev)*rz+sev*noise
        with torch.no_grad(): x=vae.decode(z/.18215).sample
        add(f'real_corrupt_{sev:g}',x,z,ry,primary,sev)
    # Every sampler's own terminal candidates receive the same fixed corruption ladder.
    for v,(_,gz) in generated.items():
        for sev in cfg['corruption_severities']:
            z=(1-sev)*gz+sev*noise
            with torch.no_grad(): x=vae.decode(z/.18215).sample
            add(f'generated_{v}_corrupt_{sev:g}',x,z,ry,primary,sev)
    for k in range(competing.shape[1]): add(f'wrong_label_{k}',real,rz,competing[:,k],primary,0.,True)
    # Short trajectories are clear, model-specific generation failures, not relabeled reals.
    for v,m in models.items():
        z=gd_recover(m,noise.clone(),ry,3,cfg['sampler']['stepsize'],'gd',.3)
        with torch.no_grad(): x=vae.decode(z/.18215).sample
        add(f'failure_{v}',x,z,ry,primary,.95)
    with torch.no_grad(): noise_image=vae.decode(noise/.18215).sample
    add('noise',noise_image,noise,ry,primary,1.)
    # Independent metrics over saved candidate pixels.
    ims=torch.stack([r.pop('image') for r in rows]).to(d); zs=torch.stack([r.pop('latent') for r in rows]).to(d); ys=torch.tensor([r['label'] for r in rows],device=d)
    dino=torch.hub.load('facebookresearch/dinov2','dinov2_vits14').to(d).eval()
    with torch.no_grad():
        din=F.normalize(dino(F.interpolate((ims+1)/2,(224,224),mode='bilinear',align_corners=False)),dim=1)
        dref=F.normalize(dino(F.interpolate((ref+1)/2,(224,224),mode='bilinear',align_corners=False)),dim=1)
        realism=(din@dref.T).max(1).values.cpu().numpy()
        logits=cls(weights.transforms()(ims.add(1).div(2))); prob=logits.softmax(1)[torch.arange(len(ims),device=d),ys].cpu().numpy(); pred=logits.argmax(1).cpu().numpy()
    scores={}
    for name,m in [('direct_energy',models['direct']),('dot_energy',models['dot'])]:
        vals=[]; anchors=[]
        for s in range(0,len(zs),a.batch_size):
            size=min(a.batch_size,len(zs)-s); tt=torch.full((size,),cfg['t_eval'],device=d)
            vals.append(scalar(m,zs[s:s+size],tt,ys[s:s+size]).cpu())
            anchors.append(scalar(m,torch.zeros_like(zs[s:s+size]),tt,ys[s:s+size]).cpu())
        raw=torch.cat(vals).numpy(); scores[name]=raw; scores[name+'_zero_anchored']=raw-torch.cat(anchors).numpy()
    vals=[]
    for s in range(0,len(zs),a.batch_size): vals.append(fieldnorm(models['none'],zs[s:s+a.batch_size],torch.full((min(a.batch_size,len(zs)-s),),cfg['t_eval'],device=d),ys[s:s+a.batch_size]).cpu())
    scores['base_field_norm']=torch.cat(vals).numpy(); quality=(rank(realism)+rank(prob))/2
    for i,r in enumerate(rows): r.update(realism=float(realism[i]),class_probability=float(prob[i]),class_correct=bool(pred[i]==r['label']),composite_quality=float(quality[i]),**{k:float(v[i]) for k,v in scores.items()})
    with (out/'candidates.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader();w.writerows(rows)
    # Conditional pairs are exactly corresponding real/correct vs wrong-label rows.
    right=np.array([i for i,r in enumerate(rows) if r['group']=='real']); right_by_source={rows[i]['source_id']:i for i in right}
    wrongi=np.array([i for i,r in enumerate(rows) if r['group'].startswith('wrong_label_')])
    families=['real']+[f'generated_{v}' for v in models]
    corrupt=[]
    for family in families:
        # Include the clean terminal candidate as severity zero; omitting this
        # transition would test only the final two rungs of the corruption ladder.
        levels=[0.]+cfg['corruption_severities']
        for lo,hi in zip(levels[:-1],levels[1:]):
            low_group=family if lo==0 else f'{family}_corrupt_{lo:g}'
            low=np.array([i for i,r in enumerate(rows) if r['group']==low_group])
            high=np.array([i for i,r in enumerate(rows) if r['group']==f'{family}_corrupt_{hi:g}'])
            corrupt.append((low,high))
    sources=np.array([r['source_id'] for r in rows])
    metrics=[]
    for si,name in enumerate(SCORES):
        qcorr,qci=cluster_boot(lambda ix:spearman(scores[name][ix],quality[ix]),sources,cfg['bootstrap_replicates'],cfg['seed']+si)
        acc,aci=cluster_boot(lambda ix:pairacc(scores[name][ix],quality[ix]),sources,cfg['bootstrap_replicates'],cfg['seed']+10+si)
        cond=np.array([scores[name][right_by_source[rows[i]['source_id']]]<scores[name][i] for i in wrongi]); cm,cci=boot(lambda ix:cond[ix].mean(),len(cond),cfg['bootstrap_replicates'],cfg['seed']+20+si)
        mono=np.concatenate([(scores[name][b]>scores[name][a]) for a,b in corrupt])
        mono_sources=np.concatenate([sources[a] for a,_ in corrupt])
        mm,mci=cluster_boot(lambda ix:mono[ix].mean(),mono_sources,cfg['bootstrap_replicates'],cfg['seed']+30+si)
        metrics += [{'score':name,'metric':'spearman_quality_clustered','estimate':qcorr,'ci95':qci},{'score':name,'metric':'pair_accuracy_clustered','estimate':acc,'ci95':aci},{'score':name,'metric':'conditional_correct_lower','estimate':cm,'ci95':cci},{'score':name,'metric':'corruption_increases_all_families','estimate':mm,'ci95':mci}]
    (out/'metrics.json').write_text(json.dumps({'config':cfg,'metrics':metrics,'interpretation':'pilot; direct passes only if all direct endpoints are strong and its CIs exceed both baselines'},indent=2)+'\n')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    groups=sorted({r['group'] for r in rows})
    for name in SCORES:
        fig,ax=plt.subplots(figsize=(12,4)); ax.boxplot([[scores[name][i] for i,r in enumerate(rows) if r['group']==g] for g in groups],tick_labels=groups,showfliers=False);ax.tick_params(axis='x',rotation=40);ax.set_ylabel(name+' (lower hypothesized better)');fig.tight_layout();fig.savefig(out/f'{name}_groups.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots();ax.scatter(quality,scores[name],c=realism,cmap='viridis');ax.set(xlabel='independent composite quality',ylabel=name);fig.tight_layout();fig.savefig(out/f'{name}_quality.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        for family in families:
            level=[0]+cfg['corruption_severities']; vals=[]
            for sev in level:
                group=family if sev==0 else f'{family}_corrupt_{sev:g}'
                vals.append(np.mean([scores[name][i] for i,r in enumerate(rows) if r['group']==group]))
            ax.plot(level,vals,marker='o',label=family)
        ax.set(xlabel='fixed latent corruption severity',ylabel=name+' (lower hypothesized better)');ax.legend();fig.tight_layout();fig.savefig(out/f'{name}_corruption_levels.png',dpi=160);plt.close(fig)
    (out/'summary.md').write_text('# Fixed-candidate scalar-energy pilot\n\n`t_eval=1.0` is the repository terminal/clean endpoint (`z_t=(1-t)noise+t*data`). Quality is independent DINO nearest-reference similarity plus supplied-label ImageNet probability.\n\n|score|metric|estimate|95% CI|\n|---|---|---:|---|\n'+''.join(f"|{m['score']}|{m['metric']}|{m['estimate']:.3f}|[{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]|\n" for m in metrics))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--data-path',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--seed',type=int);a=p.parse_args();
 if a.seed is not None:
  raw=json.loads(a.config.read_text());raw['seed']=a.seed;a.config.parent.joinpath('.runtime_confirmation.json').write_text(json.dumps(raw));a.config=a.config.parent/'.runtime_confirmation.json'
 main(a)
