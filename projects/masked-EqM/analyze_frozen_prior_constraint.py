"""Fail-closed paired aggregation for frozen-prior constraint JSONL shards."""
import argparse, json, math
from pathlib import Path
import numpy as np

def load_rows(root):
    rows=[]
    for p in Path(root).rglob('*.jsonl'):
        rows += [json.loads(x) for x in p.read_text().splitlines() if x]
    if not rows: raise RuntimeError('no result rows')
    bad=[r for r in rows if r.get('completion_status')!='ok' or not math.isfinite(r.get('lpips_missing_composite',float('nan')))]
    if bad: raise RuntimeError(f'{len(bad)} failed/nonfinite records')
    ids=[r['record_id'] for r in rows]
    if len(ids)!=len(set(ids)): raise RuntimeError('duplicate records')
    return rows

def paired(rows,a,b,metric='lpips_missing_composite',family='combined',mode_a='hard',mode_b='hard',nboot=10000,seed=20260724,require_seeds=None):
    key=lambda r:(r['checkpoint_id'].rsplit('_',1)[-1],r['dataset_index'],r['mask_family'],r['requested_visible_fraction'])
    filt=lambda r,arm,mode:r['checkpoint_id'].startswith(arm) and r['projection_mode']==mode and r['mask_family']==family and r['requested_visible_fraction']==.5
    aa=[r for r in rows if filt(r,a,mode_a)]; bb=[r for r in rows if filt(r,b,mode_b)]
    ra={key(r):r for r in aa}; rb={key(r):r for r in bb}
    if len(ra)!=len(aa) or len(rb)!=len(bb): raise RuntimeError('duplicate pair keys')
    keys=sorted(set(ra)&set(rb))
    if set(ra)!=set(rb) or len(keys)<3: raise RuntimeError(f'incomplete/mismatched pairing {a} {b} {family}')
    for k in keys:
        for fld in ('manifest_hash','mask_hash','corruption_seed','initialization_seed','sampler','sampler_steps','step_size','momentum'):
            if ra[k].get(fld)!=rb[k].get(fld): raise RuntimeError(f'paired mismatch {fld}: {k}')
    byseed={s:np.array([ra[k][metric]-rb[k][metric] for k in keys if k[0]==s]) for s in sorted({k[0] for k in keys})}
    if len(set(map(len,byseed.values())))!=1: raise RuntimeError('unequal paired image counts across checkpoint seeds')
    if require_seeds is not None and len(byseed)!=require_seeds: raise RuntimeError(f'requires exactly {require_seeds} complete matched checkpoint seeds')
    if not byseed: raise RuntimeError('no matched checkpoint seed')
    rng=np.random.default_rng(seed); seeds=list(byseed)
    reps=[]
    for _ in range(nboot):
        chosen=rng.choice(seeds,len(seeds),replace=True); reps.append(np.mean([rng.choice(byseed[s],len(byseed[s]),replace=True).mean() for s in chosen]))
    dif=np.concatenate(list(byseed.values())); reps=np.array(reps)
    null=[]
    for _ in range(nboot):
        chosen=rng.choice(seeds,len(seeds),replace=True); null.append(np.mean([rng.choice(byseed[s],len(byseed[s]),replace=True).mean()*rng.choice([-1,1]) for s in chosen]))
    return {'n':len(dif),'checkpoint_seed_count':len(byseed),'seed_effects':{s:float(x.mean()) for s,x in byseed.items()},'mean_difference':float(dif.mean()),'ci95':np.percentile(reps,[2.5,97.5]).tolist(),'p_hierarchical_sign_permutation':float((np.abs(null)>=abs(dif.mean())).mean()),'bootstrap_scope':'hierarchical checkpoint-seed plus image' if len(byseed)>1 else 'paired image only; no checkpoint-seed robustness inference','fraction_b_improves':float((dif>0).mean())}

def main(a):
    rows=load_rows(a.input_dir)
    manifest=json.loads(Path(a.manifest).read_text())
    expected={r['sample_id'] for r in manifest['rows'] if r['split']=='final'}
    if {r.get('manifest_hash') for r in rows}!={manifest['manifest_hash']}: raise RuntimeError('manifest hash mismatch')
    if any(r['sample_id'] not in expected for r in rows): raise RuntimeError('rows outside locked final manifest')
    summaries={}
    for arm in ['gaussian','bernoulli','mixed']:
        for mode in ['none','hard','soft']:
            x=[r['lpips_missing_composite'] for r in rows if r['checkpoint_id'].startswith(arm) and r['projection_mode']==mode]
            if x: summaries[f'{arm}_{mode}']={'n':len(x),'mean':float(np.mean(x)),'median':float(np.median(x)),'std':float(np.std(x,ddof=1)),'se':float(np.std(x,ddof=1)/math.sqrt(len(x)))}
    # The verified 1:1 mixed arm has seed 0 only.  Its comparisons are paired
    # at that seed and explicitly cannot support a checkpoint-seed claim.
    cg=paired(rows,'gaussian','mixed'); ig=paired(rows,'gaussian','mixed',family='irregular'); cb=paired(rows,'bernoulli','mixed'); ib=paired(rows,'bernoulli','mixed',family='irregular')
    out={'rows':len(rows),'summaries':summaries,'primary_mixed_vs_gaussian':cg,'primary_mixed_vs_bernoulli':cb,'projection_gaussian':paired(rows,'gaussian','gaussian',mode_a='none',mode_b='hard',require_seeds=3),'projection_bernoulli':paired(rows,'bernoulli','bernoulli',mode_a='none',mode_b='hard',require_seeds=3),'projection_mixed':paired(rows,'mixed','mixed',mode_a='none',mode_b='hard',require_seeds=1),'interaction_mixed_vs_gaussian_combined_minus_irregular':cg['mean_difference']-ig['mean_difference'],'interaction_mixed_vs_bernoulli_combined_minus_irregular':cb['mean_difference']-ib['mean_difference'],'limitations':['The primary 1:1 mixed checkpoint has only verified seed 0. Mixed-versus-specialist statistics are paired-image evidence at seed 0, not three-seed robustness evidence.']}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--manifest',required=True);main(p.parse_args())
