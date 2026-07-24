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

def paired(rows,a,b,metric='lpips_missing_composite',nboot=10000,seed=20260724):
    key=lambda r:(r['checkpoint_id'].rsplit('_',1)[-1],r['dataset_index'],r['mask_family'],r['requested_visible_fraction'])
    ra={key(r):r for r in rows if r['checkpoint_id'].startswith(a) and r['projection_mode']=='hard' and r['mask_family']=='combined' and r['requested_visible_fraction']==.5}
    rb={key(r):r for r in rows if r['checkpoint_id'].startswith(b) and r['projection_mode']=='hard' and r['mask_family']=='combined' and r['requested_visible_fraction']==.5}
    keys=sorted(set(ra)&set(rb));
    if len(keys)<2: raise RuntimeError(f'incomplete pairing {a} {b}')
    dif=np.array([ra[k][metric]-rb[k][metric] for k in keys]); rng=np.random.default_rng(seed)
    reps=np.array([rng.choice(dif,len(dif),replace=True).mean() for _ in range(nboot)])
    return {'n':len(dif),'mean_difference':float(dif.mean()),'ci95':np.percentile(reps,[2.5,97.5]).tolist(),'p_bootstrap':float(2*min((reps<=0).mean(),(reps>=0).mean())),'fraction_b_improves':float((dif>0).mean())}

def main(a):
    rows=load_rows(a.input_dir)
    summaries={}
    for arm in ['gaussian','bernoulli','mixed']:
        for mode in ['none','hard','soft']:
            x=[r['lpips_missing_composite'] for r in rows if r['checkpoint_id'].startswith(arm) and r['projection_mode']==mode]
            if x: summaries[f'{arm}_{mode}']={'n':len(x),'mean':float(np.mean(x)),'median':float(np.median(x)),'std':float(np.std(x,ddof=1)),'se':float(np.std(x,ddof=1)/math.sqrt(len(x)))}
    out={'rows':len(rows),'summaries':summaries,'primary_mixed_vs_gaussian':paired(rows,'gaussian','mixed'),'primary_mixed_vs_bernoulli':paired(rows,'bernoulli','mixed')}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--output',required=True);main(p.parse_args())
