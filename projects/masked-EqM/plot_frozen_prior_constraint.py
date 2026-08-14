"""Simple, deterministic final-result figures for frozen projected EqM inference."""
import argparse, json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_frozen_prior_constraint import load_rows


ARMS = ('gaussian', 'bernoulli', 'mixed')
MODES = ('none', 'hard', 'soft')


def rows_at(root):
    """Admission policy is the analyzer's, verbatim.

    Previously this kept ``completion_status=='ok'`` and carried on while the
    strict analyzer hard-failed on the same directory -- so a figure could be
    published from a non-random surviving subset of exactly the data the
    analyzer refused.  Sharing ``load_rows`` makes the two admissions identical
    by construction (fail-closed on non-ok/non-finite records and on duplicate
    record ids), so the figure cannot outlive the analysis.
    """
    return load_rows(root)


def arm(r):
    """Arm identity comes from the first-class field, never from a name fallback.

    ``r.get('model_arm', <fallback>)`` returns None when the key is present but
    null, and a ``_seed``-split fallback returns a token that is in no plot's
    arm list -- either way the arm silently disappears from every figure with no
    error and no visible gap.
    """
    value = r.get('model_arm')
    if value is None:
        raise RuntimeError(f"record {r.get('record_id')!r} has no model_arm; refusing to guess the arm")
    if value not in ARMS:
        raise RuntimeError(f"record {r.get('record_id')!r} has unrecognized model_arm {value!r}; expected one of {ARMS}")
    return value


def grouped_mean(rows, metric, where=lambda r: True):
    buckets=defaultdict(list)
    for r in rows:
        if where(r): buckets[(arm(r),r['projection_mode'])].append(float(r[metric]))
    return {k:float(np.mean(v)) for k,v in buckets.items()}


def bar(data, title, ylabel, out):
    labels=[f'{a}\n{m}' for a in ARMS for m in MODES if (a,m) in data]
    values=[data[(a,m)] for a in ARMS for m in MODES if (a,m) in data]
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(range(len(values)),values,color='#4c78a8')
    ax.set_xticks(range(len(values)),labels); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis='y',alpha=.25)
    fig.tight_layout(); fig.savefig(out,dpi=180); plt.close(fig)


def main(args):
    rows=rows_at(args.input_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    combined=lambda r:r['mask_family']=='combined' and r['requested_visible_fraction']==.5
    bar(grouped_mean(rows,'lpips_missing_composite',combined),'Combined mask: missing-region LPIPS (lower is better)','LPIPS',out/'lpips_combined_by_arm_projection.png')
    bar(grouped_mean(rows,'missing_model_mse',combined),'Combined mask: missing model-space MSE (lower is better)','MSE',out/'mse_combined_by_arm_projection.png')
    # Visibility curves, faceted by inference mode.
    fig,ax=plt.subplots(figsize=(7,4))
    for a in ARMS:
        for m,style in [('hard','-o'),('none','--x')]:
            points=[]
            for f in (.35,.5,.65):
                vals=[r['lpips_missing_composite'] for r in rows if arm(r)==a and r['projection_mode']==m and r['mask_family']=='combined' and r['requested_visible_fraction']==f]
                if vals: points.append((f,np.mean(vals)))
            if points: ax.plot(*zip(*points),style,label=f'{a} {m}')
    ax.set(xlabel='Visible fraction',ylabel='Missing-region LPIPS (lower is better)',title='Combined-mask reconstruction versus visibility'); ax.grid(alpha=.25); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(out/'lpips_vs_visibility.png',dpi=180); plt.close(fig)
    # Family control comparison under hard projection at 50% visibility.
    fig,ax=plt.subplots(figsize=(8,4)); fams=('bernoulli','block','combined','irregular'); x=np.arange(len(fams)); width=.24
    for j,a in enumerate(ARMS):
        vals=[]
        for fam in fams:
            z=[r['lpips_missing_composite'] for r in rows if arm(r)==a and r['projection_mode']=='hard' and r['mask_family']==fam and r['requested_visible_fraction']==.5]
            vals.append(np.mean(z) if z else np.nan)
        ax.bar(x+(j-1)*width,vals,width,label=a)
    ax.set_xticks(x,fams); ax.set_ylabel('Missing-region LPIPS (lower is better)'); ax.set_title('Hard projection across mask families'); ax.legend(); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(out/'lpips_mask_family_controls.png',dpi=180); plt.close(fig)
    # Observation-consistency improvement, paired by record metadata excluding projection mode.
    fig,ax=plt.subplots(figsize=(6,4)); effects=[]; labels=[]
    for a in ARMS:
        key=lambda r:(r['checkpoint_id'],r['sample_id'],r['mask_family'],r['requested_visible_fraction'])
        none_rows=[r for r in rows if arm(r)==a and r['projection_mode']=='none']; hard_rows=[r for r in rows if arm(r)==a and r['projection_mode']=='hard']
        none={key(r):r for r in none_rows}; hard={key(r):r for r in hard_rows}
        # Collapsing to a dict silently keeps only the LAST record when re-run shards
        # overlap; the analyzer treats that as fatal and so must the figure.
        if len(none)!=len(none_rows) or len(hard)!=len(hard_rows): raise RuntimeError(f'duplicate pair keys for arm {a}')
        common=sorted(set(none)&set(hard)); diff=[none[k]['lpips_missing_composite']-hard[k]['lpips_missing_composite'] for k in common]
        if diff: effects.append(diff); labels.append(a)
    if effects: ax.boxplot(effects,labels=labels,showfliers=False); ax.axhline(0,color='k',lw=.8); ax.set_ylabel('LPIPS none - hard (positive favors hard)'); ax.set_title('Hard-projection improvement'); ax.grid(axis='y',alpha=.25)
    fig.tight_layout(); fig.savefig(out/'hard_projection_improvement.png',dpi=180); plt.close(fig)
    print(out)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',required=True); p.add_argument('--output-dir',required=True); main(p.parse_args())
