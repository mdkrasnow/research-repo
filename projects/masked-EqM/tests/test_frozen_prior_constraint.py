import torch
from frozen_prior_constraint import project, visibility_mask, make_manifest, sampler_recover, shard_rows

def test_mask_polarity_and_combined_fraction():
    v, _ = visibility_mask('combined', 128, 128, .5, 11)
    assert .47 < float(v.mean()) < .53
    clean=torch.ones(1,4,2,2); fill=torch.zeros_like(clean); mask=torch.tensor([[[[1.,0.],[1.,0.]]]])
    assert torch.equal(mask*clean+(1-mask)*fill, torch.tensor([[[[1.,0.],[1.,0.]]]]).expand_as(clean))

def test_projection_contracts_observed_error():
    c=torch.ones(1,1,2,2); p=torch.zeros_like(c); v=torch.tensor([[[[1.,0.],[1.,0.]]]])
    assert torch.equal(project(p,c,v,'none'),p)
    assert torch.equal(project(p,c,v,'hard'),v*c)
    assert torch.equal(project(p,c,v,'soft',.5),.5*v)

def test_hard_sampler_keeps_projected_state_and_sign():
    def field(x,t,y): return torch.ones_like(x)
    c=torch.zeros(1,1,2,2); v=torch.tensor([[[[1.,0.],[1.,0.]]]])
    out, errors=sampler_recover(field,torch.zeros_like(c),c,v,torch.zeros(1,dtype=torch.long),3,.1,'gd',.3,'hard',1.)
    assert torch.allclose(out, torch.tensor([[[[0.,.2],[0.,.2]]]])) and max(errors)==0

def test_nag_uses_projected_state_between_steps():
    calls=[]
    def field(x,t,y): calls.append(x.clone()); return torch.ones_like(x)
    clean=torch.zeros(1,1,1,2); v=torch.tensor([[[[1.,0.]]]])
    sampler_recover(field, torch.zeros_like(clean), clean, v, torch.zeros(1,dtype=torch.long), 3, .1, 'ngd', .3, 'hard', 1.)
    # second look-ahead starts from projected first state [0,.1], not [0.1,.1]
    assert torch.allclose(calls[1], torch.tensor([[[[.03,.13]]]]))

def test_manifest_is_deterministic_and_shardable(tmp_path):
    a=make_manifest(tmp_path/'a.json',100,10,'pilot',['combined'],[.5],seed=9)
    b=make_manifest(tmp_path/'b.json',100,10,'pilot',['combined'],[.5],seed=9)
    assert a['manifest_hash']==b['manifest_hash'] and len({x['sample_id'] for x in a['rows']})==10
    shards=[shard_rows(a['rows'], i, 3) for i in range(3)]
    assert sum(map(len,shards))==len(a['rows']) and len({r['mask_hash'] for s in shards for r in s})==len(a['rows'])

def test_pilot_final_offsets_are_disjoint(tmp_path):
    pilot=make_manifest(tmp_path/'p.json',100,10,'pilot',['combined'],[.5],seed=8,index_offset=0)
    final=make_manifest(tmp_path/'f.json',100,20,'final',['combined'],[.5],seed=8,index_offset=10)
    assert {r['dataset_index'] for r in pilot['rows']}.isdisjoint({r['dataset_index'] for r in final['rows']})
