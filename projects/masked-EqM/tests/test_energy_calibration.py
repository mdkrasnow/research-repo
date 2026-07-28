from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from energy_monotonicity.evaluate_energy_calibration import (
    bootstrap, c_gamma, c_integral_to_clean, calibration_metrics, line_energy_clean_anchored,
    target_energy_curve,
)


def _case(scale=1.0, shape=None, orthogonal=0.0):
    gammas=np.linspace(0,1,21); clean=np.zeros((3,1,1,2)); eps=np.array([[[[2.,1.]]],[[[1.,-2.]]],[[[-3.,2.]]]])
    d=eps-clean; c=c_gamma(gammas) if shape is None else shape(gammas)
    field=scale*d[:,None]*c[None,:,None,None,None]
    if orthogonal:
        perp=np.stack([-d[...,1],d[...,0]],axis=-1).reshape(3,1,1,2)
        field=field+orthogonal*perp[:,None]*c[None,:,None,None,None]
    target=target_energy_curve(clean,eps,gammas); pred=line_energy_clean_anchored(field,clean,eps,gammas)
    return gammas,clean,eps,field,target,pred


def test_analytic_schedule_integral_and_clean_anchor():
    g=np.array([0,.3,.8,.9,1.]); expected=np.array([3.6,2.4,.4,.1,0.])
    assert np.allclose(c_integral_to_clean(g),expected)
    assert c_gamma(1.) == 0


def test_exact_target_half_and_double_scale():
    g,x,e,f,t,p=_case(); m=calibration_metrics(p,t,f,x,e,g)
    assert np.max(m['nece']) < 1e-12 and np.allclose(m['endpoint_ratio'],1)
    for scale,ratio in [(0.5,.5),(2.,2.)]:
        g,x,e,f,t,p=_case(scale); m=calibration_metrics(p,t,f,x,e,g)
        assert np.allclose(m['endpoint_ratio'],ratio) and np.all(m['nece'] > 0)


def test_wrong_shape_and_orthogonal_contamination_are_visible():
    # Same 21-point trapezoidal endpoint integral as the target (3.6), but wrong shape.
    g,x,e,f,t,p=_case(shape=lambda q:np.full_like(q,3.6)); m=calibration_metrics(p,t,f,x,e,g)
    assert np.allclose(m['endpoint_ratio'],1) and np.all(m['shape_error'] > .01)
    g,x,e,f,t,p=_case(orthogonal=.5); m=calibration_metrics(p,t,f,x,e,g)
    assert np.max(m['nece']) < 1e-12 and np.all(m['avg_transverse_fraction'] > 0)


def test_reversed_field_fails_and_clean_is_zero():
    g,x,e,f,t,p=_case(-1.); m=calibration_metrics(p,t,f,x,e,g)
    assert np.all(p[:, -1] == 0) and np.all(t[:, -1] == 0) and np.all(m['nece'] > 1)


def test_target_derivative_matches_repository_schedule():
    g,x,e,_,target,_=_case()
    derivative=np.gradient(target,g,axis=1)
    d2=((e-x).reshape(len(x),-1)**2).sum(1)
    # Away from the kink at .8 and the numerical endpoints the analytical derivative holds.
    interior=(g>.05)&(g<.75)
    assert np.allclose(derivative[:,interior],-d2[:,None]*c_gamma(g[interior]),rtol=.03)


def test_bootstrap_keeps_noises_together_and_is_deterministic():
    ids=np.repeat(np.arange(4),2)
    metrics={v:{'nece':np.arange(8,dtype=float)+offset} for v,offset in [('none',1),('dot',2),('direct',3)]}
    first, rows=bootstrap(metrics,ids,200,23456)
    second,_=bootstrap(metrics,ids,200,23456)
    assert np.array_equal(first['cluster_draws'],second['cluster_draws'])
    assert np.allclose(first['direct_minus_dot'],1)
    assert {r['comparison'] for r in rows} == {'direct-dot','direct-none relative excess','dot-none'}


def test_constant_curve_offsets_do_not_change_relative_metrics():
    g,x,e,f,t,p=_case()
    # Relative curves are explicitly clean anchored; a scalar's additive offset cancels.
    m0=calibration_metrics(p,t,f,x,e,g)
    m1=calibration_metrics((p+17)-(p[:,-1:]+17),t,f,x,e,g)
    assert np.allclose(m0['nece'],m1['nece'])
    assert np.allclose(m0['endpoint_ratio'],m1['endpoint_ratio'])
