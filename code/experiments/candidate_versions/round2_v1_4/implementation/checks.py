"""Synthetic implementation checks; no real data, HR score or optimizer step."""
from pathlib import Path
from dataclasses import replace
import sys,json,tempfile,hashlib
import numpy as np
import torch
from torch import nn
from frontend import process_windows,FrontendConfig
from gating import Variant
from cache import get_or_create

VERSION=Path(__file__).resolve().parents[1]
ROOT=VERSION.parents[1]
sys.path.insert(0,str(ROOT))
from models.qppg_fixed import build_model

def oracle_one(x,cfg):
    """Independent scalar FP64 reference: explicit history loops and dot products."""
    x=x.astype(np.float64);mx=x.mean(axis=1);std=x.std(axis=1,ddof=0)
    if std[0]<=cfg.eps_std_ppg:return x[0].copy(),np.zeros(3)
    sx=max(std[0],cfg.eps_std_ppg);u=(x[0]-mx[0])/sx
    refs=[];correlations=[]
    for j in range(1,4):
        if std[j]<=cfg.eps_std_acc:refs.append(np.zeros(256));correlations.append(0.)
        else:
            refs.append((x[j]-mx[j])/max(std[j],cfg.eps_std_acc))
            correlations.append(abs(np.dot(x[0]-mx[0],x[j]-mx[j]))/np.sqrt(np.dot(x[0]-mx[0],x[0]-mx[0])*np.dot(x[j]-mx[j],x[j]-mx[j])))
    w=np.zeros(24);e=[]
    for n in range(256):
        rho=np.array([refs[axis][n-lag] if n>=lag else 0. for axis in range(3) for lag in range(8)])
        error=u[n]-np.dot(w,rho);e.append(error)
        w=w+cfg.mu*error/(cfg.eps_nlms+np.dot(rho,rho))*rho
    e=np.array(e);m=u-e
    return mx[0]+sx*e,np.array([min(1.,max(correlations)),np.clip(np.mean(m*m)/(np.mean(u*u)+cfg.eps_feature),0,1),np.clip(np.mean(np.abs(m))/(np.mean(np.abs(u))+cfg.eps_feature),0,1)])

def run():
    torch.set_num_threads(2);rng=np.random.default_rng(198);x=rng.normal(size=(12,4,256)).astype(np.float32)
    x[:,0,:]=20+2*x[:,0,:]+0.7*x[:,1,:]
    original=x.copy();cfg=FrontendConfig();r=process_windows(x);tests=[]
    np.testing.assert_array_equal(x,original)
    np.testing.assert_array_equal(r['filtered_ppg'],process_windows(x)['filtered_ppg'])
    for i in range(len(x)):
        np.testing.assert_array_equal(r['filtered_ppg'][i],process_windows(x[i:i+1])['filtered_ppg'][0])
    tests+=['input not mutated','repeat and batch/single window reset equality']
    do=[];fo=[]
    for window in x:
        d,f=oracle_one(window,cfg);do.append(d);fo.append(f)
    do=np.array(do);fo=np.array(fo)
    np.testing.assert_allclose(r['filtered_ppg'],do,atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(r['features'],fo,atol=2e-5,rtol=2e-5)
    tests.append('FP32 production agrees with independent scalar FP64 oracle, including axis/lag order and pre-update residual')
    np.testing.assert_array_equal(process_windows(x,replace(cfg,mu=0))['filtered_ppg'],x[:,0,:])
    z=x.copy();z[:,1:,:]=0
    np.testing.assert_array_equal(process_windows(z)['filtered_ppg'],z[:,0,:])
    low=x.copy();low[:,0,:]=1e-8*np.sin(np.arange(256))
    lo=process_windows(low);np.testing.assert_array_equal(lo['filtered_ppg'],low[:,0,:]);assert not lo['features'].any()
    constant=x.copy();constant[:,0,:]=20
    np.testing.assert_array_equal(process_windows(constant)['filtered_ppg'],constant[:,0,:])
    axis=x.copy();axis[:,1,:]=0.1e-6*np.sin(np.arange(256));axis2=axis.copy();axis2[:,1,:]=0
    np.testing.assert_array_equal(process_windows(axis)['filtered_ppg'],process_windows(axis2)['filtered_ppg'])
    tests+=['mu0 and zero ACC exact identity','constant and low-variance PPG bypass','low-variance ACC axis equivalent to zero standardized reference']
    for value in (np.nan,np.inf):
        bad=x.copy();bad[0,0,0]=value
        try:process_windows(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    tests.append('non-finite common input rejected')
    shared=torch.load(VERSION/'initial_backbone_seed42.pt',weights_only=True,map_location='cpu')
    xt=torch.from_numpy(x);dt=torch.from_numpy(r['filtered_ppg']).requires_grad_();ft=torch.from_numpy(r['features']).requires_grad_()
    params={};mixed={}
    for method in ['A','B','C','G','D']:
        backbone=build_model();backbone.load_state_dict(shared);model=Variant(backbone,method).eval()
        assert all(torch.equal(v,shared[k]) for k,v in model.backbone.state_dict().items())
        m,a=model.mix(xt,dt,ft);mixed[method]=m.detach();params[method]=sum(p.numel() for p in model.parameters())
        assert torch.equal(m[:,1:,:],xt[:,1:,:])
        if method in ('G','D'):
            pred,_=model(xt,dt,ft);pred.square().mean().backward()
            grads=[p.grad for name,p in model.named_parameters() if not name.startswith('backbone.')]
            assert all(g is not None and torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    assert torch.equal(mixed['A'],xt) and torch.equal(mixed['B'][:,0,:],dt.detach())
    assert torch.equal(mixed['C'],mixed['D']) and torch.equal(mixed['C'],mixed['G'])
    assert dt.grad is None and ft.grad is None
    tests+=['five arms load exact shared backbone state','A/B endpoints and unchanged ACC','G/D initial mix equals C','actual backbone nonzero finite G/D gradients; fixed d/features detached']
    gate=Variant(build_model(),'D');mix,alpha=gate.mix(xt,xt[:,0,:].clone(),ft)
    mix.sum().backward();assert all(p.grad is not None and not p.grad.any() for p in gate.gate.parameters())
    tests.append('d=x yields legitimate zero gate gradients')
    with tempfile.TemporaryDirectory(prefix='candidate_cache_check_') as td:
        paths=[get_or_create(td,x,'synthetic-proposal-check') for method in ('B','C','G','D')]
        assert len(set(paths))==1
        with np.load(paths[0]) as data:np.testing.assert_array_equal(data['filtered_ppg'],r['filtered_ppg'])
        different=x.copy();different[0,0,0]+=1
        assert get_or_create(td,different,'synthetic-proposal-check')!=paths[0]
    tests.append('B/C/G/D reuse identical hash-bound immutable cache; changed input gets new artifact')
    report={'scope':'synthetic implementation tests only; no optimizer updates, real subject input or candidate performance',
            'status':'passed','checks':tests,'parameter_counts':params,'oracle_filtered_max_abs_error':float(np.max(np.abs(r['filtered_ppg']-do))),
            'oracle_feature_max_abs_error':float(np.max(np.abs(r['features']-fo))),'production_dtype':'FP32','reference_dtype':'FP64',
            'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    out=VERSION/'implementation_checks.json'
    with out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()
