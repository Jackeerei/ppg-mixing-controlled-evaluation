"""v0.1 fixed, label-free window NLMS. Production FP32 throughout NumPy ops."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class FrontendConfig:
    length:int=8
    mu:float=0.05
    eps_std_ppg:float=1e-6
    eps_std_acc:float=1e-6
    eps_nlms:float=1e-6
    eps_feature:float=1e-6

def _finite(*arrays):
    if not all(np.isfinite(a).all() for a in arrays):
        raise FloatingPointError('Non-finite frontend intermediate; common window error')

def process_windows(windows,config=FrontendConfig()):
    # Finite raw values can overflow FP32 statistics; fail before clipping.
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        return _process_windows(windows,config)

def _process_windows(windows,config):
    """Input [B,4,256] PPG/ACC only. Returns d [B,256], f [B,3], diagnostics.

    Statistics, coefficients, references, residual and features are FP32. No
    state is retained between calls/windows. Inputs are never modified. No HR,
    activity or subject identity is accepted. Batch windows remain independent.
    """
    x=np.asarray(windows,dtype=np.float32)
    if x.ndim!=3 or x.shape[1:]!=(4,256) or not len(x):raise ValueError('Expected nonempty [B,4,256]')
    if not np.isfinite(x).all():raise ValueError('Non-finite input is a common error, never silently repaired')
    if config.length!=8 or not 0<=config.mu<=1 or min(config.eps_std_ppg,config.eps_std_acc,config.eps_nlms,config.eps_feature)<=0:raise ValueError('Invalid fixed frontend configuration')
    if not all(np.isfinite(v) for v in asdict(config).values()):raise ValueError('Non-finite configuration')
    mean=x.mean(axis=2,dtype=np.float32,keepdims=True)
    centered=x-mean
    std=np.sqrt(np.mean(centered*centered,axis=2,dtype=np.float32,keepdims=True))
    _finite(mean,centered,std)
    low_ppg=std[:,0,0]<=np.float32(config.eps_std_ppg)
    low_acc=std[:,1:,0]<=np.float32(config.eps_std_acc)
    s=np.maximum(std,np.array([config.eps_std_ppg]+[config.eps_std_acc]*3,dtype=np.float32)[None,:,None])
    standardized=centered/s
    _finite(s,standardized)
    u=standardized[:,0,:]
    acc=standardized[:,1:,:].copy()
    acc[low_acc]=0
    weights=np.zeros((len(x),3*config.length),dtype=np.float32)
    history=np.zeros((len(x),3,config.length),dtype=np.float32)
    e=np.empty_like(u)
    for n in range(256):
        history[:,:,1:]=history[:,:,:-1].copy()
        history[:,:,0]=acc[:,:,n]
        rho=history.reshape(len(x),-1)
        predicted=np.sum(weights*rho,axis=1,dtype=np.float32)
        residual=u[:,n]-predicted
        e[:,n]=residual
        gain=np.float32(config.mu)*residual/(np.float32(config.eps_nlms)+np.sum(rho*rho,axis=1,dtype=np.float32))
        weights+=gain[:,None]*rho
        _finite(predicted,residual,gain,weights)
    d=mean[:,0,:]+s[:,0,:]*e
    d[low_ppg]=x[low_ppg,0,:]
    # Exact identity paths avoid forward/inverse floating-point round-trip loss.
    identity=low_ppg|np.all(low_acc,axis=1)|(config.mu==0)
    d[identity]=x[identity,0,:]
    m=u-e
    q_raw=np.mean(m*m,axis=1,dtype=np.float32)/(np.mean(u*u,axis=1,dtype=np.float32)+np.float32(config.eps_feature))
    l1_raw=np.mean(np.abs(m),axis=1,dtype=np.float32)/(np.mean(np.abs(u),axis=1,dtype=np.float32)+np.float32(config.eps_feature))
    _finite(d,e,m,q_raw,l1_raw)
    # Pearson is invariant to the positive normalization factors on valid axes.
    # Use their standardized vectors and never multiply large raw energies.
    ppg_energy=np.sum(u*u,axis=1,dtype=np.float32)
    acc_energy=np.sum(acc*acc,axis=2,dtype=np.float32)
    numerator=np.abs(np.sum(u[:,None,:]*acc,axis=2,dtype=np.float32))
    denominator=np.sqrt(ppg_energy[:,None])*np.sqrt(acc_energy)
    _finite(ppg_energy,acc_energy,numerator,denominator)
    correlations=np.zeros_like(numerator)
    valid=(~low_acc)&(~low_ppg[:,None])
    if np.any(denominator[valid]<=0):raise FloatingPointError('Non-positive Pearson denominator on a valid axis')
    np.divide(numerator,denominator,out=correlations,where=valid)
    c=np.clip(correlations.max(axis=1),0,1)
    features=np.stack([c,np.clip(q_raw,0,1),np.clip(l1_raw,0,1)],axis=1)
    features[low_ppg]=0;q_raw[low_ppg]=0;l1_raw[low_ppg]=0
    if not np.isfinite(d).all() or not np.isfinite(features).all():raise FloatingPointError('Non-finite frontend output')
    return {'filtered_ppg':d,'features':features,'q_raw':q_raw,'f_l1_raw':l1_raw,
            'low_variance_ppg':low_ppg,'low_variance_acc':low_acc,'config':asdict(config)}
