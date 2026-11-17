"""Necessary synthetic checks only; no real training, backbone, or outer loading."""
from pathlib import Path
import json,math,shutil,subprocess
import numpy as np
import torch,psutil
import engine
HERE=Path(__file__).resolve().parent
def main():
    errors=[-1000.,-8.,-5.,-1.,0.,1.,5.,8.,1000.];checks={}
    for name in ['logcosh','MSE','MAE','Huber5']:
        x=torch.tensor(errors,dtype=torch.float64,requires_grad=True);loss=engine.objective(x,name);loss.backward()
        if name=='logcosh':expected=[abs(e)+math.log1p(math.exp(-2*abs(e)))-math.log(2) for e in errors];grad=[math.tanh(e) for e in errors]
        elif name=='MSE':expected=[e*e for e in errors];grad=[2*e for e in errors]
        elif name=='MAE':expected=[abs(e) for e in errors];grad=[(1 if e>0 else -1 if e<0 else 0) for e in errors]
        else:expected=[.5*e*e if abs(e)<=5 else 5*(abs(e)-2.5) for e in errors];grad=[max(-5,min(5,e)) for e in errors]
        assert abs(loss.item()-np.mean(expected))<1e-10 and np.allclose(x.grad.numpy(),np.array(grad)/len(errors),atol=1e-12,rtol=1e-12)
        checks[name]={'value_and_analytic_gradient':'passed','extreme_error_finite':True,'delta':5 if name=='Huber5' else None}
    # Imbalanced inner subjects prove equal-person scoring, rather than pooled choice.
    s=engine.score_predictions(np.array([1.]*100+[9.]),np.zeros(101),np.array([1]*100+[2]),[1,2]);assert s['subject_macro_mae']==5 and s['pooled_mae']<1.1
    cfg=json.loads((HERE/'config.json').read_text());ids=cfg['runs'];assert len(ids)==len({(r['fold'],r['seed'],r['method'],r['objective']) for r in ids})==90
    assert sum(r['objective']=='logcosh' for r in ids)==36
    for f in cfg['folds']:
        sets=[set(f[k]) for k in ['train','inner','outer']];assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    # Tie never counts as an improvement. The production train source uses this exact metric/rule.
    text=(HERE/'engine.py').read_text();assert "if scores['subject_macro_mae']<best:" in text and "best=scores['subject_macro_mae']" in text
    assert engine.batch_indices(np.arange(129),128)[0].size==127 and engine.batch_indices(np.arange(129),128)[1].size==2
    rows={'status':'passed','scope':'synthetic objective/gradient, inner macro weighting, fold/90 identity/singleton and strict selection checks; lifecycle source review still pending','objectives':checks,'synthetic_inner_macro_mae':s['subject_macro_mae'],'real_training_started':False,'outer_data_loaded':False,'available_ram_bytes_after_torch':psutil.virtual_memory().available,'free_disk_bytes':shutil.disk_usage(HERE).free,'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:rows['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=20).stdout.strip()
    except Exception as exc:rows['nvidia_smi_error']=repr(exc)
    with (HERE/'checks.json').open('x') as f:json.dump(rows,f,indent=2)
    print(json.dumps(rows))
if __name__=='__main__':main()
