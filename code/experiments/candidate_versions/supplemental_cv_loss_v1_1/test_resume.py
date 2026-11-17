"""Isolated actual-backbone/Adam continuity test; no real dataset or outer scoring."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import copy,datetime,json,random,sys
import numpy as np,torch
from resumable import Trainer,core,rng_state
from checkpoints import Store,atomic_json
ROOT=Path(__file__).resolve().parents[2]
def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b);return
    if isinstance(a,np.ndarray):assert np.array_equal(a,b);return
    if isinstance(a,dict):assert a.keys()==b.keys();[equal(a[k],b[k]) for k in a];return
    if isinstance(a,(list,tuple)):assert len(a)==len(b);[equal(x,y) for x,y in zip(a,b)];return
    assert a==b,(a,b)
def data(n,subjects,seed):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,4,256)).astype(np.float32)
    return {'x':x,'d':(x[:,0]*.8).copy(),'f':rng.normal(size=(n,3)).astype(np.float32),'y':rng.uniform(60,90,n).astype(np.float32),'subject':np.resize(np.array(subjects),n),'activity':np.zeros(n,int),'start':np.arange(n)*64,'required_subjects':subjects}
def build():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    return core.Variant(core.build_model(),'D')
def test_device(stage,device):
    train=data(17,[501],1);inner=data(5,[601,602],2)
    options={'batch_size':4,'objective':'logcosh','max_epochs':3,'patience':20,'active_budget_seconds':7200,'optimizer':{'backbone_lr':.001,'gate_lr':.001,'backbone_weight_decay':0.,'gate_weight_decay':0.,'amsgrad':True}}
    identity={'seed':42,'run_id':'SYNTHETIC_ONLY_'+device,'fold':'SYNTHETIC','loss':'logcosh','source_version':'resume_test_pending_freeze'}
    model=build().to(device);opt,_=core.optimizer_for(model,options);a=Trainer(stage/'continuous',model,opt,train,inner,options,identity,device,checkpoint_updates=2)
    assert a.advance()=='complete';state_a=copy.deepcopy(a.state);rng_a=rng_state();opt_a=copy.deepcopy(opt.state_dict());weights_a=copy.deepcopy(model.state_dict())
    model=build().to(device);opt,_=core.optimizer_for(model,options);b=Trainer(stage/'interrupted',model,opt,train,inner,options,identity,device,checkpoint_updates=2)
    assert b.advance(stop_after_updates=7)=='paused_for_test' and b.state['next_batch']==2
    # Simulate process restart disturbing all global RNGs and constructors first.
    random.random();np.random.random(10);torch.rand(10)
    model=build().to(device);opt,_=core.optimizer_for(model,options);c=Trainer(stage/'interrupted',model,opt,train,inner,options,identity,device,checkpoint_updates=2)
    assert c.state['updates']==7 and c.advance()=='complete'
    equal(weights_a,model.state_dict());equal(opt_a,opt.state_dict());equal(rng_a,rng_state())
    for key in ['phase','epoch','next_batch','updates','loss_sum','seen','best_inner_mae','best_update','stale','history','batch_order_hashes','best_model']:equal(state_a[key],c.state[key])
    # Actual on-disk latest damage must choose a complete previous generation.
    index=json.loads(c.store.index.read_text());latest=c.store.path/index['generations'][0]['name'];latest.write_bytes(b'INTENTIONALLY_CORRUPTED_SYNTHETIC_CHECKPOINT')
    state,fallback=c.store.load();assert fallback['fallback_to_previous'] and state['phase']=='complete'
    # Same trusted identity required, not merely same dimensions.
    try:Store(c.store.path,{**c.identity,'run_id':'WRONG'}).load();raise AssertionError('mismatch accepted')
    except AssertionError as exc:assert str(exc)=='Checkpoint identity mismatch'
    return {'device':device,'continuous_vs_non_epoch_restart':'exact','weights_BN_optimizer_AMSGrad_RNG_order_earlystop_metric':'exact','updates':c.state['updates'],'cut_at_update':7,'latest_corruption_fallback':'passed','identity_mismatch_rejected':True,'active_seconds_non_decreasing':c.state['active_seconds']>=b.state['active_seconds']}
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    stage=ROOT/'reports'/('resume_synthetic_v1_1_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));stage.mkdir(exist_ok=False);results=[]
    for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
        folder=stage/device.replace(':','_');folder.mkdir();results.append(test_device(folder,device))
    atomic_json(stage/'CHECKS.json',{'status':'passed','scope':'synthetic actual backbone/gate/Adam only, no CV data or outer results','results':results});print(json.dumps(results))
if __name__=='__main__':main()
