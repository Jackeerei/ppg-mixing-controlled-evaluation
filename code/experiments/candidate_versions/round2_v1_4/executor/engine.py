"""Five-arm engine. No resume: every arm directory is exclusive and immutable."""
from pathlib import Path
import csv,hashlib,json,math,random,sys,time,traceback
import numpy as np
import torch
from torch import nn

VERSION=Path(__file__).resolve().parents[1]
ROOT=VERSION.parents[1]
sys.path.insert(0,str(VERSION/'implementation'))
sys.path.insert(0,str(ROOT))
from gating import Variant
from models.qppg_fixed import build_model

def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def epoch_order(n,seed,epoch):
    return torch.randperm(n,generator=torch.Generator().manual_seed(seed+1000003*epoch)).numpy()

def optimizer_for(model,cfg):
    options=cfg['optimizer'];groups=[];description=[]
    for label,pairs in [('backbone',list(model.backbone.named_parameters())),
                        ('gate',[(n,p) for n,p in model.named_parameters() if not n.startswith('backbone.')])]:
        if not pairs:continue
        groups.append({'params':[p for _,p in pairs],'lr':options[label+'_lr'],'weight_decay':options[label+'_weight_decay']})
        description.append({'group':label,'parameter_names':[n for n,_ in pairs],'parameter_count':sum(p.numel() for _,p in pairs),
                            'lr':options[label+'_lr'],'weight_decay':options[label+'_weight_decay']})
    ids=[id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
    return torch.optim.Adam(groups,amsgrad=options['amsgrad']),description

def validate_data(data):
    n=len(data['y'])
    assert data['y'].shape==(n,) and n and data['x'].shape==(n,4,256) and data['d'].shape==(n,256) and data['f'].shape==(n,3)
    assert all(data[k].dtype==np.float32 for k in ('x','d','f','y'))
    assert all(len(data[k])==n for k in ('subject','activity','start'))
    assert all(np.isfinite(data[k]).all() for k in ('x','d','f','y'))
    assert np.all(data['start']>=0) and np.all(data['start']%64==0)
    keys=list(zip(data['subject'].tolist(),data['start'].tolist()))
    assert len(set(keys))==n

def batch_indices(order,batch):
    parts=[order[k:k+batch] for k in range(0,len(order),batch)]
    if len(parts)>1 and len(parts[-1])==1:
        joined=np.concatenate(parts[-2:]);parts[-2:]=[joined[:batch-1],joined[batch-1:]]
    assert sum(map(len,parts))==len(order) and min(map(len,parts))>=2
    return parts

def score_predictions(pred,y,subjects,required):
    assert set(map(int,np.unique(subjects)))==set(required)
    assert len(pred)==len(y) and np.isfinite(pred).all() and np.isfinite(y).all()
    error=np.asarray(pred,np.float64)-np.asarray(y,np.float64)
    loss=np.logaddexp(error,-error)-math.log(2)
    per={str(s):{'n':int((subjects==s).sum()),'logcosh':float(loss[subjects==s].mean()),'mae':float(np.abs(error[subjects==s]).mean())} for s in required}
    return {'validation_macro_logcosh':float(np.mean([v['logcosh'] for v in per.values()])),
            'validation_pooled_logcosh':float(loss.mean()),'pooled_mae':float(np.abs(error).mean()),
            'subject_macro_mae':float(np.mean([v['mae'] for v in per.values()])), 'per_subject':per}

def evaluate(model,data,device,batch):
    model.eval();pred=[];alpha=[]
    with torch.inference_mode():
        for start in range(0,len(data['y']),batch):
            args=[torch.from_numpy(data[k][start:start+batch]).to(device) for k in ('x','d','f')]
            p,a=model(*args);pred.extend(p.cpu().numpy().reshape(-1));alpha.extend(a.cpu().numpy().reshape(-1))
    pred=np.asarray(pred,dtype=np.float64);alpha=np.asarray(alpha,dtype=np.float64)
    assert pred.shape==alpha.shape==data['y'].shape and np.isfinite(pred).all() and np.isfinite(alpha).all()
    assert ((alpha>=0)&(alpha<=1)).all()
    return score_predictions(pred,data['y'],data['subject'],data['required_subjects']),pred,alpha

def train_arm(out,method,seed,cfg,train,val,initial,device='cpu',scope='REAL_DEVELOPMENT',backbone_factory=build_model):
    out=Path(out);out.mkdir(parents=True,exist_ok=False) # Never rewrite a completed/partial/failed run.
    started=time.monotonic();updates=0;history=[];order_hashes=[]
    try:
        validate_data(train);validate_data(val)
        assert not set(train['subject']) & set(val['subject'])
        n=len(train['y']);batch=cfg['batch_size'];epoch_batches=math.ceil(n/batch)
        assert epoch_batches==cfg['updates_per_full_training_epoch']==cfg['validation_interval_updates']
        assert batch>=3 and set(map(int,val['subject']))==set(cfg['validation_subjects']) and set(map(int,train['subject']))==set(cfg['train_subjects'])
        assert cfg['max_updates_per_method_seed']%epoch_batches==0
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
        backbone=backbone_factory();backbone.load_state_dict(initial,strict=True)
        assert all(torch.equal(v.cpu(),initial[k].cpu()) for k,v in backbone.state_dict().items())
        model=Variant(backbone,method).to(device)
        torch.manual_seed(seed) # Reset after gate construction, in addition to independent batch generator.
        optimizer,groups=optimizer_for(model,cfg)
        initial_gate={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not n.startswith('backbone.')}
        metadata={'scope':scope,'method':method,'seed':seed,'resume_supported':False,'config':cfg,
                  'optimizer_groups':groups,'fresh_optimizer_state_entries':len(optimizer.state),
                  'train_subjects':sorted(map(int,set(train['subject']))),'validation_subjects':sorted(map(int,set(val['subject']))),
                  'train_window_count':n,'validation_window_count':len(val['y']),
                  'initial_state_tensor_sha256':{k:array_sha(v.cpu().numpy()) for k,v in initial.items()},
                  'ordered_input_sha256':{role:{k:array_sha(data[k]) for k in ('x','d','f','y','subject','activity','start')} for role,data in [('train',train),('validation',val)]}}
        (out/'config_and_groups.json').write_text(json.dumps(metadata,indent=2))
        with (out/'eligible.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['subject','start_sample_32hz','reference_bpm'])
            w.writerows(zip(map(int,val['subject']),map(int,val['start']),map(float,val['y'])))
        best=math.inf;stale=0;best_update=None;stop='max_updates'
        for epoch in range(cfg['max_updates_per_method_seed']//epoch_batches):
            order=epoch_order(n,seed,epoch);order_hashes.append(array_sha(order));model.train();loss_sum=0.;seen=0
            for ids in batch_indices(order,batch):
                args=[torch.from_numpy(train[k][ids]).to(device) for k in ('x','d','f')]
                target=torch.from_numpy(train['y'][ids,None]).to(device)
                optimizer.zero_grad(set_to_none=True);prediction,_=model(*args)
                assert prediction.shape==target.shape
                error=prediction-target;loss=(error+nn.functional.softplus(-2*error)-math.log(2)).mean()
                assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert all(p.grad is not None for n,p in model.named_parameters() if not n.startswith('backbone.'))
                optimizer.step();updates+=1;seen+=len(ids);loss_sum+=loss.item()*len(ids)
                assert all(torch.isfinite(p).all() for p in model.parameters())
            assert seen==n and updates%cfg['validation_interval_updates']==0
            scores,_,_=evaluate(model,val,device,batch)
            assert all(math.isfinite(scores[k]) for k in ('validation_macro_logcosh','validation_pooled_logcosh','pooled_mae','subject_macro_mae'))
            row={'epoch':epoch+1,'updates':updates,'train_logcosh':loss_sum/seen,'batch_order_sha256':order_hashes[-1],**scores}
            history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2))
            if scores['validation_macro_logcosh']<best:
                best=scores['validation_macro_logcosh'];best_update=updates;stale=0;torch.save(model.state_dict(),out/'best.pt')
            else:stale+=1
            if stale>=cfg['early_stop_patience_validation_checks']:stop='validation_early_stop';break
        torch.save(model.state_dict(),out/'last.pt')
        gate_delta=sum(float((p.detach().cpu()-initial_gate[n]).abs().sum()) for n,p in model.named_parameters() if n in initial_gate)
        model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True));scores,pred,alpha=evaluate(model,val,device,batch)
        best_gate_delta=sum(float((p.detach().cpu()-initial_gate[n]).abs().sum()) for n,p in model.named_parameters() if n in initial_gate)
        with (out/'predictions.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['scope','method','seed','subject','activity','window_id','start_seconds','end_seconds','reference_bpm','prediction_bpm','filtered_weight_alpha','eligible','status','role'])
            for j in range(len(pred)):
                w.writerow([scope,method,seed,int(val['subject'][j]),int(val['activity'][j]),int(val['start'][j]//64),float(val['start'][j]/32),float(val['start'][j]/32+8),float(val['y'][j]),float(pred[j]),float(alpha[j]),1,'ok','development_validation'])
        summary={'scope':scope,'status':'completed','method':method,'seed':seed,'actual_updates':updates,'best_update':best_update,
                 'stop_reason':stop,'history_checks':len(history),'eligible_windows':len(pred),'output_windows':len(pred),'coverage':1.0,
                 'scores':scores,'gate_parameter_l1_change_at_last':gate_delta,'resume_supported':False,'test_accessed':False,
                 'gate_parameter_l1_change_at_best':best_gate_delta,
                 'host_train_validation_output_seconds':time.monotonic()-started,'timing_excludes_frontend_cache_generation':True,
                 'batch_order_hashes':order_hashes,'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()}
        (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary
    except BaseException as e:
        (out/'failure.json').write_text(json.dumps({'scope':scope,'status':'failed','method':method,'seed':seed,'actual_updates_before_failure':updates,
            'exception':repr(e),'traceback':traceback.format_exc(),'resume_supported':False,'policy':'Preserve this directory; retry from shared initial state in a new bundle, never resume or overwrite.'},indent=2))
        raise
