"""Synthetic only: macro selection, full coverage batching and bounded CPU cost."""
from pathlib import Path
import json,time,copy
import numpy as np
import torch
from torch import nn
import engine
import runner
from cache import get_or_create
HERE=Path(__file__).resolve().parent
class Toy(nn.Module):
    def __init__(self):super().__init__();self.fc=nn.Linear(4,1)
    def forward(self,x):return self.fc(x.mean(2))
def data(n,subjects):
    r=np.random.default_rng(n);x=r.normal(size=(n,4,256)).astype(np.float32)
    return {'x':x,'y':np.full(n,3,np.float32),'d':x[:,0]+1,'f':r.random((n,3),dtype=np.float32),
            'subject':np.asarray(subjects,np.int16),'start':np.arange(n)*64,'activity':np.zeros(n,np.int16),'required_subjects':[11,12]}
def main():
    torch.set_num_threads(2);out=HERE/('synthetic_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));out.mkdir(exist_ok=False)
    checks={}
    for n,expected in [(128,[128]),(129,[127,2]),(257,[128,127,2])]:
        order=engine.epoch_order(n,42,0);parts=engine.batch_indices(order,128)
        assert list(map(len,parts))==expected and np.array_equal(np.concatenate(parts),order)
    checks['tail_all_windows_once_and_exact_updates']=True
    subjects=np.array([11]+[12]*9);y=np.zeros(10)
    x=engine.score_predictions(np.arccosh(np.exp([1]+[9]*9)),y,subjects,[11,12])
    z=engine.score_predictions(np.arccosh(np.exp([6]+[5]*9)),y,subjects,[11,12])
    assert x['validation_macro_logcosh']<z['validation_macro_logcosh'] and x['validation_pooled_logcosh']>z['validation_pooled_logcosh']
    for pred,sub in [(np.zeros(10),np.full(10,11)),(np.full(10,np.nan),subjects)]:
        try:engine.score_predictions(pred,y,sub,[11,12])
        except AssertionError:pass
        else:raise AssertionError('Missing/nonfinite subject accepted')
    checks['unequal_subject_macro_pooled_reversal_and_missing_nonfinite_rejection']=True
    old=json.loads((engine.ROOT/'candidate_versions/v0.1.1_ci01_2ee6e442e02d/development_protocol.json').read_text())
    cfg=copy.deepcopy(old);cfg.update(train_subjects=[1],validation_subjects=[11,12],batch_size=128,updates_per_full_training_epoch=2,validation_interval_updates=2,max_updates_per_method_seed=12,early_stop_patience_validation_checks=2)
    train=data(129,[1]*129);val=data(10,subjects);original=engine.evaluate;calls=[0]
    def injected(*args):
        scores,p,a=original(*args);i=min(calls[0],3);calls[0]+=1
        scores['validation_macro_logcosh']=[5,5.5,6,6][i];scores['validation_pooled_logcosh']=[8.2,5.1,4,3][i];return scores,p,a
    try:
        engine.evaluate=injected
        summary=engine.train_arm(out/'selection','D',42,cfg,train,val,Toy().state_dict(),scope='SYNTHETIC_INJECTED_SELECTION_ONLY',backbone_factory=Toy)
    finally:engine.evaluate=original
    assert summary['best_update']==2 and summary['actual_updates']==6 and summary['stop_reason']=='validation_early_stop'
    checks['actual_optimizer_macro_best_single_checkpoint_patience_full_epoch']=True
    model=engine.Variant(Toy(),'D');model.load_state_dict(torch.load(out/'selection/best.pt',weights_only=True))
    _,pred,_=original(model,val,'cpu',128)
    import csv
    with (out/'selection/predictions.csv').open() as f:saved=list(csv.DictReader(f))
    assert np.array_equal(pred,np.array([float(r['prediction_bpm']) for r in saved]))
    class FailPartial(Toy):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,x):
            self.calls+=1;v=super().forward(x);return v if self.calls==1 else v*float('nan')
    try:engine.train_arm(out/'partial','A',42,cfg,train,val,Toy().state_dict(),scope='SYNTHETIC_PARTIAL_FAILURE',backbone_factory=FailPartial)
    except AssertionError:pass
    else:raise AssertionError('Partial epoch did not fail')
    failure=json.loads((out/'partial/failure.json').read_text());assert failure['actual_updates_before_failure']==1 and not (out/'partial/history.json').exists() and not (out/'partial/best.pt').exists()
    checks['partial_epoch_no_validation_or_checkpoint_and_best_exact_reload']=True
    p=get_or_create(out/'cache',val['x'],'SYNTHETIC_PROPOSAL')
    with np.load(p) as z:
        bridge={**val,'d':z['filtered_ppg'],'f':z['features']};engine.validate_data(bridge);assert len(bridge['d'])==10
    p2=get_or_create(out/'cache',val['x'],'SYNTHETIC_PROPOSAL');assert p==p2
    checks['actual_production_cache_bridge_and_reuse']=True
    metrics={(s,m,k):float(s+k+list('ABCGD').index(m)) for s in [11,12] for m in 'ABCGD' for k in [42,43,44]}
    pairs=runner.paired(metrics);assert pairs['D-C']['equal_subject_mean']==2 and pairs['D-G']['equal_subject_mean']==1
    del metrics[11,'A',42]
    try:runner.paired(metrics)
    except AssertionError:pass
    else:raise AssertionError('Missing subject/seed accepted')
    checks['complete_two_by_three_pairs_and_incomplete_comparison_rejected']=True
    model=engine.Variant(engine.build_model(),'D');opt,_=engine.optimizer_for(model,old)
    sample=data(128,[1]*128);args=[torch.from_numpy(sample[k]) for k in ['x','d','f']];times=[]
    for i in range(23):
        start=time.perf_counter();opt.zero_grad(set_to_none=True);pred,_=model(*args);loss=pred.square().mean();loss.backward();opt.step()
        if i>=3:times.append(time.perf_counter()-start)
    model.eval();valtimes=[]
    with torch.inference_mode():
        for i in range(23):
            start=time.perf_counter();model(*args)
            if i>=3:valtimes.append(time.perf_counter()-start)
    report={'status':'passed','scope':'SYNTHETIC_SHAPE_AND_CONTROL_ONLY_NO_REAL_DEVELOPMENT_ERRORS','checks':checks,
            'train_step_seconds_p95':float(np.percentile(times,95)),'validation_batch_seconds_p95':float(np.percentile(valtimes,95)),
            'device':'cpu','threads':2,'batch':128,'warmup':3,'measured':20,'memory_note':'Array memory is estimated separately from actual N; no stress/peak-memory claim.',
            'selection_injection_note':'Only synthetic selection scalars injected; does not represent HR metrics or convergence.'}
    (out/'checks_and_cost.json').write_text(json.dumps(report,indent=2));print(json.dumps({'directory':str(out),**report},indent=2))
if __name__=='__main__':main()
