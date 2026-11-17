"""Final-only adapter of reviewed wrist_external_v1/infer.py; default verifies only."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,shutil,sys,time,traceback

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parent
LEDGER=ROOT/'logs/final_evaluation_v1_attempt.json'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def put(p,v,mode='x'):
    with p.open(mode,encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)

def table(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def verify():
    assert not sys.flags.optimize
    cfg=read(HERE/'execution_config.json');prepared=Path(cfg['prepared'])
    assert sha(prepared/'PREPARATION_MANIFEST.json')==cfg['preparation_manifest_sha256']
    assert sha(prepared/'models.json')==cfg['models_sha256']
    assert sha(prepared/'config.json')==cfg['preparation_config_sha256']
    assert read(prepared/'config.json')['final_predictions_allowed'] is False
    for r in read(HERE/'EXECUTION_FREEZE.json')['files']:assert sha(PROJECT/r['path'])==r['sha256']
    for r in read(prepared/'PREPARATION_MANIFEST.json')['files']:assert sha(prepared/r['path'])==r['sha256']
    models=read(prepared/'models.json')['models']
    expected={('original_CPU_five_methods',m,s,'cpu') for m in 'ABCGD' for s in [42,43,44]}|{('paired_CUDA_c_ablation',m,s,'cuda') for m in ['D_full','D_without_c'] for s in [42,43,44]}
    assert len(models)==21 and {(m['set'],m['method'],m['seed'],m['training_device']) for m in models}==expected
    assert cfg['subjects']==[13,14,15] and cfg['inference']=={'device':'cpu','dtype':'float32','batch_size':128,'threads':2,'AMP':False,'TF32':False}
    return cfg,prepared,models

def evaluate(review):
    # No torch import, model construction, weight decoding or NPZ load in default verification.
    import msvcrt
    cfg,prepared,models=verify()
    assert review.is_relative_to(PROJECT/'review') and review.is_file()
    receipt=read(review)
    assert receipt['status']=='passed' and receipt['execution_freeze_sha256']==sha(HERE/'EXECUTION_FREEZE.json')
    assert receipt['preparation_manifest_sha256']==cfg['preparation_manifest_sha256']
    with (ROOT/'final_evaluation_v1.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        assert not LEDGER.exists(),'Already attempted; preserve ledger and all outputs; no automatic restart'
        out=ROOT/'results'/('final_evaluation_v1_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
        intent={'status':'running','bundle':str(out),'execution_freeze_sha256':sha(HERE/'EXECUTION_FREEZE.json'),'preparation_manifest_sha256':cfg['preparation_manifest_sha256'],'models_sha256':cfg['models_sha256'],'review':str(review),'review_sha256':sha(review),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_performed':False,'test_accessed':True}
        put(LEDGER,intent);put(out/'intent.json',intent)
        started=time.monotonic();scores=[];failures=[];activity_rows=[];metrics={}
        try:
            for p in [HERE/'infer.py',HERE/'execution_config.json',HERE/'EXECUTION_FREEZE.json',prepared/'models.json',prepared/'config.json',prepared/'PREPARATION_MANIFEST.json']:shutil.copy2(p,out/p.name)
            with (prepared/'candidate_windows.csv').open(newline='') as f:candidates=list(csv.DictReader(f))
            eligible=[r for r in candidates if r['eligible']=='1']
            assert len(candidates)==len(eligible)==cfg['candidate_windows']==13007
            arms={}
            for m in models:
                arm=out/f"{m['set']}_seed{m['seed']}_{m['method']}";arm.mkdir(exist_ok=False);arms[m['set'],m['method'],m['seed']]=arm
                table(arm/'predictions.csv',[{**r,**{k:m[k] for k in ['set','method','seed','training_device']},'inference_device':'cpu','prediction_bpm':None,'filtered_weight_alpha':None,'prediction_status':'not_started' if r['eligible']=='1' else 'excluded'} for r in candidates])
            import numpy as np
            import psutil,torch
            assert psutil.virtual_memory().available>=2.5*1024**3 and shutil.disk_usage(ROOT).free>=10*1024**3
            for r in read(prepared/'data_bindings.json')['files']:assert sha(r['path'])==r['sha256']
            for m in models:assert sha(m['checkpoint'])==m['sha256']
            sys.path.insert(0,str(ROOT/'candidate_versions/round2_v1_4/executor'))
            import engine
            from frontend import process_windows
            sys.path.insert(0,str(ROOT/'candidate_versions/c_ablation_cuda_v1'))
            from gate import GateAblation
            torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            parts=[]
            for s in cfg['subjects']:
                with np.load(ROOT/f'data/processed/S{s}.npz',allow_pickle=False) as z:
                    rows=[r for r in eligible if int(r['subject'])==s];n=len(rows)
                    assert len(z['y'])==n and z['x'].dtype==z['y'].dtype==np.float32
                    assert np.array_equal(z['y'],np.asarray([float(r['reference_bpm']) for r in rows],dtype=np.float32))
                    assert np.array_equal(z['start_sample_32hz'],np.asarray([int(r['start_sample_32hz']) for r in rows]))
                    assert np.array_equal(z['activity'],np.asarray([int(r['activity']) for r in rows]))
                    parts.append({k:z[k].copy() for k in ['x','y']})
            data={k:np.concatenate([p[k] for p in parts]) for k in ['x','y']}
            data.update(subject=np.asarray([int(r['subject']) for r in eligible]),activity=np.asarray([int(r['activity']) for r in eligible]),start=np.asarray([int(r['start_sample_32hz']) for r in eligible]),required_subjects=cfg['subjects'])
            t=time.monotonic();front=process_windows(data['x']);data.update(d=front['filtered_ppg'],f=front['features']);front_seconds=time.monotonic()-t
            engine.validate_data(data)
            np.savez(out/'final_inputs_and_frontend.npz',**{k:v for k,v in data.items() if k!='required_subjects'})
            for m in models:
                assert time.monotonic()-started<cfg['maximum_seconds'],'Single-run time bound; no automatic restart'
                arm=arms[m['set'],m['method'],m['seed']];success=False;pred=alpha=None
                try:
                    factory=engine.Variant if m['set']=='original_CPU_five_methods' else GateAblation
                    model=factory(engine.build_model(),m['method'])
                    model.load_state_dict(torch.load(m['checkpoint'],weights_only=True,map_location='cpu'),strict=True);model.eval()
                    t=time.monotonic();sc,pred,alpha=engine.evaluate(model,data,'cpu',128);duration=time.monotonic()-t
                    if m['method'] in 'ABC':assert (alpha=={'A':0,'B':1,'C':.5}[m['method']]).all()
                    summary={**m,'status':'completed','inference_device':'cpu','inference_dtype':'float32','candidate_windows':len(candidates),'eligible_windows':len(eligible),'prediction_success':len(pred),'coverage_over_candidates':len(pred)/len(candidates),'coverage_over_eligible':1.,'subject_macro_mae':sc['subject_macro_mae'],'pooled_mae':sc['pooled_mae'],'per_subject':sc['per_subject'],'inference_and_metric_seconds':duration}
                    put(arm/'summary.json',summary);scores.append(summary);success=True
                    for s in cfg['subjects']:metrics[m['set'],m['method'],m['seed'],s]=sc['per_subject'][str(s)]['mae']
                    error=np.abs(pred-data['y'].astype(np.float64))
                    for s in cfg['subjects']:
                        for a in range(9):
                            mask=(data['subject']==s)&(data['activity']==a);n=int(mask.sum())
                            activity_rows.append({**{k:m[k] for k in ['set','method','seed']},'subject':s,'activity':a,'candidate_windows':n,'eligible_windows':n,'prediction_success':n,'coverage':1. if n else None,'mae_bpm':float(error[mask].mean()) if n else None})
                    del model
                except Exception as exc:
                    failure={**m,'status':'inference_failed','exception':repr(exc),'traceback':traceback.format_exc()};failures.append(failure);put(arm/'failure.json',failure)
                table(arm/'predictions.csv',[{**r,**{k:m[k] for k in ['set','method','seed','training_device']},'inference_device':'cpu','prediction_bpm':float(pred[i]) if success else None,'filtered_weight_alpha':float(alpha[i]) if success else None,'prediction_status':'ok' if success else 'model_failure'} for i,r in enumerate(candidates)])
                print(json.dumps({'set':m['set'],'method':m['method'],'seed':m['seed'],'status':'completed' if success else 'failed'}),flush=True)
            complete=len(scores)==21 and not failures;groups={};contrasts={};paired=[]
            if complete:
                for group,methods in [('original_CPU_five_methods',list('ABCGD')),('paired_CUDA_c_ablation',['D_full','D_without_c'])]:
                    groups[group]={}
                    for method in methods:
                        rr=[r for r in scores if r['set']==group and r['method']==method];values=[r['subject_macro_mae'] for r in rr]
                        groups[group][method]={'subject_macro_mae_mean':float(np.mean(values)),'subject_macro_mae_seed_sample_sd':float(np.std(values,ddof=1)),'pooled_mae_mean':float(np.mean([r['pooled_mae'] for r in rr]))}
                for group,left,right in [('original_CPU_five_methods','D',c) for c in 'CGAB']+[('paired_CUDA_c_ablation','D_full','D_without_c')]:
                    cells={str(s):[metrics[group,left,k,s]-metrics[group,right,k,s] for k in [42,43,44]] for s in cfg['subjects']};means={s:float(np.mean(v)) for s,v in cells.items()};byseed=[float(np.mean([v[i] for v in cells.values()])) for i in range(3)]
                    contrasts[left+'-'+right]={'set':group,'subject_seed_deltas':cells,'subject_seed_mean':means,'equal_subject_mean':float(np.mean(list(means.values()))),'per_seed_subject_macro_delta':byseed,'seed_macro_delta_sample_sd':float(np.std(byseed,ddof=1))}
                    for s,values in cells.items():
                        for seed,value in zip([42,43,44],values):paired.append({'set':group,'contrast':left+'-'+right,'subject':int(s),'seed':seed,'mae_delta_bpm':value})
            report={'status':'completed' if complete else 'incomplete','subject_count':3,'seeds':[42,43,44],'candidate_windows_per_model':len(candidates),'eligible_windows_per_model':len(eligible),'excluded_windows_per_model':0,'model_sets':groups,'arms':scores,'contrasts':contrasts,'failures':failures,'frontend_seconds':front_seconds,'elapsed_seconds':time.monotonic()-started,'training_performed':False,'test_accessed':True,'test_driven_adaptation':False,'seed_and_windows_are_not_independent_people':True}
            put(out/'aggregate.json',report)
            if activity_rows:table(out/'subject_activity_metrics.csv',activity_rows)
            if paired:table(out/'paired_subject_seed_deltas.csv',paired)
            intent['status']=report['status']
        except BaseException as exc:
            intent.update(status='failed',exception=repr(exc),traceback=traceback.format_exc());put(out/'execution_failure.json',intent);raise
        finally:
            intent.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
            put(out/'terminal.json',intent);put(LEDGER,intent,mode='w')
            put(out/'MANIFEST.json',{'files':[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]})
        print(json.dumps({'bundle':str(out),'status':intent['status']}),flush=True)
        if intent['status']!='completed':raise RuntimeError('Incomplete evaluation preserved; no automatic restart')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--review');a=p.parse_args()
    if a.execute:
        assert a.review,'Passed frozen preparation/execution review receipt required'
        evaluate(Path(a.review).resolve())
    else:
        cfg,prepared,models=verify();print(json.dumps({'status':'configuration_verified_only','models':len(models),'prepared':str(prepared),'model_loaded':False,'predictions_computed':False,'execution_freeze_sha256':sha(HERE/'EXECUTION_FREEZE.json')}))
