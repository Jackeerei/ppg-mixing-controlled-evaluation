"""One complete external evaluation of all 15 fixed checkpoints, after mapping check."""
from pathlib import Path
import argparse,csv,datetime,json,shutil,sys,time,traceback
import numpy as np
import psutil,torch
from adapter import sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parent
V=ROOT/'candidate_versions/round2_v1_4';LEDGER=ROOT/'logs/wrist_external_v1_attempt.json'
sys.path.insert(0,str(V/'executor'))
import engine
from frontend import process_windows
def read(p):return json.loads(Path(p).read_text())
def put(p,v,mode='x'):
    with p.open(mode,encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def table(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def evaluate(prepared,review):
    import msvcrt
    with (ROOT/'wrist_external_v1.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        assert not LEDGER.exists(),'Already attempted; preserve previous evaluation'
        assert prepared.is_relative_to(ROOT/'reports') and review.is_relative_to(PROJECT/'review') and review.is_file()
        for r in read(prepared/'MANIFEST.json')['files']:assert sha(prepared/r['path'])==r['sha256']
        assert read(prepared/'PREPARATION.json')['status']=='ready_for_mapping_check'
        assert psutil.virtual_memory().available>=2.5*1024**3 and shutil.disk_usage(ROOT).free>=10*1024**3
        models=read(prepared/'models.json')['models'];assert {(m['method'],m['seed']) for m in models}=={(m,s) for m in 'ABCGD' for s in [42,43,44]}
        for m in models:assert sha(Path(m['checkpoint']))==m['checkpoint_sha256']
        out=ROOT/'results'/('wrist_external_v1_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
        intent={'status':'running','scope':'WRIST_EXTERNAL_PHYSICAL_UNIT_NOMINAL_INDEX','prepared':str(prepared),'prepared_manifest_sha256':sha(prepared/'MANIFEST.json'),'mapping_review':str(review),'mapping_review_sha256':sha(review),'source_sha256':sha(Path(__file__)),'bundle':str(out),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        put(LEDGER,intent);put(out/'intent.json',intent);shutil.copy2(__file__,out/'inference_source.py');shutil.copy2(prepared/'models.json',out/'models.json');shutil.copy2(prepared/'config.json',out/'adapter_config.json')
        started=time.monotonic();torch.set_num_threads(2);scores=[];subject_activity=[];record_metrics=[];sources=[];failures=[]
        try:
            with (prepared/'candidate_windows.csv').open(newline='') as f:candidates=list(csv.DictReader(f))
            eligible=[r for r in candidates if r['eligible']=='True'];assert len(eligible)==3320 and len(candidates)==3332
            xparts=[]
            for record in read(prepared/'PREPARATION.json')['records']:
                with np.load(prepared/(record['record']+'.npz'),allow_pickle=False) as z:
                    rr=[r for r in eligible if r['record']==record['record']]
                    assert len(rr)==len(z['x']) and np.array_equal(z['candidate_window_ids'],np.array([int(r['window_id']) for r in rr]))
                    xparts.append(z['x'])
            data={'x':np.concatenate(xparts),'y':np.asarray([float(r['reference_bpm']) for r in eligible],dtype=np.float64),'subject':np.asarray([int(r['subject']) for r in eligible]),'required_subjects':[1,2,3,4,5,6,8,9]}
            front_start=time.monotonic();front=process_windows(data['x']);data.update(d=front['filtered_ppg'],f=front['features']);front_seconds=time.monotonic()-front_start
            np.savez(out/'external_inputs_and_frontend.npz',**{k:v for k,v in data.items() if k!='required_subjects'})
            keys=[(r['record'],r['window_id']) for r in eligible];assert len(set(keys))==len(keys);key_to_i={key:i for i,key in enumerate(keys)}
            subjects=data['required_subjects'];activities=sorted(set(r['activity'] for r in candidates));metrics={}
            for m in models:
                assert time.monotonic()-started<1800,'1800-second evaluation ceiling'
                method=m['method'];seed=m['seed'];arm=out/f'seed{seed}_{method}';arm.mkdir(exist_ok=False);success=True;pred=None;alpha=None
                try:
                    model=engine.Variant(engine.build_model(),method);model.load_state_dict(torch.load(m['checkpoint'],weights_only=True,map_location='cpu'));model.eval()
                    t=time.monotonic();sc,pred,alpha=engine.evaluate(model,data,'cpu',128);duration=time.monotonic()-t
                    assert len(pred)==len(eligible) and np.isfinite(pred).all() and np.isfinite(alpha).all() and ((alpha>=0)&(alpha<=1)).all()
                    if method in 'ABC':assert (alpha=={'A':0,'B':1,'C':.5}[method]).all()
                    summary={'status':'completed','method':method,'seed':seed,'checkpoint_sha256':m['checkpoint_sha256'],'candidate_windows':len(candidates),'eligible_windows':len(eligible),'prediction_success':len(pred),'coverage_over_candidates':len(pred)/len(candidates),'coverage_over_eligible':1.,'external_subject_macro_mae':sc['subject_macro_mae'],'external_pooled_mae':sc['pooled_mae'],'per_subject':sc['per_subject'],'inference_and_metric_seconds':duration,'scope':intent['scope']}
                    scores.append(summary)
                    for s in subjects:metrics[s,method,seed]=sc['per_subject'][str(s)]['mae']
                    error=np.abs(pred-data['y'])
                    for rec in sorted(set(r['record'] for r in candidates)):
                        mask=np.array([r['record']==rec for r in eligible]);cr=[r for r in candidates if r['record']==rec];n=int(mask.sum())
                        record_metrics.append({'method':method,'seed':seed,'record':rec,'subject':cr[0]['subject'],'activity':cr[0]['activity'],'candidate_windows':len(cr),'eligible_windows':n,'prediction_success':n,'coverage':n/len(cr),'mae_bpm':float(error[mask].mean()) if n else None})
                    for s in subjects:
                        for a in activities:
                            cr=[r for r in candidates if int(r['subject'])==s and r['activity']==a];mask=np.array([int(r['subject'])==s and r['activity']==a for r in eligible]);n=int(mask.sum())
                            subject_activity.append({'method':method,'seed':seed,'subject':s,'activity':a,'candidate_windows':len(cr),'eligible_windows':n,'prediction_success':n,'coverage':n/len(cr) if cr else None,'mae_bpm':float(error[mask].mean()) if n else None})
                    put(arm/'summary.json',summary);del model
                except Exception as exc:
                    success=False;failure={'method':method,'seed':seed,'status':'inference_failed','exception':repr(exc),'traceback':traceback.format_exc()};failures.append(failure);put(arm/'failure.json',failure)
                output=[]
                for r in candidates:
                    ok=r['eligible']=='True';i=key_to_i.get((r['record'],r['window_id']))
                    output.append({**r,'scope':intent['scope'],'method':method,'seed':seed,'prediction_bpm':float(pred[i]) if ok and success else None,'filtered_weight_alpha':float(alpha[i]) if ok and success else None,'prediction_status':'ok' if ok and success else ('model_failure' if ok else 'excluded')})
                table(arm/'predictions.csv',output);sources.append({'method':method,'seed':seed,'checkpoint_sha256':m['checkpoint_sha256'],'predictions_sha256':sha(arm/'predictions.csv')})
                print(json.dumps({'method':method,'seed':seed,'status':'completed' if success else 'failed','macro_mae':scores[-1]['external_subject_macro_mae'] if success else None}),flush=True)
            complete=len(scores)==15 and not failures;contrasts={};method_summary={}
            if complete:
                for method in 'ABCGD':
                    rr=[r for r in scores if r['method']==method];maes=[r['external_subject_macro_mae'] for r in rr]
                    method_summary[method]={'subject_macro_mae_mean':float(np.mean(maes)),'subject_macro_mae_seed_sample_sd':float(np.std(maes,ddof=1)),'pooled_mae_mean':float(np.mean([r['external_pooled_mae'] for r in rr]))}
                for c in 'CGAB':
                    per={str(s):[metrics[s,'D',k]-metrics[s,c,k] for k in [42,43,44]] for s in subjects};means={s:float(np.mean(v)) for s,v in per.items()};seedmeans=[float(np.mean([v[i] for v in per.values()])) for i in range(3)]
                    contrasts['D-'+c]={'subject_seed_deltas':per,'subject_seed_mean':means,'equal_subject_mean':float(np.mean(list(means.values()))),'per_seed_subject_macro_delta':seedmeans,'seed_macro_delta_sample_sd':float(np.std(seedmeans,ddof=1))}
            report={'status':'completed' if complete else 'incomplete','scope':intent['scope'],'subject_count':len(subjects),'record_count':19,'candidate_windows_per_model':len(candidates),'eligible_windows_per_model':len(eligible),'shared_excluded_windows':len(candidates)-len(eligible),'model_summary':method_summary,'arms':scores,'contrasts':contrasts,'failures':failures,'sources':sources,'frontend_seconds':front_seconds,'elapsed_seconds':time.monotonic()-started,'test_accessed':False,'training_performed':False,'external_tuning':False,'limitations':read(prepared/'config.json')['limitations']}
            put(out/'aggregate.json',report)
            if record_metrics:table(out/'record_metrics.csv',record_metrics)
            if subject_activity:table(out/'subject_activity_metrics.csv',subject_activity)
            lines=['## Material Passport','','- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: run','- Origin Date: 2026-11-07','- Verification Status: UNVERIFIED','- Version Label: wrist_external_v1','','Wrist全19条记录、8人，固定原15 best权重/物理mV与g单位/作者名义时间索引，未训练或外部调参。','',f'每模型候选{len(candidates)}，eligible {len(eligible)}，共同比例{len(eligible)/len(candidates):.6%}；全部排除窗和失败保留逐窗状态。','','|方法|8人macro MAE mean ± seed SD (BPM)|pooled MAE均值|','|---|---:|---:|']
            for m,v in method_summary.items():lines.append(f"|{m}|{v['subject_macro_mae_mean']:.6f} ± {v['subject_macro_mae_seed_sample_sd']:.6f}|{v['pooled_mae_mean']:.6f}|")
            lines+=['','原始PPG物理mV与训练E4输出尺度不等价；名义同索引保留原作者同步局限；此处跨设备结果必须带此固定适配范围解释。逐seed/逐人原值及D-C/D-G配对见aggregate.json，不能用外部结果挑新缩放覆盖本轮。']
            (out/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8');intent.update(status=report['status'],finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());put(out/'terminal.json',intent);put(LEDGER,intent,mode='w')
            put(out/'MANIFEST.json',{'files':[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]})
            print(json.dumps({'bundle':str(out),'status':report['status'],'summary':method_summary,'contrasts':contrasts}),flush=True)
        except BaseException as exc:
            intent.update(status='failed',exception=repr(exc));put(out/'execution_failure.json',intent);put(LEDGER,intent,mode='w');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--mapping-review',required=True);a=p.parse_args();evaluate(Path(a.prepared).resolve(),Path(a.mapping_review).resolve())
