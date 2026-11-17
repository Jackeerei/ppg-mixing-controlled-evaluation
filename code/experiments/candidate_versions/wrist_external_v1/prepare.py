from pathlib import Path
import csv,datetime,json,shutil,sys,time,traceback
import numpy as np
from adapter import adapt_record,sha,header
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DATA=ROOT/'data/raw/external_public/wrist-1.0.0'
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    freeze=json.loads((HERE/'FREEZE.json').read_text())
    for r in freeze['files']:assert sha(HERE/r['path'])==r['sha256']
    cfg=json.loads((HERE/'config.json').read_text());models=json.loads((HERE/'models.json').read_text())
    for m in models['models']:assert sha(Path(m['checkpoint']))==m['checkpoint_sha256']
    for line in (DATA/'SHA256SUMS.txt').read_text().splitlines():
        h,n=line.split(maxsplit=1);assert sha(DATA/n)==h
    out=ROOT/'reports'/('wrist_external_prepare_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,out/p.name)
    rows=[];records=[];examples=[];began=time.monotonic()
    for record in (DATA/'RECORDS').read_text().splitlines():
        assert time.monotonic()-began<600
        try:
            rr,x,ids,report,example=adapt_record(DATA,record)
            rows.extend(rr);records.append(report)
            with (out/(record+'.npz')).open('xb') as f:np.savez(f,x=x,candidate_window_ids=ids)
            if example:examples.append(example)
            print(json.dumps({'record':record,'candidates':len(rr),'eligible':len(x),'wraps':report['timing']['wrap_count']}),flush=True)
        except Exception as exc:
            h=header(DATA/(record+'.hea'));n=max(0,(h['samples']-2048)//512+1)
            records.append({'record':record,'status':'preparation_failed','candidate_windows':n,'eligible_windows':0,'exception':repr(exc),'traceback':traceback.format_exc()})
            for k in range(n):rows.append({'record':record,'subject':int(record.split('_')[0][1:]),'activity':record.split('_',1)[1],'window_id':k,'start_sample_256hz':k*512,'stop_sample_256hz':k*512+2048,'start_seconds':k*2,'end_seconds':k*2+8,'input_available':None,'reference_available':None,'eligible':False,'reference_bpm':None,'r_peak_count':None,'first_r_sample':None,'last_r_sample':None,'reason':'record_preparation_failed'})
            print(json.dumps({'record':record,'error':repr(exc)}),flush=True)
    with (out/'candidate_windows.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'status':'ready_for_mapping_check' if all(r.get('status')!='preparation_failed' for r in records) else 'preparation_has_record_failures','record_count':len(records),'candidate_windows':len(rows),'eligible_windows':sum(r['eligible'] for r in rows),'records':records,'external_model_predictions_computed':False,'config_sha256':sha(HERE/'config.json'),'adapter_freeze_sha256':sha(HERE/'FREEZE.json'),'models_sha256':sha(HERE/'models.json'),'seconds':time.monotonic()-began}
    write(out/'PREPARATION.json',result);write(out/'HAND_MAPPING.json',examples)
    (out/'HANDOFF.md').write_text('## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n- Origin Mode: run\n- Origin Date: 2026-11-07\n- Verification Status: UNVERIFIED\n- Version Label: wrist_external_v1\n\n19记录固定名义索引适配准备；未计算任何模型预测或误差。配置/15份最佳权重SHA/全部candidate资格/无效尾补/时间回绕及注释基准见PREPARATION.json、candidate_windows.csv、HAND_MAPPING.json。一次必要映射检查通过后直接完整推理，不在外部调参选模型。原论文时间局限和PPG振幅跨域局限保留。\n',encoding='utf-8')
    write(out/'MANIFEST.json',{'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({'output':str(out),'status':result['status'],'candidate_windows':result['candidate_windows'],'eligible_windows':result['eligible_windows']}),flush=True)
if __name__=='__main__':main()
