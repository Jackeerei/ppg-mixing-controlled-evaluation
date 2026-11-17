"""Bind all 12 real prepared subjects and freeze one reviewed-protocol candidate; no training."""
from pathlib import Path
import json,math,hashlib
ROOT=Path(__file__).resolve().parents[1];V=ROOT/'candidate_versions/round2_v1'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
def main():
    subjects=[]
    for s in range(1,13):
        mpath=ROOT/f'data/manifests/S{s}_windows.json';rpath=ROOT/f'data/manifests/S{s}_recovery.json';npz=ROOT/f'data/processed/S{s}.npz'
        m=json.loads(mpath.read_text());r=json.loads(rpath.read_text());assert m['subject']==s and r['member_complete'] and r['crc_verified'] and r['deflate_eof']
        assert m['excluded_windows']==0 and m['candidate_windows']==m['input_shape'][0]
        files=[mpath,rpath,npz,Path(r['output'])];audit=ROOT/f'data/manifests/S{s}_data_audit.json'
        if s not in [10,11]:
            a=json.loads(audit.read_text());assert a['status']=='passed' and a['processed_npz_sha256']==sha(npz);files.append(audit)
        subjects.append({'subject':s,'role':'train' if s<=10 else 'development_validation','windows':m['candidate_windows'],'excluded_windows':0,'activity_counts':m['activity_counts'],
                         'files':[{'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]})
    n=sum(r['windows'] for r in subjects if r['subject']<=10);nv=sum(r['windows'] for r in subjects if r['subject']>10);steps=math.ceil(n/128)
    catalog={'scope':'ALL_PREDECLARED_DEVELOPMENT_SUBJECTS_NO_TEST','subjects':subjects,'train_windows':n,'validation_windows':nv,'test_accessed':False}
    write(V/'data_catalog.json',catalog)
    old=ROOT/'candidate_versions/v0.1.1_ci01_2ee6e442e02d';cfg=json.loads((old/'development_protocol.json').read_text())
    import shutil
    shutil.copy2(old/'CANDIDATE_V0.md',V/'CANDIDATE_V0.md')
    shutil.copy2(__file__,V/'configuration_builder.py')
    cfg['model_source_bindings']=[{'path':str((ROOT/r).resolve()),'sha256':sha(ROOT/r)} for r in ['models/qppg_fixed.py','third_party/q-ppg-main/precision_search/model/TimePPG_float.py']]
    costs=[json.loads(p.read_text()) for p in sorted((V/'executor').glob('synthetic_*/checks_and_cost.json'))];assert len(costs)>=2 and all(c['status']=='passed' for c in costs)
    step=max(c['train_step_seconds_p95'] for c in costs);validation=max(c['validation_batch_seconds_p95'] for c in costs)
    estimated=200*(steps*step+math.ceil(nv/128)*validation);timeout=max(3600,math.ceil((estimated*3+600)/60)*60);assert timeout<=21600
    array_bytes=(n+nv)*(4*256+256+3+1)*4+(n+nv)*32
    cfg.update(stage='ROUND2_EXECUTION_CANDIDATE_PENDING_REVIEW_NO_REAL_TRAINING',train_subjects=list(range(1,11)),validation_subjects=[11,12],train_windows=n,validation_windows=nv,
               max_epochs=200,updates_per_full_training_epoch=steps,validation_interval_updates=steps,max_updates_per_method_seed=200*steps,
               selection_metric='validation_macro_logcosh',concatenation_order='subject ascending then start ascending',training_weight='uniform windows',tail_policy='128+1 ->127+2; retain every window once',
               paired_aggregation='subject x seed deltas -> each subject mean across 3 seeds -> equal mean across two subjects; retain six deltas and per-seed macro/pooled',
               operations={'device':'cpu','cpu_threads':2,'prepare_timeout_seconds':1800,'per_arm_timeout_seconds':timeout,'minimum_available_ram_bytes':max(int(2.5*1024**3),array_bytes*3+512*1024**2),
                           'minimum_free_disk_bytes':10*1024**3,'maximum_whole_bundle_attempts':2,'automatic_retry':False,'resume_supported':False},
               resource_estimate={'scope':'Synthetic only; not measured real development errors or memory peak','estimated_prepared_array_bytes':array_bytes,'step_seconds_max_p95':step,
                                  'validation_batch_seconds_max_p95':validation,'estimated_200_epoch_seconds':estimated,'watchdog_formula':'ceil_minutes(3*estimate+600), floor3600 cap21600'})
    cfg['status']='awaiting independent review; no real round2 cache or training';write(V/'protocol.json',cfg)
    names=[p for p in V.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    write(V/'EXECUTION_FREEZE.json',{'scope':'Round2 execution candidate and synthetic evidence only; independent review required','data_catalog_sha256':sha(V/'data_catalog.json'),
          'files':[{'path':str(p.relative_to(V)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(names)]})
    print(json.dumps({'train_windows':n,'validation_windows':nv,'steps_per_epoch':steps,'max_updates':200*steps,'per_arm_timeout_seconds':timeout,'freeze_sha256':sha(V/'EXECUTION_FREEZE.json')},indent=2))
if __name__=='__main__':main()
