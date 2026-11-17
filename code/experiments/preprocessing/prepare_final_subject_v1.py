"""Import a CRC-verified complete member only, without using an incomplete ZIP.

Authorized S13-S15 final-data preparation only; no model predictions.
Identical reviewed Q-PPG active-loader numeric mapping is retained.
"""
from pathlib import Path
import argparse, json, pickle, hashlib, time, gc
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]

def prepare(s=10):
    cfg=json.loads((ROOT/'configs/development.json').read_text())
    assert s in {13,14,15} and s in cfg['sealed_test_subjects']
    assert not (ROOT/f'data/processed/S{s}.npz').exists() and not (ROOT/f'data/manifests/S{s}_windows.json').exists()
    split='final_test_preparation_only'
    evidence=json.loads((ROOT/f'data/manifests/S{s}_recovery.json').read_text())
    assert evidence['member_complete'] and evidence['crc_verified'] and evidence['deflate_eof']
    source=Path(evidence['output'])
    with source.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert digest==evidence['sha256']
    assert psutil.virtual_memory().available>2.5*1024**3, 'Insufficient free RAM for this real subject load; retry after resource check'
    started=time.monotonic()
    print(json.dumps({'stage':'loading_verified_member','subject':s,'available_bytes':psutil.virtual_memory().available}),flush=True)
    with source.open('rb') as f: d=pickle.load(f,encoding='latin1')
    ident=str(d['subject'])
    assert ident in (str(s),f'S{s}'), ('Subject identity mismatch',ident)
    wrist=d['signal']['wrist']
    raw_shapes={k:list(v.shape) for k,v in wrist.items()}
    bvp=np.asarray(wrist['BVP'][::2],dtype=np.float32).copy()
    acc=np.asarray(wrist['ACC'],dtype=np.float32).copy()
    y=np.asarray(d['label'],dtype=np.float32).reshape(-1).copy()
    activity=np.asarray(d['activity']).reshape(-1).copy()
    del d,wrist
    gc.collect()
    assert bvp.shape==(len(acc),1) and acc.shape[1]==3
    starts=np.arange(0,len(acc)-255,64)
    assert len(starts)==len(y)>0, ('Window/label mismatch; no trimming',len(starts),len(y))
    assert np.isfinite(activity).all() and np.isin(activity,np.arange(9)).all()
    sig=np.concatenate([bvp,acc],axis=1)
    x=np.stack([sig[k:k+256].T for k in starts])
    assert np.isfinite(x).all() and np.isfinite(y).all()
    acts=[]; mixed=0
    for k in starts:
        a=activity[k//8:k//8+32]
        assert len(a)==32
        u,c=np.unique(a,return_counts=True)
        acts.append(u[c.argmax()]); mixed+=int(len(u)>1)
    acts=np.asarray(acts,dtype=np.int16)
    out=ROOT/'data/processed'; out.mkdir(exist_ok=True)
    np.savez(out/f'S{s}.npz',x=x,y=y,activity=acts,start_sample_32hz=starts)
    u,c=np.unique(acts,return_counts=True)
    mapping=[{'window_id':int(j),'interval_seconds':[float(j*2),float(j*2+8)],'signal_samples_32hz':[int(starts[j]),int(starts[j]+256)],
              'activity_samples_4hz':[int(starts[j]//8),int(starts[j]//8+32)],'reference_label_index':int(j)} for j in sorted({0,len(y)//2,len(y)-1})]
    report={'subject':s,'split':split,'source_sha256':digest,'source_member':evidence['member'],
        'source_archive_complete':False,'raw_wrist_shapes':raw_shapes,'input_shape':list(x.shape),'target_shape':list(y.shape),
        'sample_rate_hz':32,'bvp_downsampling':'[::2], matches official active Q-PPG loader','normalization':'none',
        'candidate_windows':len(starts),'excluded_windows':0,'activity_counts':dict(zip(map(str,u),map(int,c))),
        'mixed_activity_windows':mixed,'boundary_mapping':mapping,'preparation_seconds':time.monotonic()-started,
        'test_accessed':True,'final_predictions_computed':False,'training_labels_only':s in cfg['train_subjects']}
    (ROOT/f'data/manifests/S{s}_windows.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',type=int,default=10)
    prepare(p.parse_args().subject)
