"""Prepare only development subjects; fail on mismatched window/label counts.

Matches official Q-PPG active loader BVP[::2], unchanged ACC, 8s/2s,
PPG-first order and no fitted normalization. Raw originals remain read-only.
"""
from pathlib import Path
import json, pickle, zipfile, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def prepare():
    cfg = json.loads((ROOT/'configs/development.json').read_text())
    groups = [set(cfg[k]) for k in ('train_subjects','validation_subjects','sealed_test_subjects')]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i+1,3))
    archive = ROOT/'data/raw/ppg_dalia_data.zip'
    if not archive.exists():
        raise FileNotFoundError('Complete official archive missing; run scripts/download_dalia.py first. Partial ZIP is never used.')
    out = ROOT/'data/processed'
    out.mkdir(exist_ok=True)
    manifest = []
    with zipfile.ZipFile(archive) as z:
        for split in ('train_subjects','validation_subjects'):
            for s in cfg[split]:
                matches = [n for n in z.namelist() if n.replace('\\','/').endswith(f'/S{s}/S{s}.pkl')]
                assert len(matches) == 1, (s, matches)
                # Reading to EOF also verifies this archive member's CRC.
                blob = z.read(matches[0])
                subject = pickle.loads(blob, encoding='latin1')
                assert int(subject['subject']) == s, ('Subject identity mismatch',s,subject['subject'])
                wrist = subject['signal']['wrist']
                ppg = np.asarray(wrist['BVP'][::2],dtype=np.float32)
                acc = np.asarray(wrist['ACC'],dtype=np.float32)
                y = np.asarray(subject['label'],dtype=np.float32).reshape(-1)
                activity = np.asarray(subject['activity']).reshape(-1)
                assert np.isfinite(activity).all() and np.isin(activity,np.arange(9)).all(), 'Invalid activity labels: stop before categorical conversion'
                assert ppg.shape == (len(acc),1) and acc.shape[1] == 3
                starts = np.arange(0,len(acc)-255,64)
                assert len(starts) == len(y), ('Label/window mismatch: do not silently trim',s,len(starts),len(y))
                sig = np.concatenate([ppg,acc],axis=1)
                x = np.stack([sig[k:k+256].T for k in starts])
                acts = []
                for k in starts:
                    a = activity[k//8:k//8+32]
                    assert len(a) == 32
                    u,c = np.unique(a,return_counts=True)
                    acts.append(u[c.argmax()])
                assert np.isfinite(x).all() and np.isfinite(y).all(), 'Invalid windows need shared explicit policy; no automatic dropping'
                acts = np.asarray(acts,dtype=np.int16)
                np.savez(out/f'S{s}.npz',x=x,y=y,activity=acts,start_sample_32hz=starts)
                u,c=np.unique(acts,return_counts=True)
                manifest.append({'subject':s,'split':split,'windows':len(y),'activity_counts':dict(zip(map(str,u),map(int,c))),
                                 'raw_member':matches[0],'raw_sha256':hashlib.sha256(blob).hexdigest(),
                                 'candidate_windows':len(starts),'excluded_windows':0,
                                 'boundary_mapping':[{'window_id':int(j),'interval_seconds':[float(j*2),float(j*2+8)],
                                     'signal_samples_32hz':[int(starts[j]),int(starts[j]+256)],
                                     'activity_samples_4hz':[int(starts[j]//8),int(starts[j]//8+32)],
                                     'reference_label_index':int(j)} for j in sorted(set([0,len(y)//2,len(y)-1]))]})
                print(json.dumps(manifest[-1]),flush=True)
                del subject,blob,x,sig
    (ROOT/'data/manifests/development_windows.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__ == '__main__':
    prepare()
