"""Frozen label-independent nominal-index Wrist adapter and manual-R reference."""
from pathlib import Path
import collections,hashlib,json,re
import numpy as np
from scipy.signal import resample_poly

CHANNELS=['wrist_ppg']+['wrist_low_noise_accelerometer_'+a for a in 'xyz']
TIME='sample_times_for_all_signals_apart_from_ecg'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def header(path):
    lines=Path(path).read_text().splitlines();name,nc,fs,ns=lines[0].split()[:4];signals=[]
    for line in lines[1:int(nc)+1]:
        s=line.split();assert s[1]=='16'
        g=re.fullmatch(r'([\d.eE+-]+)\((-?\d+)\)/(.+)',s[2]);assert g
        signals.append({'name':s[-1],'gain':float(g[1]),'baseline':int(g[2]),'units':g[3],'checksum':int(s[6])})
    assert len(set(s['name'] for s in signals))==int(nc) and float(fs)==256
    return {'record':name,'channels':signals,'fs':float(fs),'samples':int(ns)}
def annotations(path,ns):
    """MIT format, including signed PDP-11 SKIP, AUX and metadata pseudo-notes."""
    raw=Path(path).read_bytes();assert len(raw)%2==0;i=0;t=0;events=[];ended=False;special=collections.Counter()
    while i<len(raw):
        word=int.from_bytes(raw[i:i+2],'little');i+=2;code=word>>10;interval=word&1023
        if word==0:ended=True;break
        if code==59:
            assert interval==0 and i+4<=len(raw)
            v=(int.from_bytes(raw[i:i+2],'little')<<16)|int.from_bytes(raw[i+2:i+4],'little');i+=4
            t+=v-(1<<32) if v&(1<<31) else v;special['SKIP']+=1
        elif code in [60,61,62]:
            assert events;events[-1][{60:'num',61:'subtype',62:'channel'}[code]]=interval;special[str(code)]+=1
        elif code==63:
            assert events and i+interval+(interval%2)<=len(raw)
            events[-1]['aux']=raw[i:i+interval].decode('ascii');i+=interval+interval%2;special['AUX']+=1
        else:
            assert code<=49,('Unexpected annotation code',code)
            t+=interval;events.append({'sample':t,'code':code})
    assert ended and not any(raw[i:]),'Missing EOF or nonzero trailing bytes'
    notes=[e for e in events if e['code']==22]
    resolution=[e['aux'] for e in notes if e.get('aux','').startswith('## time resolution:')]
    assert resolution==['## time resolution: 256'],'Annotation timebase must explicitly match 256Hz header'
    unexpected=[e for e in events if e['code'] not in [0,1,22]]
    assert not unexpected,('Uninterpreted reference annotation',unexpected[:3])
    peaks=np.asarray([e['sample'] for e in events if e['code']==1],dtype=np.int64)
    assert len(peaks)>1 and np.all(np.diff(peaks)>0) and peaks[0]>=0 and peaks[-1]<ns
    return peaks,{'symbol_counts':dict(collections.Counter(str(e['code']) for e in events)),'metadata_notes':notes,'special_counts':dict(special),'reference_code':1,'reference_symbol':'N','annotation_fs_hz':256,'first_r_sample':int(peaks[0]),'last_r_sample':int(peaks[-1]),'peak_count':len(peaks)}
def physical(digital,signal):
    valid=digital!=-32768
    x=np.full(digital.shape,np.nan,dtype=np.float64)
    x[valid]=(digital[valid].astype(np.float64)-signal['baseline'])/signal['gain']
    return x,valid
def adapt_record(directory,record):
    directory=Path(directory);h=header(directory/(record+'.hea'));signals=h['channels'];n=h['samples']
    raw=np.fromfile(directory/(record+'.dat'),dtype='<i2').reshape(n,len(signals))
    for i,s in enumerate(signals):assert int(raw[:,i].astype(np.int64).sum())%65536==s['checksum']%65536
    indices=[next(i for i,s in enumerate(signals) if s['name']==name) for name in CHANNELS]
    assert signals[indices[0]]['units']=='mV' and all(signals[i]['units']=='ms^-2' for i in indices[1:])
    values=[];masks=[];invalid={}
    for i,s in enumerate(signals):
        mask=raw[:,i]!=-32768;where=np.flatnonzero(mask)
        end=int(where[-1]+1) if len(where) else 0
        invalid[s['name']]={'invalid_count':int((~mask).sum()),'trailing_invalid_count':n-end,'internal_invalid_count':int((~mask[:end]).sum())}
    for channel,i in enumerate(indices):
        v,mask=physical(raw[:,i],signals[i]);values.append(v/(9.80665 if channel else 1));masks.append(mask)
    values=np.stack(values);valid=np.all(np.stack(masks),axis=0)&np.isfinite(values).all(axis=0)
    ti=next(i for i,s in enumerate(signals) if s['name']==TIME);tv,tm=physical(raw[:,ti],signals[ti]);tt=tv[tm]
    delta=np.diff(tt);wraps=delta < -30;unwrapped=tt+np.r_[0,np.cumsum(wraps)]*60;dt=np.diff(unwrapped)
    timing={'valid_timestamp_samples':len(tt),'wrap_count':int(wraps.sum()),'nonpositive_unwrapped_deltas':int((dt<=0).sum()),'timestamp_min':float(tt.min()),'timestamp_max':float(tt.max()),'first_seconds_field':float(tt[0]),'last_seconds_field':float(tt[-1]),'elapsed_timestamp_seconds':float(unwrapped[-1]-unwrapped[0]),'effective_mean_hz':float((len(tt)-1)/(unwrapped[-1]-unwrapped[0])),'unwrapped_minus_nominal_duration_seconds':float(unwrapped[-1]-unwrapped[0]-(len(tt)-1)/256),'usage':'Diagnostics only; nominal index used without offset search or subtracting timestamp origin from ECG'}
    peaks,annotation_report=annotations(directory/(record+'.atr'),n)
    rows=[];windows=[];eligible_indices=[]
    for k,start in enumerate(range(0,n-2048+1,512)):
        stop=start+2048;r=peaks[(peaks>=start)&(peaks<stop)];input_ok=bool(valid[start:stop].all());ref_ok=len(r)>=2
        hr=60*(len(r)-1)*256/(int(r[-1])-int(r[0])) if ref_ok else None
        reasons=[]
        if not input_ok:reasons.append('invalid_PPG_or_selected_ACC')
        if not ref_ok:reasons.append('fewer_than_two_manual_R_peaks')
        row={'record':record,'subject':int(record.split('_')[0][1:]),'activity':record.split('_',1)[1],'window_id':k,'start_sample_256hz':start,'stop_sample_256hz':stop,'start_seconds':start/256,'end_seconds':stop/256,'input_available':input_ok,'reference_available':ref_ok,'eligible':input_ok and ref_ok,'reference_bpm':hr,'r_peak_count':len(r),'first_r_sample':int(r[0]) if len(r) else None,'last_r_sample':int(r[-1]) if len(r) else None,'reason':';'.join(reasons) or 'eligible'}
        if row['eligible']:
            window=resample_poly(values[:,start:stop],1,8,axis=1,window=('kaiser',5.0),padtype='constant').astype(np.float32)
            assert window.shape==(4,256) and np.isfinite(window).all()
            eligible_indices.append(k);windows.append(window)
        rows.append(row)
    result={'record':record,'subject':int(record.split('_')[0][1:]),'activity':record.split('_',1)[1],'samples':n,'nominal_duration_seconds':n/256,'selected_channel_indices_zero_based':indices,'selected_channels':[signals[i] for i in indices],'invalid_by_channel':invalid,'timing':timing,'annotation':annotation_report,'candidate_windows':len(rows),'input_available':sum(r['input_available'] for r in rows),'reference_available':sum(r['reference_available'] for r in rows),'eligible_windows':len(windows),'exclusion_counts':dict(collections.Counter(r['reason'] for r in rows)),'input_physical_ranges':{name:{'min':float(np.nanmin(values[j])),'max':float(np.nanmax(values[j])),'unit':'mV' if j==0 else 'g'} for j,name in enumerate(CHANNELS)}}
    example=None
    if windows:
        k=eligible_indices[0];start=rows[k]['start_sample_256hz'];points=[start,start+1,start+8,start+2047]
        example={'record':record,'window':rows[k],'physical_hand_mapping':[{'source_sample':t,'channel':signals[i]['name'],'digital':int(raw[t,i]),'baseline':signals[i]['baseline'],'gain':signals[i]['gain'],'unit_divisor':9.80665 if j else 1,'physical_value':float(values[j,t])} for t in points for j,i in enumerate(indices)],'output_32hz_first_last':[windows[0][:,0].tolist(),windows[0][:,-1].tolist()],'resampling_note':'Output centres spaced every 8 original indices; Kaiser FIR includes neighbouring samples strictly within this 2048-point window, zero outside'}
    return rows,np.stack(windows) if windows else np.empty((0,4,256),np.float32),np.asarray(eligible_indices,dtype=np.int64),result,example
