"""Stream-verify an author-linked selective member and create compatible provenance."""
from pathlib import Path
import argparse,hashlib,json,zlib

ROOT=Path(__file__).resolve().parents[1]

def verify(s):
    cfg=json.loads((ROOT/'configs/development.json').read_text())
    groups=[set(cfg[k]) for k in ('train_subjects','validation_subjects','sealed_test_subjects')]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
    assert s in groups[0]|groups[1] and s not in groups[2]
    source=ROOT/f'data/raw/author_members/S{s}.deflate'
    meta=json.loads(source.with_name(f'S{s}_transport.json').read_text(encoding='utf-8-sig'))
    assert meta['subject']==s and meta['member']==f'PPG_FieldStudy/S{s}/S{s}.pkl' and meta['s10_uci_metadata_match']
    assert source.stat().st_size==meta['compressed_bytes']
    final=ROOT/f'data/raw/dalia_verified_members/S{s}/S{s}.pkl'
    assert not final.exists(), 'Existing verified raw member must be checked separately, no overwrite'
    final.parent.mkdir(parents=True,exist_ok=True);temp=final.with_suffix('.pkl.verifying')
    dec=zlib.decompressobj(-15);size=0;crc=0;sha=hashlib.sha256();compressed_sha=hashlib.sha256()
    with source.open('rb') as f,temp.open('xb') as out:
        while block:=f.read(65536):
            compressed_sha.update(block);data=dec.decompress(block);size+=len(data)
            assert size<=meta['uncompressed_bytes']
            crc=zlib.crc32(data,crc);sha.update(data);out.write(data)
        tail=dec.flush();out.write(tail);size+=len(tail);sha.update(tail);crc=zlib.crc32(tail,crc)
    assert dec.eof and not dec.unused_data and not dec.unconsumed_tail
    assert size==meta['uncompressed_bytes'] and f'{crc&0xffffffff:08x}'==meta['crc32']
    temp.rename(final)
    report={**meta,'origin':meta['source'],'sha256':sha.hexdigest(),'compressed_sha256':compressed_sha.hexdigest(),
            'member_complete':True,'deflate_eof':True,'crc_verified':True,'output':str(final),
            'scope':'Author-linked original archive, exact HTTP206 member transport, local-header/central CRC/length match; full archive not downloaded.'}
    path=ROOT/f'data/manifests/S{s}_recovery.json'
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',type=int,default=11);verify(p.parse_args().subject)
