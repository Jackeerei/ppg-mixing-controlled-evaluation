"""Recover a complete, CRC-verified ZIP member from an interrupted download.

The partial archive is never called complete. Only known local headers with
explicit sizes, no descriptors/encryption/Zip64, and complete deflate streams
are supported. Original bytes remain read-only. Default is training subject S10.
Only predeclared development training/validation subjects are recoverable.
"""
from pathlib import Path
import argparse, hashlib, json, struct, zlib

ROOT = Path(__file__).resolve().parents[1]

def recover(subject=10):
    cfg=json.loads((ROOT/'configs/development.json').read_text())
    assert subject in cfg['train_subjects']+cfg['validation_subjects'] and subject not in cfg['sealed_test_subjects'], 'Recovery restricted to predefined development groups'
    src=ROOT/'data/raw/ppg_dalia_data.zip.part'
    size=src.stat().st_size
    target=f'PPG_FieldStudy/S{subject}/S{subject}.pkl'
    dest=ROOT/f'data/raw/dalia_verified_members/S{subject}/S{subject}.pkl'
    report_path=ROOT/f'data/manifests/S{subject}_recovery.json'
    with src.open('rb') as f:
        while f.tell()+30<=size:
            offset=f.tell()
            h=f.read(30)
            assert h[:4]==b'PK\x03\x04', ('Unsupported ZIP structure',offset)
            _,version,flags,method,_,_,crc,compressed,uncompressed,n,e=struct.unpack('<4s5H3I2H',h)
            name=f.read(n).decode('utf-8')
            f.read(e)
            assert not flags & 9 and max(compressed,uncompressed)<0xffffffff, 'Unsupported descriptor/encryption/Zip64'
            start=f.tell()
            if start+compressed>size:
                raise EOFError(f'Incomplete member {name}; target not recoverable')
            if name!=target:
                f.seek(compressed,1)
                continue
            if dest.exists():
                old=json.loads(report_path.read_text())
                actual=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()
                assert old['sha256']==actual and dest.stat().st_size==uncompressed
                print(json.dumps({'status':'already_verified','path':str(dest)}))
                return
            assert method==8 and uncompressed<2*1024**3
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_suffix('.pkl.recovering')
            dec=zlib.decompressobj(-15)
            remaining=compressed; out_size=0; actual_crc=0
            out_hash=hashlib.sha256(); compressed_hash=hashlib.sha256()
            with tmp.open('xb') as out:
                while remaining:
                    block=f.read(min(64*1024,remaining))
                    assert block
                    remaining-=len(block); compressed_hash.update(block)
                    decoded=dec.decompress(block)
                    out_size+=len(decoded); actual_crc=zlib.crc32(decoded,actual_crc); out_hash.update(decoded)
                    assert out_size<=uncompressed
                    out.write(decoded)
                tail=dec.flush(); out.write(tail); out_size+=len(tail)
                actual_crc=zlib.crc32(tail,actual_crc); out_hash.update(tail)
            assert dec.eof and not dec.unused_data and not dec.unconsumed_tail
            assert out_size==uncompressed and actual_crc & 0xffffffff==crc, 'Size/CRC mismatch; do not import'
            tmp.rename(dest)
            report={'subject':subject,'member':target,'local_header_offset':offset,'compressed_data_offset':start,
                'compressed_bytes':compressed,'uncompressed_bytes':uncompressed,'crc32':f'{crc:08x}',
                'compressed_sha256':compressed_hash.hexdigest(),'sha256':out_hash.hexdigest(),
                'member_complete':True,'deflate_eof':True,'crc_verified':True,'whole_archive_complete':False,
                'origin':'https://archive.ics.uci.edu/ml/machine-learning-databases/00495/data.zip',
                'source_partial_bytes':size,'output':str(dest),
                'scope':'Development member recovery. Local ZIP CRC establishes member integrity, not independent publisher SHA verification.'}
            report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report,indent=2))
            return
    raise FileNotFoundError(target)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--subject',type=int,default=10)
    recover(parser.parse_args().subject)
