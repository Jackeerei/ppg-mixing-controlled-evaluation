from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
manifest=json.loads((a.root/'SHA256_MANIFEST.json').read_text())
for row in manifest['files']:
 path=a.root/row['path'];assert path.resolve().is_relative_to(a.root.resolve())
 with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256'],row['path']
 assert path.stat().st_size==row['bytes']
print('Verified',len(manifest['files']),'files')
