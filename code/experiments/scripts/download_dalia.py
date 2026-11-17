"""Stream the official archive, keeping partial bytes and structured failure logs."""
from pathlib import Path
from datetime import datetime, timezone
import json, time, urllib.request, hashlib, zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00495/data.zip'
PART = ROOT / 'data/raw/ppg_dalia_data.zip.part'
FINAL = PART.with_suffix('')
LOG = ROOT / 'logs/dalia_download.jsonl'

def event(**kwargs):
    row = {'time_utc': datetime.now(timezone.utc).isoformat(), **kwargs}
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)

if __name__ == '__main__':
    try:
        if FINAL.exists():
            event(status='already_present', path=str(FINAL))
            raise SystemExit(0)
        if PART.exists():
            raise RuntimeError('Partial archive exists; inspect log before retrying. No overwrite.')
        event(status='starting', url=URL)
        start = time.monotonic()
        with urllib.request.urlopen(URL, timeout=45) as r, PART.open('xb') as f:
            event(status='connected', http_status=r.status, content_length=r.headers.get('Content-Length'))
            size = 0
            previous = 0
            sha = hashlib.sha256()
            while block := r.read(1024 * 1024):
                f.write(block)
                sha.update(block)
                size += len(block)
                if size - previous >= 64 * 1024**2:
                    event(status='progress', bytes=size, elapsed_seconds=round(time.monotonic()-start, 1))
                    previous = size
        with zipfile.ZipFile(PART) as z:
            members = z.namelist()
        PART.rename(FINAL)
        event(status='downloaded_zip_directory_readable', bytes=size, sha256=sha.hexdigest(), members=members,
              note='CRC and subject checks remain required before training')
    except Exception as e:
        event(status='failed', error=repr(e), partial_bytes=PART.stat().st_size if PART.exists() else 0)
        raise
