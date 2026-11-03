"""Download WikiText-103 (raw) and tokenize it with the GPT-2 BPE vocabulary.

The archived panel uses n_token = 50264 (GPT-2 BPE), so the controlled runs use the same tokenizer
(tiktoken "gpt2", 50257 tokens). Source: the Hugging Face copy of WikiText-103 raw v1
(Salesforce/wikitext). Each file is checked against its published SHA-256.

Output (git-ignored, about 0.5 GB): controlled/data/{train,valid,test}.npy, uint16 token ids.

Run from project root (download time depends on the network; tokenizing takes a few minutes):
  .venv/Scripts/python.exe controlled/prepare_data.py
"""
from pathlib import Path
import hashlib
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import tiktoken

BASE = 'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/'
FILES = {
    'train': [('train-00000-of-00002.parquet', '74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212'),
              ('train-00001-of-00002.parquet', 'ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720')],
    'valid': [('validation-00000-of-00001.parquet', '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c')],
    'test': [('test-00000-of-00001.parquet', '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91')],
}
OUT = Path(__file__).resolve().parent / 'data'


def fetch(name, sha):
    """Download one parquet file into data/raw unless a verified copy exists."""
    path = OUT / 'raw' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(20):
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            break
        have = path.stat().st_size if path.exists() else 0
        print(f'downloading {name} from byte {have:,} (attempt {attempt + 1})')
        try:  # resume with an HTTP range request; connections can drop mid-file
            req = urllib.request.Request(BASE + name, headers={'Range': f'bytes={have}-'})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, 'ab') as f:
                if have and r.status != 206:
                    f.truncate(0)
                while buf := r.read(1 << 20):
                    f.write(buf)
        except OSError as e:
            print(f'  interrupted: {e}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        raise RuntimeError(f'{name}: SHA-256 mismatch ({digest})')
    return path


def main():
    enc = tiktoken.get_encoding('gpt2')
    for split, files in FILES.items():
        text = ''.join(t for name, sha in files for t in pq.read_table(fetch(name, sha)).column('text').to_pylist())
        ids = np.array(enc.encode_ordinary(text), dtype=np.uint16)
        np.save(OUT / f'{split}.npy', ids)
        print(f'{split}: {len(ids):,} tokens')


if __name__ == '__main__':
    main()
