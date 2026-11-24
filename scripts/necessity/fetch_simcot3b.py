"""t50: fetch the inference weights of the released internlm/SIM_COT-LLaMA3-CODI-3B without its training-only step decoder.

The release is 3 shards, 15.3 GB: the CODI student (codi.*: bf16 base, fp32 LoRA, 128,259-row embeddings and lm_head), the projection
(prj.*, fp32) and SIM-CoT's step decoder (decoder.*, a second full Llama-3.2-3B, used only in training). This script reads each shard's
safetensors header with HTTP range requests and copies only the codi.* and prj.* tensors' bytes, unchanged, into one local safetensors
file (~8 GB): BASE/grid_checkpoints/simcot_llama3b_codi/model.safetensors. The disk had 23 GiB free, too little for the full release.
Resumable: completed byte ranges are recorded in progress.json next to the .part file.

  ../codi/.venv/bin/python scripts/necessity/fetch_simcot3b.py
"""
import json
import struct
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url

sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE = Path(__file__).resolve().parents[3]
REPO_ID = 'internlm/SIM_COT-LLaMA3-CODI-3B'
REV = HfApi().model_info(REPO_ID).sha  # pinned in provenance.json
SHARDS = [f'model-0000{i}-of-00003.safetensors' for i in (1, 2, 3)]
KEEP = ('codi.', 'prj.')
OUT = BASE / 'grid_checkpoints/simcot_llama3b_codi'; OUT.mkdir(parents=True, exist_ok=True)
PART, PROG, FINAL = OUT / 'model.safetensors.part', OUT / 'progress.json', OUT / 'model.safetensors'
S = requests.Session()


def get_range(url, a, b):  # bytes [a, b)
    r = S.get(url, headers={'Range': f'bytes={a}-{b - 1}'}, stream=True, timeout=60, allow_redirects=True)
    r.raise_for_status(); assert r.status_code == 206, r.status_code
    return r


# 1. headers
plan = []  # (url, src_abs_start, src_abs_end, key, info)
for sh in SHARDS:
    url = hf_hub_url(REPO_ID, sh, revision=REV)
    n = struct.unpack('<Q', get_range(url, 0, 8).content)[0]
    hdr = json.loads(get_range(url, 8, 8 + n).content)
    hdr.pop('__metadata__', None)
    for k, v in sorted(hdr.items(), key=lambda kv: kv[1]['data_offsets'][0]):
        if k.startswith(KEEP):
            a, b = v['data_offsets']
            plan.append((url, 8 + n + a, 8 + n + b, k, dict(dtype=v['dtype'], shape=v['shape'])))

# 2. output header with new contiguous offsets
out_hdr, off = {}, 0
for url, a, b, k, info in plan:
    out_hdr[k] = dict(**info, data_offsets=[off, off + (b - a)]); off += b - a
out_hdr['__metadata__'] = dict(format='pt', source=f'{REPO_ID}@{REV}', kept='codi.*, prj.*')
hb = json.dumps(out_hdr, separators=(',', ':')).encode()
hb += b' ' * (-len(hb) % 8)
data0 = 8 + len(hb); total = data0 + off
print(f'{len(plan)} tensors, {off / 1e9:.2f} GB of data', flush=True)

# 3. merge adjacent source ranges into chunks
chunks = []  # (url, src_a, src_b, dst_a)
dst = data0
for url, a, b, k, info in plan:
    if chunks and chunks[-1][0] == url and chunks[-1][2] == a:
        chunks[-1][2] = b
    else:
        chunks.append([url, a, b, dst])
    dst += b - a
print(f'{len(chunks)} byte ranges', flush=True)

done = set(json.load(open(PROG))['done']) if PROG.exists() and PART.exists() else set()
if not PART.exists():
    with open(PART, 'wb') as f:
        f.write(struct.pack('<Q', len(hb))); f.write(hb); f.truncate(total)
t0, got = time.time(), 0
with open(PART, 'r+b') as f:
    for ci, (url, a, b, d) in enumerate(chunks):
        if ci in done: continue
        f.seek(d); w = 0
        r = get_range(url, a, b)
        for buf in r.iter_content(1 << 22):
            f.write(buf); w += len(buf); got += len(buf)
            if got % (1 << 29) < (1 << 22):
                el = time.time() - t0
                print(time.strftime('%H:%M'), f'{got / 1e9:.1f} GB, {got / el / 1e6:.0f} MB/s', flush=True)
        assert w == b - a, (ci, w, b - a)
        f.flush(); done.add(ci); json.dump(dict(done=sorted(done)), open(PROG, 'w'))

# 4. check and finalise
from safetensors import safe_open  # noqa: E402
with safe_open(str(PART), 'pt') as g:
    ks = list(g.keys()); assert len(ks) == len(plan), (len(ks), len(plan))
    for k in ('prj.ln.weight', 'codi.base_model.model.model.norm.weight'):
        t = g.get_tensor(k); print(k, t.dtype, tuple(t.shape), f'mean {t.float().mean():.4f}', flush=True)
PART.rename(FINAL); PROG.unlink()
json.dump(dict(repo=REPO_ID, revision=REV, shards=SHARDS, kept=list(KEEP), n_tensors=len(plan), bytes=total,
               note='byte-identical copies of the codi.* and prj.* tensors; decoder.* (training-only step decoder) not fetched'),
          open(OUT / 'provenance.json', 'w'), indent=1)
print(f'wrote {FINAL} ({total / 1e9:.2f} GB) in {time.time() - t0:.0f}s', flush=True)
