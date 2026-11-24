"""t47 gate: GSM8K-Aug test accuracy of the released zen-E/CODI-llama3.2-1b-Instruct through adapters.GSM8KCODILlama (greedy, 6 latents).
The CODI paper reports 55.6% for CODI on LLaMA-3.2-1B; the hop count (hops.py gsm8k_llama) runs only if this gate holds.
Also writes per-item predictions for the depth analyses.

  ../codi/.venv/bin/python scripts/necessity/gsm8k_llama_acc.py <device> <out json>
  env N (default all 1,319), DTYPE fp32 (default) | bf16
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters  # noqa: E402
from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import is_correct  # noqa: E402

DEV, OUT = sys.argv[1:3]
N = int(os.environ.get('N', '0')) or None
DT = dict(fp32=torch.float32, bf16=torch.bfloat16)[os.environ.get('DTYPE', 'fp32')]

t0 = time.time()
REPO_ID = os.environ.get('CKPT_REPO', 'zen-E/CODI-llama3.2-1b-Instruct')  # t49: internlm/SIM_COT-LLaMA3-CODI-1B
if os.environ.get('MODEL'):  # t50: MODEL=gsm8k_simllama3b (CKPT_REPO names the release; the adapter sets dtype and paths)
    ad = adapters.LLAMA_GSM8K[os.environ['MODEL']][1](DEV); DT = next(ad.m.codi.parameters()).dtype
else:
    ad = adapters.GSM8KCODILlama(DEV, ckpt=adapters._hf_snapshot(REPO_ID), dtype=DT)
lr = ad.load_result
print(f'loaded in {time.time() - t0:.0f}s; missing keys {len(lr.missing_keys)}, unexpected keys {len(lr.unexpected_keys)}', flush=True)
print('  missing (first 5):', lr.missing_keys[:5], '\n  unexpected (first 5):', lr.unexpected_keys[:5], flush=True)

te = load_gsm8k_aug(split='test', n=None, seed=None)
if N: te = te[:N]
items, right, t1 = [], 0, time.time()
with torch.no_grad():
    for j, e in enumerate(te):
        ctx, past, z = ad.begin([dict(question=e.question)])
        for s in range(ad.n_latents):
            past, z = ad.latent(ctx, past, z, s)
        (pred,) = ad.finish(ctx, past)
        ok = bool(pred is not None and pred != '' and is_correct(str(pred), str(e.answer)))
        right += ok
        items.append(dict(idx=e.idx, question=e.question, gold=e.answer, pred=pred, right=ok))
        if j % 50 == 0 or j == len(te) - 1:
            el = time.time() - t1
            print(time.strftime('%H:%M'), f'{j + 1}/{len(te)} acc {right / (j + 1):.3f} | {el / (j + 1):.2f}s/item', flush=True)
        if DEV == 'mps' and j % 100 == 0: torch.mps.empty_cache()

res = dict(model=REPO_ID, dtype=str(DT), n=len(te), acc=right / len(te), published_acc=float(os.environ.get('PUBLISHED', '0.556')),
           missing_keys=len(lr.missing_keys), unexpected_keys=len(lr.unexpected_keys), seconds=time.time() - t0, items=items)
json.dump(res, open(OUT, 'w'), indent=1)
print(f'accuracy {res["acc"]:.4f} on {len(te)} (published {res["published_acc"]}); wrote {OUT}', flush=True)
