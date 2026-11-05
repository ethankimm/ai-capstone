import json, sys, re, glob, os
import numpy as np
from scipy import stats
S=sys.argv[1]
rng=np.random.default_rng(0)
A=[json.loads(l) for l in open(glob.glob(S+'/preds/*gsm8k_aug_codi.jsonl')[0])]
B=[json.loads(l) for l in open(glob.glob(S+'/preds/*gsm8k_aug_codi0.jsonl')[0])]
assert [x['id'] for x in A]==[x['id'] for x in B]
def num(x):
    try: return float(str(x).replace(',','').strip())
    except: return None
# gold rationale for intermediates
cache=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--zen-E--GSM8k-Aug/snapshots/*/*'))
print([os.path.basename(c) for c in cache])
gold={}
for c in cache:
    if 'test' in os.path.basename(c):
        raw=open(c).read()
        try:
            d=json.loads(raw)
        except Exception:
            d=[json.loads(l) for l in raw.splitlines() if l.strip()]
        if isinstance(d,dict):
            if 'question' in d:
                keys=list(d); n=len(d['question']); d=[{k:d[k][i] for k in keys} for i in range(n)]
            else:
                d=next(v for v in d.values() if isinstance(v,list))
        for r in d: gold[r['question'].strip()]=r
print('gold rationales',len(gold))
def steps(q):
    r=gold.get(q.strip())
    if not r: return None
    return [float(v.replace(',','')) for v in re.findall(r'=\s*(-?[\d,]*\.?\d+)>>',r['cot'])]
def table(name,rows):
    print('##',name,'overall',round(np.mean([r['correct'] for r in rows]),3))
    print(' steps  n  acc | wrong: |e|<=2 (chance) rel<=10% (chance) | pred = an intermediate of the gold chain (chance)')
    for lo,hi in ((1,1),(2,2),(3,3),(4,4),(5,8)):
        rr=[r for r in rows if lo<=r['level']<=hi]
        acc=np.mean([r['correct'] for r in rr])
        w=[(num(r['predicted_answer']),num(r['gold_answer']),steps(r['question'])) for r in rr if not r['correct']]
        w=[x for x in w if x[0] is not None and x[1] is not None]
        p=np.array([x[0] for x in w]); g=np.array([x[1] for x in w])
        e=np.abs(p-g); rel=e/np.maximum(np.abs(g),1)
        ch=[];chr_=[];chi=[]
        def inter(pp):
            h=n=0
            for v,x in zip(pp,w):
                if x[2] is None: continue
                n+=1; h+= any(abs(v-s)<1e-6 for s in x[2][:-1])
            return h/max(n,1)
        for _ in range(30):
            q=rng.permutation(p); ee=np.abs(q-g); ee=ee[ee>0]
            ch.append(np.mean(ee<=2)); chr_.append(np.mean(ee/np.maximum(np.abs(g[:len(ee)]),1)<=.1)); chi.append(inter(q))
        print(f' {lo}-{hi} {len(rr):4d} {acc:.3f} | {np.mean(e<=2):.3f} ({np.mean(ch):.3f}) {np.mean(rel<=.1):.3f} ({np.mean(chr_):.3f}) | {inter(p):.3f} ({np.mean(chi):.3f})')
table('GSM8K CODI',A); table('GSM8K CODI-0',B)
# what do the items CODI wins look like? what did CODI-0 answer on them
win=[(a,b) for a,b in zip(A,B) if a['correct'] and not b['correct']]
loss=[(a,b) for a,b in zip(A,B) if b['correct'] and not a['correct']]
print('CODI-only',len(win),'CODI-0-only',len(loss))
def desc(pairs,which):
    v=[]
    for a,b in pairs:
        r=b if which=='b' else a
        p=num(r['predicted_answer']); g=num(r['gold_answer'])
        if p is None or g is None: continue
        v.append((abs(p-g),abs(p-g)/max(abs(g),1),steps(r['question']),p))
    e=np.array([x[0] for x in v]); rel=np.array([x[1] for x in v])
    it=np.mean([any(abs(x[3]-s)<1e-6 for s in (x[2] or [])[:-1]) for x in v])
    print(f'  wrong side: |e|<=2 {np.mean(e<=2):.3f} rel<=10% {np.mean(rel<=.1):.3f} median|e| {np.median(e):.1f} pred=intermediate {it:.3f}')
print('CODI-0 errors on items CODI wins:'); desc(win,'b')
print('CODI errors on items CODI-0 wins:'); desc(loss,'a')
