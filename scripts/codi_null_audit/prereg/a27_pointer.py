"""Reads b24 output. Usage: a27_pointer.py <npz>. Rows: bot, p0..p5. COM 0 = first step, 1 = last. Move = last row minus first row, per item.
Head selection: pick 20 heads with the largest mean slope on one half of items, score on the other half, swap, average. Bootstrap over items."""
import sys, numpy as np
d=np.load(sys.argv[1],allow_pickle=True); ks=sorted({k.split('::')[0] for k in d.files}); rng=np.random.default_rng(0)
x=np.arange(7.)-3; 
def slope(c,rows=None):
    r=np.arange(7) if rows is None else np.array(rows); xr=r-r.mean(); return (c[:,r]*xr[None,:,None]).sum(1)/(xr**2).sum()   # [n,heads]
def boot(v):
    b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(2000)]; return np.percentile(b,[2.5,97.5])
def cv(c,top=20,rows=None):
    s=slope(c,rows); n=len(s); p=rng.permutation(n); a,b=p[:n//2],p[n//2:]; out=np.zeros(n)
    for f,e in ((a,b),(b,a)):
        h=np.argsort(s[f].mean(0))[-top:]; out[e]=s[e][:,h].mean(1)
    return out
print('group n right | COM rows bot,p0..p5 (all heads) | slope x6, all heads [CI] | held-out top-20 heads slope x6 [CI] | same, rows bot,p1,p3 x(span) | share on set by row')
for k in ks:
    c=d[k+'::com'].astype(np.float64); right=d[k+'::right'].astype(bool); n=len(c)
    if n<20: print(k,'n',n,'skipped'); continue
    for lbl,I in (('all',np.ones(n,bool)),('right',right)):
        if I.sum()<20: continue
        cc=c[I]; s=slope(cc).mean(1)*6; v=cv(cc)*6; w=cv(cc,rows=[0,2,4])*4
        ci=boot(s); cj=boot(v); ck=boot(w)
        print(f'{k:24s} {lbl:5s} n={I.sum():4d} '+' '.join(f'{m:.2f}' for m in cc.mean((0,2)))+f' | {s.mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] | {v.mean():+.3f} [{cj[0]:+.3f},{cj[1]:+.3f}] | {w.mean():+.3f} [{ck[0]:+.3f},{ck[1]:+.3f}]')
    print('    tot '+' '.join(f'{t:.3f}' for t in d[k+'::tot'])); sh=d[k+'::share']
    for r,nm in enumerate(['bot','p0','p1','p2','p3','p4','p5']): print(f'    {nm:3s} '+' '.join(f'{t:.2f}' for t in sh[r]))
