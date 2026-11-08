"""b24, latent passes only. Rows: 0 bot, 1 p0, 2 p1, 3 p2, 4 p3, 5 p4, 6 p5. The <bot> jump is excluded: every model has it, relay included.
Same-parity steps only, because GSM8K latents alternate. Held-out heads: top 20 by the move on one half of items, scored on the other half."""
import sys, numpy as np
d=np.load(sys.argv[1],allow_pickle=True); rng=np.random.default_rng(0)
G=['GSM8K released|J2','GSM8K released|J3','GSM8K released|J4','startlast CODI|L3','startlast CODI|L4','startlast CODI|L5','startlast CODI|L6','lr3 relay|L4','lr3 relay|L5']
def ci(v):
    b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(2000)]; return np.percentile(b,[2.5,97.5])
def held(c,a,b,top=20):
    m=c[:,b]-c[:,a]; n=len(m); p=rng.permutation(n); f,e=p[:n//2],p[n//2:]; o=np.zeros(n)
    for x,y in ((f,e),(e,f)):
        h=np.argsort(m[x].mean(0))[-top:]; o[y]=m[y][:,h].mean(1)
    return o
print('move in COM (0-1 along the chain). all heads mean [CI] | held-out top-20 heads [CI]')
for k in G:
    c=d[k+'::com'].astype(np.float64); print(k,'n',len(c))
    for nm,a,b in (('p1->p3',2,4),('p3->p5',4,6),('p0->p2',1,3),('p2->p4',3,5)):
        v=(c[:,b]-c[:,a]).mean(1); h=held(c,a,b); x=ci(v); y=ci(h)
        print(f'   {nm} {v.mean():+.3f} [{x[0]:+.3f},{x[1]:+.3f}] | {h.mean():+.3f} [{y[0]:+.3f},{y[1]:+.3f}]')
    # how different are passes at all: mean |COM difference| per head, same parity
    print('   mean |diff| per head p1-p3 %.3f  p3-p5 %.3f'%(np.abs(c[:,4]-c[:,2]).mean(),np.abs(c[:,6]-c[:,4]).mean()))
print('attention on step tokens by row, bot p0..p5; alternation = mean(bot,p1,p3,p5)/mean(p0,p2,p4)')
for k in G:
    t=d[k+'::tot']; print(f'{k:22s} '+' '.join(f'{x:.3f}' for x in t)+f'  ratio {t[[0,2,4,6]].mean()/t[[1,3,5]].mean():.2f}')
