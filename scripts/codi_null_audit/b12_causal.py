"""Causal edits of single rules. Taken-branch constant: answer should shift by the propagated amount. Untaken-branch constant: answer must not move.
x0 shift keeping the path. Branch flip at rule j: does the answer follow the new path or stick to the old answer?"""
import sys, json, time, random; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV=sys.argv[2]
TE=BASE+'/synthetic_data_v2_l1to7w/depth2/test.json'
base=rows(TE,per_level=140,seed=31,levels=[4,5,6,7])
def text(ru):
    k,t,a,b=ru
    if k=='P': return f"if it is even, halve it; otherwise add {a}"
    if k=='T': return f"if it is at least {t}, subtract {b}; otherwise add {a}"
    return f"if it is below {t}, double it; otherwise subtract {b}"
def render(x0,rules):
    return f"Start with {x0}. "+' '.join(f"Rule {i+1}: {text(r)}." for i,r in enumerate(rules))+" What is the final number?"
def ok(x0,rules):
    v,b=walk(x0,rules); return all(0<=x<=99 for x in v),v,b
def bounds(ru):
    k,t,a,b=ru
    if k=='P': return (1,25),None
    if k=='T': return (1,min(20,100-t)),(1,min(20,t))
    return None,(1,t)
rng=random.Random(5)
Q=[]   # dict(kind,item,level,j,d,new,true_delta...)
for r in base:
    x0,rules=parse(r['question']); assert render(x0,rules)==r['question'].strip()
    n=len(rules); v,br=walk(x0,rules); gold=v[-1]
    for j,ru in enumerate(rules):
        k,t,a,b=ru
        taken=None;untaken=None
        if k=='P': (untaken,taken)=('a',None) if br[j] else (None,'a')
        elif k=='T': taken,untaken=('b','a') if br[j] else ('a','b')
        else: (taken,untaken)=(None,'b') if br[j] else ('b',None)
        for kind,which in (('taken',taken),('untaken',untaken)):
            if which is None: continue
            ba,bb=bounds(ru)
            for dl in rng.sample([3,-3],2):
                na,nb=(a+dl,b) if which=='a' else (a,b+dl)
                lo,hi=(ba if which=='a' else bb)
                if not (lo<=(na if which=='a' else nb)<=hi): continue
                nr=list(rules); nr[j]=(k,t,na,nb); g,v2,b2=ok(x0,nr)
                if not g or b2!=br: continue
                if kind=='taken' and v2[-1]==gold: continue
                Q.append(dict(kind=kind,item=r['id'],level=n,j=j+1,d=n-1-j,q=render(x0,nr),new=v2[-1],gold=gold)); break
    for dl in rng.sample([1,-1,2,-2,3,-3],6):
        g,v2,b2=ok(x0+dl,rules) if 0<=x0+dl<=99 else (False,0,0)
        if g and b2==br and v2[-1]!=gold:
            Q.append(dict(kind='x0',item=r['id'],level=n,j=0,d=n,q=render(x0+dl,rules),new=v2[-1],gold=gold)); break
    # branch flips: threshold moved to the other side of the input, margin >= 4
    for j,ru in enumerate(rules):
        k,t,a,b=ru
        if k=='P': continue
        s=v[j]
        for _ in range(40):
            if k=='T':
                nt=rng.randint(20,80)
                if (s>=nt)==br[j] or abs(s-nt)<4 or a>min(20,100-nt) or b>min(20,nt): continue
            else:
                nt=rng.randint(10,50)
                if (s<nt)==br[j] or abs(s-nt)<4 or b>nt: continue
            nr=list(rules); nr[j]=(k,nt,a,b); g,v2,b2=ok(x0,nr)
            if g and abs(v2[-1]-gold)>=10:
                Q.append(dict(kind='flip',item=r['id'],level=n,j=j+1,d=n-1-j,q=render(x0,nr),new=v2[-1],gold=gold)); break
    for _ in range(40):
        nx=rng.randint(0,99); g,v2,b2=ok(nx,rules)
        if g and abs(v2[-1]-gold)>=10 and nx!=x0:
            Q.append(dict(kind='flipx0',item=r['id'],level=n,j=0,d=n,q=render(nx,rules),new=v2[-1],gold=gold)); break
print('items',len(base),'queries',len(Q),{k:sum(q['kind']==k for q in Q) for k in ('taken','untaken','x0','flip','flipx0')},flush=True)
NUM={}
def answers(m,tok,qs,nl):
    if not NUM:
        for n in range(100): NUM[tok.encode(f"The answer is: {n}")[-1]]=n
    dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV); out=[None]*len(qs)
    idx=sorted(range(len(qs)),key=lambda i:len(qs[i]))
    for i in range(0,len(idx),32):
        ii=idx[i:i+32]; bt=[dict(question=qs[j],steps=['x'],answer='0') for j in ii]
        b=encode(bt,tok,m,DEV); lg,_=fast_answer(m,b,nl,dec_ids=dec)
        for j,t in zip(ii,lg.argmax(-1).tolist()): out[j]=NUM.get(t,-999)
        del lg,b
        if DEV=='mps': torch.mps.empty_cache()
    return out
res={}; STORE={}
for name,ck,nl in (('CODI-0 stab',CK['stab0']%17820,0),('CODI stab',CK['stab']%17820,6),('CODI lr3e-3',CK['lr3'],6),('CODI-0 round1',CK['r10'],0)):
    t0=time.time(); m,tok,ta=load(ck,nl,DEV); m.eval()
    pb=dict(zip([r['id'] for r in base],answers(m,tok,[r['question'].strip() for r in base],nl)))
    pn=answers(m,tok,[q['q'] for q in Q],nl); del m
    STORE[name]=dict(base=pb,edits=pn)
    if DEV=='mps': torch.mps.empty_cache()
    R={}
    def sel(kind,f=lambda q:True): return [(q,p) for q,p in zip(Q,pn) if q['kind']==kind and f(q)]
    good=lambda q:abs(pb[q['item']]-q['gold'])<=2     # base answer was close, so the path is roughly right
    # taken constant and x0 shift: response gain
    def gain(pairs):
        pairs=[(q,p) for q,p in pairs if good(q) and abs(p-pb[q['item']])<=25]
        dt=np.array([q['new']-q['gold'] for q,p in pairs],float); dm=np.array([p-pb[q['item']] for q,p in pairs],float)
        return dict(n=len(pairs),gain=round(float((dm*dt).sum()/(dt*dt).sum()),3),exact_shift=round(float(np.mean(dm==dt)),3),right_sign=round(float(np.mean(np.sign(dm)==np.sign(dt))),3))
    for dd,lab in ((0,'last rule'),(1,'1 before last'),(2,'2 before'),(3,'3 before'),(4,'4 before'),(5,'5 before'),(6,'6 before')):
        R[f'taken {lab}']=gain(sel('taken',lambda q:q['d']==dd))
    R['x0 shift, same path']=gain(sel('x0'))
    # untaken constant: answer should not move
    for lab,f in (('last 2 rules',lambda q:q['d']<=1),('middle',lambda q:2<=q['d']<=3),('early (>=4 before last)',lambda q:q['d']>=4)):
        pr=[(q,p) for q,p in sel('untaken',f) if good(q)]
        dm=np.array([abs(p-pb[q['item']]) for q,p in pr],float)
        R[f'untaken {lab}']=dict(n=len(pr),moved=round(float(np.mean(dm>0)),3),moved_ge3=round(float(np.mean(dm>=3)),3))
    # flips
    def flips(pairs):
        pairs=[(q,p) for q,p in pairs if good(q)]
        nw=np.array([abs(p-q['new'])<=2 for q,p in pairs]); od=np.array([abs(p-q['gold'])<=2 for q,p in pairs]); ex=np.array([p==q['new'] for q,p in pairs])
        return dict(n=len(pairs),follow_new=round(float(nw.mean()),3),stick_old=round(float(od.mean()),3),exact_new=round(float(ex.mean()),3))
    for lab,f in (('rule 1',lambda q:q['j']==1),('rule 2',lambda q:q['j']==2),('rule 3',lambda q:q['j']==3),('rule 4',lambda q:q['j']==4),('rule 5+',lambda q:q['j']>=5)):
        R[f'flip {lab}']=flips(sel('flip',f))
    R['new start value']=flips(sel('flipx0'))
    for L in (4,5,6,7):
        e=[abs(pb[r['id']]-int(r['answer'])) for r in base if int(r['level'])==L]
        R[f'base L{L}']=dict(exact=round(float(np.mean(np.array(e)==0)),3),close=round(float(np.mean(np.array(e)<=2)),3))
    res[name]=R; print('##',name,f'{time.time()-t0:.0f}s',flush=True)
    for k,v in R.items(): print('  ',k,v,flush=True)
json.dump(res,open(sys.argv[1]+'/b12_causal.json','w'),indent=1)
json.dump(dict(Q=Q,store=STORE,base=[dict(id=r['id'],level=int(r['level']),question=r['question'],answer=r['answer']) for r in base]),open(sys.argv[1]+'/b12_preds.json','w'))
print('DONE')
