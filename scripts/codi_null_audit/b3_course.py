"""Per-epoch time course on VALID: exact vs within-2 by level for both arms; all-latent mean ablation for the CODI arm.
Also train-vs-valid at the final checkpoint (memorisation pressure). Teacher-forced answer token at ':' (= generation within 0.004)."""
import sys, json, time; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
DEV=sys.argv[2]
torch.manual_seed(0)
D=BASE+'/synthetic_data_v2_l1to7w/depth2/'
va=json.load(open(D+'valid.json'))
tr=rows(D+'train.json',per_level=400,seed=7)
EPOCHS=list(range(1,13))
def batches(rs,bs=128):
    rs=sorted(rs,key=lambda r:len(r['question']))
    for i in range(0,len(rs),bs): yield rs[i:i+bs]
NUM={}
def numtok(tok):
    if NUM: return NUM
    for n in range(0,100):
        ids=tok.encode(f"The answer is: {n}"); pre=tok.encode('The answer is:')
        assert len(ids)==len(pre)+1
        NUM[ids[-1]]=n
    return NUM
def run(m,tok,rs,nl,mu=None,collect=False):
    nt=numtok(tok); out={}; Z=[]
    dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV)
    for bt in batches(rs,32):
        b=encode(bt,tok,m,DEV)
        lg,z=fast_answer(m,b,nl,mu=mu,dec_ids=dec)
        for r,i in zip(bt,lg.argmax(-1).tolist()): out[r['id']]=nt.get(i,-999)
        if collect: Z.append(z.float().cpu())
        del lg,z,b
        if DEV=='mps': torch.mps.empty_cache()
    return out,(torch.cat(Z) if collect else None)
def score(rs,pred):
    t={}
    for L in sorted({int(r['level']) for r in rs}):
        e=np.array([abs(pred[r['id']]-int(r['answer'])) for r in rs if int(r['level'])==L])
        t[L]=(float(np.mean(e==0)),float(np.mean(e<=2)),float(np.mean(e<=5)))
    return t
res={}
for arm,nl in (('stab',6),('stab0',0)):
    for ep in EPOCHS:
        t0=time.time()
        m,tok,ta=load(CK[arm]%(1485*ep),nl,DEV); m.eval()
        p,Z=run(m,tok,va,nl,collect=bool(nl))
        r={'clean':score(va,p)}
        if nl:
            mu=Z.mean(0).to(DEV)
            pa,_=run(m,tok,va,nl,mu=mu)
            r['ablated']=score(va,pa)
        if ep==12:
            pt,_=run(m,tok,tr,nl); r['train']=score(tr,pt)
        res[f'{arm}_{ep}']=r
        c=r['clean']; mean=lambda t,i:np.mean([t[L][i] for L in t])
        msg=f"{arm} e{ep}: exact {mean(c,0):.3f} within2 {mean(c,1):.3f} | L5 {c[5][0]:.3f}/{c[5][1]:.3f} L7 {c[7][0]:.3f}/{c[7][1]:.3f}"
        if nl: msg+=f" | ablated exact {mean(r['ablated'],0):.3f}"
        print(msg,f'({time.time()-t0:.0f}s)',flush=True)
        del m
        if DEV=='mps': torch.mps.empty_cache()
        json.dump(res,open(sys.argv[1]+'/b3_course.json','w'))
print('DONE')
