"""(1) Where does the error enter? Query every prefix of each chain and track error persistence.
(2) Do extra latent passes refine near-misses into exact answers?  Uses test items (audit only; nothing is selected on them)."""
import sys, json, time, re; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV=sys.argv[2]
TE=BASE+'/synthetic_data_v2_l1to7w/depth2/test.json'
te=rows(TE,per_level=200,seed=5,levels=[5,6,7])
def prefixes(r):
    x0,rules=parse(r['question']); v,_=walk(x0,rules)
    parts=re.split(r'(?= Rule \d+: )',r['question'].replace(' What is the final number?',''))
    out=[]
    for j in range(1,len(rules)+1):
        q=''.join(parts[:j+1])+' What is the final number?'
        out.append({'id':f"{r['id']}|{j}",'question':q,'steps':['x']*j,'answer':str(v[j]),'level':j})
    return out,x0,rules,v
P=[];meta={}
for r in te:
    pf,x0,rules,v=prefixes(r); P+=pf; meta[r['id']]=(x0,rules,v)
print('prefix queries',len(P),flush=True)
# sanity: the n-rule prefix must reproduce the item's own question
assert prefixes(te[0])[0][-1]['question']==te[0]['question']
NUM={}
def numtok(tok):
    if not NUM:
        for n in range(100): NUM[tok.encode(f"The answer is: {n}")[-1]]=n
    return NUM
def answers(m,tok,rs,k):
    nt=numtok(tok); dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV); out={}
    rs=sorted(rs,key=lambda r:len(r['question']))
    for i in range(0,len(rs),32):
        bt=rs[i:i+32]; b=encode(bt,tok,m,DEV); lg,_=fast_answer(m,b,k,dec_ids=dec)
        for r,t in zip(bt,lg.argmax(-1).tolist()): out[r['id']]=nt.get(t,-999)
        del lg,b
        if DEV=='mps': torch.mps.empty_cache()
    return out
res={}
for name,ck,nl in (('CODI-0 stab',CK['stab0']%17820,0),('CODI stab',CK['stab']%17820,6),('CODI lr3e-3',CK['lr3'],6)):
    t0=time.time(); m,tok,ta=load(ck,nl,DEV); m.eval()
    a=answers(m,tok,P,nl)
    st={'exact_after_exact':[0,0],'exact_after_near':[0,0],'prop_after_near':[0,0],'near_after_near':[0,0],'exact_after_far':[0,0],'follow_own':[0,0],'follow_own_when_wrong':[0,0]}
    for rid,(x0,rules,v) in meta.items():
        for j in range(2,len(rules)+1):
            p0=a[f'{rid}|{j-1}']; p1=a[f'{rid}|{j}']
            e0=p0-v[j-1]; ru=rules[j-1]
            own=apply(ru,p0) if 0<=p0<=199 else -999       # rule j applied to the model's own previous answer
            st['follow_own'][1]+=1; st['follow_own'][0]+= p1==own
            if p1!=v[j]:
                st['follow_own_when_wrong'][1]+=1; st['follow_own_when_wrong'][0]+= p1==own
            if e0==0:
                st['exact_after_exact'][1]+=1; st['exact_after_exact'][0]+= p1==v[j]
            elif abs(e0)<=2:
                st['exact_after_near'][1]+=1; st['exact_after_near'][0]+= p1==v[j]
                st['near_after_near'][1]+=1; st['near_after_near'][0]+= 0<abs(p1-v[j])<=4
                st['prop_after_near'][1]+=1; st['prop_after_near'][0]+= p1==own
            else:
                st['exact_after_far'][1]+=1; st['exact_after_far'][0]+= p1==v[j]
    res[name]={k:(round(x/max(n,1),3),n) for k,(x,n) in st.items()}
    print(name,json.dumps(res[name]),f'{time.time()-t0:.0f}s',flush=True)
    del m
    if DEV=='mps': torch.mps.empty_cache()
# refinement sweep over latent passes on items L3-7
T2=rows(TE,per_level=150,seed=6,levels=[3,4,5,6,7])
ref={}
for name,ck in (('CODI lr3e-3',CK['lr3']),('CODI stab',CK['stab']%17820)):
    m,tok,ta=load(ck,6,DEV); m.eval()
    for k in (0,1,2,3,4,6,8,12):
        a=answers(m,tok,T2,k); e=np.array([abs(a[r['id']]-int(r['answer'])) for r in T2]); lv=np.array([int(r['level']) for r in T2]); h=lv>=5
        ref[f'{name} k={k}']=dict(exact=float(np.mean(e==0)),close=float(np.mean(e<=2)),exact_hard=float(np.mean(e[h]==0)),close_hard=float(np.mean(e[h]<=2)))
        print(name,'k',k,{x:round(y,3) for x,y in ref[f'{name} k={k}'].items()},flush=True)
    del m
    if DEV=='mps': torch.mps.empty_cache()
json.dump({'prefix':res,'refine':ref},open(sys.argv[1]+'/b6_prefix.json','w'),indent=1)
print('DONE')
