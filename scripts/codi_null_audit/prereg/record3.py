import json, sys, re
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W=sys.argv[1]
sys.path.insert(0,'scripts/codi_null_audit')
from rc import parse, walk
D=json.load(open(W+'/b13_raw.json')); E=json.load(open(W+'/b15_raw.json')); P=json.load(open(W+'/b14_probes.json'))
ZS=D['ZS']; ST=D['store']; S=E['store']; FZ=E['FZ']
def tcount(q):
    x0,ru=parse(q); v,b=walk(x0,ru); c=0
    for j,r in enumerate(ru):
        if r[0]!='P':
            s=v[j]; c+=((s-r[1]+1) if s>=r[1] else (r[1]-s))<=2
    return c
tc=np.array([tcount(z['q']) for z in ZS]); gold=np.array([z['gold'] for z in ZS]); L=np.array([z['level'] for z in ZS])
preds=[]
for i,z in enumerate(ZS):
    row=dict(idx=i,id=f"constructed-{i:05d}",question=z['q'],gold_answer=z['gold'],level=z['level'],tight_set=z['tight'],p_after_halving=z['halve'],n_tight=int(tc[i]))
    for n,s in ST.items(): row['pred '+n]=s['z'][i]
    for n,s in S.items():
        if 'abl' in s: row['pred ablated '+n]=s['abl'][i]
    preds.append(row)
def acc(n,ii): return round(float(np.mean(np.array(ST[n]['z'])[ii]==gold[ii])),3)
pair={}
for a,b,lab in (('stab CODI','stab CODI-0','stabilised'),('r1 CODI','r1 CODI-0','round1')):
    A=np.array(ST[a]['z'])==gold; B=np.array(ST[b]['z'])==gold; pair[lab]={}
    for c,ii in (('0',tc==0),('1',tc==1),('2',tc==2),('3',tc==3),('4+',tc>=4)):
        pair[lab][c]=dict(n=int(ii.sum()),codi=acc(a,ii),codi0=acc(b,ii),wins=int((A[ii]&~B[ii]).sum()),losses=int((~A[ii]&B[ii]).sum()))
abl={}
for c in ('stab CODI','r1 CODI','lr3 CODI'):
    I=np.array(ST[c]['z'])==gold; X=np.array(S[c]['abl'])==gold
    abl[c]={k:dict(intact=round(float(I[ii].mean()),3),ablated=round(float(X[ii].mean()),3)) for k,ii in (('0 tight',tc==0),('1-3 tight L3-4',(tc>=1)&(tc<=3)&(L<=4)),('1-3 tight L5-7',(tc>=1)&(tc<=3)&(L>=5)))}
def gain(rows,k):
    dt=np.array([q['new']-q['gold'] for q,r in rows],float); dm=np.array([r[k]-r['clean'] for q,r in rows],float); return round(float((dm*dt).sum()/(dt*dt).sum()),2)
route={}
for n,s in S.items():
    g=[(q,r) for q,r in zip(FZ,s['route']) if q['kind']=='const' and abs(r['clean']-q['gold'])<=2 and abs(r['edit']-q['new'])<=2 and all(abs(r[k]-r['clean'])<=30 for k in ('Bnext','Blast','Anext','Alast'))]
    route[n]={k:gain(g,k) for k in ('edit','Bnext','Blast','Anext','Alast')}; route[n]['n']=len(g)
base=D['base']; g0={r['id']:int(r['answer']) for r in base}
b=ST['stab CODI-0']['base']; fa=float(np.mean([b[i]==g0[i] for i in g0]))
rec=RunRecord(run_id=new_run_id('codi','v2-l1to7w-value-resolution-route'),mechanism='codi',stage='full_run',
 model=ModelInfo(backbone='gpt2',checkpoint='checkpoint-17820 of 20261105-024111 and 20261105-044444; 20261103-193425; round-1 CODI-0 and CODI',n_params=144499200),
 dataset=DatasetInfo(name='synthetic-depth2',split='test',n_examples=len(g0),seed=41),
 metrics=dict(final_answer_accuracy=round(fa,4),unparseable_rate=0.0,compute_steps=0,extra=dict(accuracy_kind='stabilised CODI-0, exact, 200 test items per level L3-7; predictions.jsonl holds the 10,000 constructed chains',paired_exact_by_tight_decisions=pair,ablation_constructed=abl,one_rule_route_gain=route,counts=dict(threshold_queries=sum(q['kind']=='psy' for q in D['Q']),small_edits=sum(q['kind']=='bit' for q in D['Q']),freeze_pairs=len(D['FZ']),route_pairs=len(FZ),constructed=len(ZS)))),
 hyperparams=dict(scripts='scripts/codi_null_audit/ (a15-a19, b13-b15)',items_per_level=200,levels=[3,4,5,6,7],build_seed=13,dtype='float32',raw_outputs='~/Documents/Penn/CIS5980/codi_audit_artifacts'),
 seed=41,hardware='MacBook M4 Pro 24 GB, MPS',notes='How precise is the running value, how does it travel, and where does the CODI gain sit?')
rec.save(predictions=preds); print(rec.run_id); print(json.dumps(pair)); print(json.dumps(abl)); print(json.dumps(route))
