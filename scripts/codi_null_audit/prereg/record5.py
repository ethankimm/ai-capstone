import json, sys
import numpy as np
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W=sys.argv[1]; D=json.load(open(W+'/b17_raw.json'))
def tgt(x): return x['bad'] if x['kind'].startswith('wrong') else x['gold']
ex={}
for n,s in D.items():
    ex[n]={'wrong_items':s['n_wrong']}
    for k in ('same length donor','short donor','wrong value donor'):
        rr=[x for x in s['rows'] if x['kind']==k]
        ex[n][k]=dict(n=len(rr),hits_target=round(float(np.mean([x['ans']==tgt(x) for x in rr])),3),within_2=round(float(np.mean([abs(x['ans']-tgt(x))<=2 for x in rr])),3),unchanged=round(float(np.mean([x['ans']==x['pred'] for x in rr])),3),
            by_level={str(L):round(float(np.mean([x['ans']==tgt(x) for x in rr if x['level']==L])),3) for L in range(4,8) if any(x['level']==L for x in rr)})
rows=[]
for n,s in D.items():
    for i,x in enumerate(s['rows']):
        rows.append(dict(idx=len(rows),id=f"{n}|{x['item']}|{x['kind']}",model=n,kind=x['kind'],level=x['level'],gold_answer=x['gold'],original_prediction=x['pred'],wrong_value_target=x['bad'],predicted_answer=x['ans']))
rec=RunRecord(run_id=new_run_id('codi','v2-l1to7w-latent-repair'),mechanism='codi',stage='full_run',
 model=ModelInfo(backbone='gpt2',checkpoint='stabilised CODI checkpoint-17820 (20261105-044444); lr-3e-3 CODI (20261103-193425); round-1 CODI',n_params=144499200),
 dataset=DatasetInfo(name='synthetic-depth2',split='test',n_examples=600,seed=71),
 metrics=dict(final_answer_accuracy=ex['r1 CODI']['short donor']['hits_target'],unparseable_rate=0.0,compute_steps=6,extra=dict(accuracy_kind='round-1 CODI: share of its wrong L3-7 answers made right by latents from a short donor with the true value; not task accuracy',repair=ex)),
 hyperparams=dict(scripts='scripts/codi_null_audit/b17_repair.py',wrong_items_per_model=600,levels=[3,4,5,6,7],seed=17,dtype='float32'),
 seed=71,hardware='MacBook M4 Pro 24 GB, MPS',notes='Where do wrong answers come from? Patch latents holding the true value into wrongly answered items.')
rec.save(predictions=rows); print(rec.run_id); print(json.dumps({n:{k:(v if not isinstance(v,dict) else {a:b for a,b in v.items() if a!='by_level'}) for k,v in e.items()} for n,e in ex.items()})[:900])
