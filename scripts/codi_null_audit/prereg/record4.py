import json, sys
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id
W=sys.argv[1]; D=json.load(open(W+'/b16_raw.json')); P=D['P']; S=D['store']
def follows(name,cond,sel):
    s=S[name]; r=s['res'][cond]; ii=[i for i in s['keep'] if P[i]['kind']=='diff' and sel(P[i])]
    return round(sum(r[str(i)]==P[i]['dgold'] for i in ii)/len(ii),3),len(ii)
ex={n:{'through_latents_before_last_rule':follows(n,'all',lambda p:p['left']>=1),'direct_before_last_rule':follows(n,'direct',lambda p:p['left']>=1),'through_latents_last_rule':follows(n,'all',lambda p:p['left']==0),'direct_last_rule':follows(n,'direct',lambda p:p['left']==0),
   'own_latent_control':round(sum(S[n]['res']['self'][str(i)]==P[i]['gold'] for i in S[n]['keep'] if P[i]['kind']=='diff')/sum(P[i]['kind']=='diff' for i in S[n]['keep']),3)} for n in S}
rows=[]
for i,p in enumerate(P):
    row=dict(idx=i,id=f"{p['item']}|{p['kind']}|{i}",question=p['rq'],donor_question=p['dq'],gold_answer=p['gold'],donor_answer=p['dgold'],kind=p['kind'],level=p['level'],rule=p['rule'],rules_left=p['left'])
    for n,s in S.items():
        row[n]={c:v[str(i)] for c,v in s['res'].items() if str(i) in v}
    rows.append(row)
rec=RunRecord(run_id=new_run_id('codi','v2-l1to7w-latent-donor-mediation'),mechanism='codi',stage='full_run',
 model=ModelInfo(backbone='gpt2',checkpoint='stabilised CODI checkpoint-17820 (20261105-044444); lr-3e-3 CODI (20261103-193425); round-1 CODI',n_params=144499200),
 dataset=DatasetInfo(name='synthetic-depth2',split='test',n_examples=len({p['item'] for p in P}),seed=61),
 metrics=dict(final_answer_accuracy=ex['r1 CODI']['through_latents_before_last_rule'][0],unparseable_rate=0.0,compute_steps=6,extra=dict(accuracy_kind='round-1 CODI: share of edits before the last rule whose answer follows the donor when all latents are patched (exact); not task accuracy',routes=ex,pairs=dict(different_value=sum(p['kind']=='diff' for p in P),same_value=sum(p['kind']=='same' for p in P)))),
 hyperparams=dict(scripts='scripts/codi_null_audit/ (b16_donor, a21_donor)',items_per_level=200,levels=[2,3,4,5,6,7],build_seed=16,dtype='float32'),
 seed=61,hardware='MacBook M4 Pro 24 GB, MPS',notes='Which route carries the running value to the answer? Donor patch, mediation and same-value donors on three CODI models.')
rec.save(predictions=rows); print(rec.run_id); print(json.dumps(ex))
