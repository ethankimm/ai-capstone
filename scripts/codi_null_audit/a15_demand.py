"""No model. Accuracy table, what chains demand of the running value, and where errors and the CODI gain sit.
Usage: a15_demand.py <this dir> <folder of round-1 predictions>   (run from the repo root)"""
import json, sys, collections
sys.path.insert(0,sys.argv[1]); PR=sys.argv[2]
import numpy as np
from rc import *
from scipy.stats import binomtest
R='results/'
ALL=[('stab CODI',R+'20261105-044550_codi_v2-l1to7w-stab-depth2-codi-canonical/predictions.jsonl'),
 ('stab CODI-0',R+'20261105-024204_codi_v2-l1to7w-stab-depth2-codi0-canonical/predictions.jsonl'),
 ('lr3 CODI L1-7w',R+'20261103-193425_codi_v2-l1to7w-depth2-codi/predictions.jsonl'),
 ('r1 CODI',PR+'/20261102-000204_codi_v2-depth2-gpt2_depth2_codi_bs64-sel.jsonl'),
 ('r1 CODI-0',PR+'/20261102-060537_codi_v2-depth2-gpt2_depth2_codi0.jsonl'),
 ('no_thoughts',PR+'/20261101-153922_baseline_no_cot_v2-depth2-gpt2_depth2_no_thoughts_m6.jsonl'),
 ('no-CoT',PR+'/20261101-154024_baseline_no_cot_v2-depth2-gpt2_depth2_no_cot_plateau.jsonl'),
 ('Coconut',PR+'/20261102-022150_coconut_v2-depth2-gpt2_depth2_coconut_c1m6.jsonl'),
 ('SIM-Coconut',PR+'/20261104-144235_coconut_v2-depth2-pilot_k8_simcot_c1m6.jsonl')]
def num(x):
    try: return int(str(x).strip())
    except: return -999
print('0. exact/within 2 by level')
print('model'.ljust(16)+''.join(f'L{L}'.rjust(11) for L in range(1,13)))
for name,p in ALL:
    rows=[json.loads(l) for l in open(p)]; d={}
    for L in {r['level'] for r in rows}:
        e=np.abs(np.array([num(r['predicted_answer']) for r in rows if r['level']==L])-np.array([int(r['gold_answer']) for r in rows if r['level']==L]))
        d[L]=(np.mean(e==0),np.mean(e<=2))
    print(name.ljust(16)+''.join((f'{d[L][0]:.2f}/{d[L][1]:.2f}'.rjust(11) if L in d else '-'.rjust(11)) for L in range(1,13)))
P=dict(ALL[:2]+ALL[3:5])
D={k:{r['id']:r for r in map(json.loads,open(p))} for k,p in P.items()}
ids=list(D['stab CODI'])
def margins(x0,rules):
    v,b=walk(x0,rules); out=[]
    for j,r in enumerate(rules):
        k,t,a,c=r; s=v[j]
        out.append(None if k=='P' else ((s-t+1) if s>=t else (t-s)))
    return v,b,out
print('\n1. decisions per item by what they need')
print('level  P rules  T/D margin<=1  <=2  <=4  >4 | share of T/D with margin>4 | items with every T/D margin>2 | items with no P after a halving and every T/D margin>2')
for L in range(2,9):
    c=collections.Counter(); allw=0;n=0;easy=0
    for i in ids:
        r=D['stab CODI'][i]
        if int(r['level'])!=L: continue
        x0,ru=parse(r['question']); v,b,m=margins(x0,ru); n+=1
        td=[x for x in m if x is not None]
        c['P']+=sum(x is None for x in m); c['1']+=sum(x<=1 for x in td); c['2']+=sum(x<=2 for x in td); c['4']+=sum(x<=4 for x in td); c['w']+=sum(x>4 for x in td)
        allw+=all(x>2 for x in td)
        h=False;pa=False
        for rr,bb in zip(ru,b):
            if rr[0]=='P':
                pa=pa or h
                h=h or bb
        easy+=all(x>2 for x in td) and not pa
    print(f"L{L}   {c['P']/n:.2f}   {c['1']/n:.2f} {c['2']/n:.2f} {c['4']/n:.2f} {c['w']/n:.2f} | {c['w']/(c['4']+c['w']):.2f} | {allw/n:.2f} | {easy/n:.2f}")
print('\n2. discordant exact items by smallest T/D margin: (items, CODI only, CODI-0 only)')
for pa,pb in (('r1 CODI','r1 CODI-0'),('stab CODI','stab CODI-0')):
    for lv in ((2,3),(4,4),(5,7)):
        tab=collections.defaultdict(lambda:[0,0,0])
        for i in ids:
            a=D[pa][i];b=D[pb][i]
            if not lv[0]<=int(a['level'])<=lv[1]: continue
            x0,ru=parse(a['question']); v,br,m=margins(x0,ru); td=[x for x in m if x is not None]; mm=min(td) if td else 99
            key='<=1' if mm<=1 else '2' if mm==2 else '3-4' if mm<=4 else '5+'
            g=int(a['gold_answer']); ca=num(a['predicted_answer'])==g; cb=num(b['predicted_answer'])==g
            tab[key][0]+=1; tab[key][1]+=ca and not cb; tab[key][2]+=cb and not ca
        print(pa,lv,{k:tuple(tab[k]) for k in ('<=1','2','3-4','5+')})
print('\n3. rule-level branch check where true and single-flip outcomes differ by >=10 (L3-7); wrong-branch share = flip/(true+flip)')
for name in P:
    tab=collections.defaultdict(lambda:[0,0,0])
    for i in ids:
        r=D[name][i]; L=int(r['level'])
        if not 3<=L<=7: continue
        x0,ru=parse(r['question']); v,br,m=margins(x0,ru); fl=flip_outcomes(x0,ru); g=v[-1]; p=num(r['predicted_answer']); nh=0
        for j,(rr,b) in enumerate(zip(ru,br)):
            if abs(fl[j]-g)>=10 and 0<=fl[j]<=99:
                if rr[0]=='P': key=f"P, {min(nh,2)} earlier halvings"
                else:
                    mm=m[j]; key='T/D margin 1-2' if mm<=2 else 'T/D margin 3-4' if mm<=4 else 'T/D margin 5-8' if mm<=8 else 'T/D margin 9+'
                pos='rule 1' if j==0 else 'rule 2-3' if j<=2 else 'rule 4+'
                t=tab[key+' | '+pos]
                if abs(p-g)<=2: t[0]+=1
                elif abs(p-fl[j])<=2: t[1]+=1
                else: t[2]+=1
            if rr[0]=='P' and b: nh+=1
    print('##',name)
    for k in sorted(tab):
        t=tab[k]; n=sum(t)
        if n<25: continue
        print(f'  {k:38s} n={n:5d} true {t[0]/n:.3f} flip {t[1]/n:.3f} other {t[2]/n:.3f} wrong-branch share {t[1]/max(1,t[0]+t[1]):.3f}')
