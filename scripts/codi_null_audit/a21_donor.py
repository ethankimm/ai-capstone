"""Reads b16 output. Usage: a21_donor.py <raw json>"""
import json, sys, collections
import numpy as np
D=json.load(open(sys.argv[1])); P=D['P']
def cat(p,a,tol):
    if abs(a-p['gold'])<=tol: return 'correct'
    if abs(a-p['dgold'])<=tol: return 'donor'
    return 'other'
def rate(pairs,res,tol=0):
    c=collections.Counter(cat(P[i],res[str(i)],tol) for i in pairs if str(i) in res); n=sum(c.values())
    return (c['correct']/n,c['donor']/n,c['other']/n,n) if n else (0,0,0,0)
for name,s in D['store'].items():
    R=s['res']; keep=s['keep']; diff=[i for i in keep if P[i]['kind']=='diff']; same=[i for i in keep if P[i]['kind']=='same']
    print('##',name,'different-value pairs',len(diff),'same-value pairs',len(same))
    print('  control, own latents written back, answer unchanged:',round(float(np.mean([R['self'][str(i)]==P[i]['gold'] for i in diff])),3))
    print('  by level: correct anyway / follows donor / other, exact')
    for L in range(2,8):
        ii=[i for i in diff if P[i]['level']==L]
        if len(ii)<10: continue
        a=rate(ii,R['all']); d=rate(ii,R['direct']); sv=[i for i in same if P[i]['level']==L]; c=rate(sv,R['all'])
        print(f'   L{L} n={len(ii):4d} through latents {a[0]:.2f}/{a[1]:.2f}/{a[2]:.2f} | direct route {d[0]:.2f}/{d[1]:.2f}/{d[2]:.2f} | same-value donor keeps answer {c[0]:.2f} (n={c[3]})')
    print('  by rules left after the edited rule: correct anyway / follows donor / other')
    for k,lab in ((0,'0 (last rule)'),(1,'1'),(2,'2'),(3,'3+')):
        ii=[i for i in diff if (P[i]['left']==k if k<3 else P[i]['left']>=3)]
        a=rate(ii,R['all']); d=rate(ii,R['direct']); print(f'   {lab:14s} n={a[3]:4d} through latents {a[0]:.2f}/{a[1]:.2f}/{a[2]:.2f} | direct route {d[0]:.2f}/{d[1]:.2f}/{d[2]:.2f}')
    print('  follows donor by site, edits before the last rule: single_s | suffix_s')
    early=[i for i in diff if P[i]['left']>=1]
    for t in range(s['nl']):
        a=rate(early,R[f'single_{t}']); b=rate(early,R[f'suffix_{t}']) if t else rate(early,R['all'])
        print(f'   z_{t} {a[1]:.2f} | {b[1]:.2f}')
