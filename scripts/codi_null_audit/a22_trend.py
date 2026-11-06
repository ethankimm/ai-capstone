"""Reads b19 output. Usage: a22_trend.py <json> [name ...]"""
import json, sys
R=json.load(open(sys.argv[1])); names=sys.argv[2:] or list(R)
for n in names:
    o=R[n]; print('##',n,'exact by level',{k:round(v,2) for k,v in o['exact'].items()})
    for L in ('3','4','5'):
        if L not in o['read']: continue
        T=len(o['base'][L]); print(f'  L{L} readout, exact (within 2): rows are sites, columns v_0..v_{T-1}; last column is the answer')
        print('    base '+' '.join(f"{x:.2f}".rjust(10) for x in o['base'][L]))
        for s,(ex,w2) in enumerate(o['read'][L]): print(f'    h_{s}  '+' '.join(f"{a:.2f} ({b:.2f})".rjust(10) for a,b in zip(ex,w2)))
    if 'tail' in o:
        print('  exact with latents ablated from site s on, by level')
        for s,v in o['tail'].items(): print(f'    s={s} '+' '.join(f"L{k} {x:.2f}" for k,x in v.items()))
    if 'patch' in o:
        p=o['patch']['pairs']; C=o['patch']['cond']; ii=list(range(len(p)))
        f=lambda c:(sum(C[c][i]==p[i]['dgold'] for i in ii)/max(len(ii),1),sum(C[c][i]==p[i]['gold'] for i in ii)/max(len(ii),1))
        lv={}
        for x in p: lv[x['level']]=lv.get(x['level'],0)+1
        print(f"  patches, {len(p)} pairs by level {dict(sorted(lv.items()))}. own latents unchanged {f('self')[1]:.3f}. all latents follows donor {f('all')[0]:.3f}")
        for k in ('single','prefix','suffix'):
            print(f'    {k}: follows donor / unchanged  '+'  '.join(f"s{c.split('_')[1]} {f(c)[0]:.2f}/{f(c)[1]:.2f}" for c in C if c.startswith(k)))
