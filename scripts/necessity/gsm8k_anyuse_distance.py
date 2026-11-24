"""t54 post hoc: within-item hop slope on the labelled gold edit distance vs the any-use gold distance (longest path from any gold step whose
expression contains the edited number). Run from ai-capstone/: ../codi/.venv/bin/python -I scripts/necessity/gsm8k_anyuse_distance.py"""
import sys, numpy as np, json
from collections import defaultdict
src=open('scripts/necessity/cot_route_score.py').read().split("R = {}")[0]
sys.argv=['x','/dev/null']; g={'__file__':'scripts/necessity/cot_route_score.py'}; exec(src,g)
rng=np.random.default_rng(0)
def slope(rows,key):
    by=defaultdict(list)
    for r in rows: by[r['i']].append(r)
    grp=[v for v in by.values() if len({r[key] for r in v})>1]
    def sl(sel):
        xs,ys=[],[]
        for v in sel:
            mx=np.mean([r[key] for r in v]); my=np.mean([r['h'] for r in v]); xs+=[r[key]-mx for r in v]; ys+=[r['h']-my for r in v]
        xs,ys=np.array(xs,float),np.array(ys,float); return float((xs*ys).sum()/(xs*xs).sum())
    bs=[sl([grp[j] for j in rng.integers(0,len(grp),len(grp))]) for _ in range(2000)]
    return f"{sl(grp):+.2f} [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}] ({len(grp)} items, {sum(len(v) for v in grp)} pairs)"
for m,allpairs in (('codi',False),('sim',False),('codi',True)):
    G={r['question']:r for r in json.load(open(g['A']/f't54/{m}.json'))['rows']}
    rows=[]; multi=0
    for fn in (f't31/gsm8k_{m}.json', f't31/long_gsm8k_{m}.json'):
        for p in json.load(open(g['A']/fn))['pairs']:
            if not (p.get('qualified') and p.get('hops') is not None): continue
            ok=p.get('R_right') and p.get('D_right')
            if not allpairs and not ok: continue
            qr,qd=p['R']['question'],p['D']['question']; i=g['BYQ'].get(g['qtext'](qr)); v=g['edited_number'](qr,qd)
            if v is None or i is None: continue
            gi=g['GOLD'][i]['strict']; ex=[e for e,_ in g['STEP'].findall(g['te'][i].rationale)]
            uses=[k for k,e in enumerate(ex) if any(g['same'](g['fnum'](x),g['fnum'](v)) for x in g['NUM'].findall(e))]
            du=[d for d in (g['down'](gi,k) for k in uses) if d is not None]
            lab=g['down'](gi,int(p['edit'][4:]))
            if lab is None or not du: continue
            multi+= max(du)!=lab
            rows.append(dict(i=i,h=p['hops'],lab=lab,use=max(du)))
    tag=f"{m} {'all qualified pairs' if allpairs else 'both-right pairs'}"
    print(f"{tag}: {len(rows)} pairs; any-use gold distance differs from the pair's label on {multi/len(rows):.2f}")
    print('   slope on labeled gold distance  ', slope(rows,'lab'))
    print('   slope on any-use gold distance  ', slope(rows,'use'))
