"""Reads b20 output. Usage: a24_grid.py <json> [levels]. For each target, where is it best read, and how does each method compare?"""
import json, sys
R=json.load(open(sys.argv[1])); LV=sys.argv[2].split(',') if len(sys.argv)>2 else ['3','4','5']
for name,o in R.items():
    print('##',name)
    for L in LV:
        if L not in o: continue
        d=o[L]; T=len(d['base']); S=d['sites']; print(f' L{L} targets v_0..v_{T-1} (v_{T-2} = value before the last rule, v_{T-1} = answer). base {d["base"]}')
        print('   best linear cell per target: exact @ site|layer; best lens; best MLP')
        for j in range(T):
            bl=max(d['lin'],key=lambda k:d['lin'][k][0][j]); bn=max(d['lens'],key=lambda k:d['lens'][k][j]); bm=max(d['mlp'],key=lambda k:d['mlp'][k][0][j]) if d['mlp'] else None
            print(f"    v_{j}: lin {d['lin'][bl][0][j]:.2f} @ {bl}   lens {d['lens'][bn][j]:.2f} @ {bn}   mlp {(d['mlp'][bm][0][j] if bm else 0):.2f} @ {bm}")
        print('   linear exact, best layer per site, rows = sites')
        print('          '+''.join(f'v_{j}'.rjust(7) for j in range(T)))
        for s in S:
            print(f'    {s:7s}'+''.join(f"{max(d['lin'][f'{s}|{l}'][0][j] for l in range(13)):.2f}".rjust(7) for j in range(T)))
        if d['mlp']:
            print('   MLP exact, best of layers 4-12, rows = sites')
            for s in S:
                c=[k for k in d['mlp'] if k.split('|')[0]==s]
                if c: print(f'    {s:7s}'+''.join(f"{max(d['mlp'][k][0][j] for k in c):.2f}".rjust(7) for j in range(T)))
        print('   fit-size curve, exact per target (lin | mlp):')
        for n,v in d['curve'].items(): print(f'    n={n}: {v["lin"][0]} | {v["mlp"][0]}')
