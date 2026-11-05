"""Reads b15_raw.json. Centre of the single-layer patch gain, by edited rule j and hop k, so timing is not read only relative to the edit. Usage: a19_hoptiming.py <this dir> <artifacts dir>"""
import json, sys
W=sys.argv[2]
import numpy as np
E=json.load(open(W+'/b15_raw.json')); FZ=E['FZ']; S=E['store']
print('peak layer (centre of gain over layers, weights clipped at 0) of patching rule j+k; constant edits on items answered right; n in brackets')
for n,s in S.items():
    g=[(q,r) for q,r in zip(FZ,s['route']) if q['kind']=='const' and abs(r['clean']-q['gold'])<=2 and abs(r['edit']-q['new'])<=2]
    print(' ##',n)
    print('   by absolute rule patched (r = j+k) and hop k')
    print('       '+''.join(f'k={k}'.rjust(13) for k in (1,2,3,4)))
    for R in range(2,8):
        row=''
        for k in (1,2,3,4):
            c=[(q,r) for q,r in g if q['j']+k==R and q['dist']>=k and q['dist']>k]   # rule R is not the last rule
            if len(c)<8: row+='-'.rjust(13); continue
            dt=np.array([q['new']-q['gold'] for q,r in c],float); w=[]
            for l in range(12):
                dm=np.clip(np.array([r['grid'][f'{k}|{l}']-r['clean'] for q,r in c],float),-30,30); w.append(max(0,float((dm*dt).sum()/(dt*dt).sum())))
            w=np.array(w); row+=(f'{(w*np.arange(1,13)).sum()/max(w.sum(),1e-9):.1f} ({len(c)})').rjust(13)
        print(f'   rule {R}'+row)
