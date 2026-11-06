"""Reads b21 output. Usage: a23_reads.py <json> [levels]"""
import json, sys
R=json.load(open(sys.argv[1])); LV=sys.argv[2].split(',') if len(sys.argv)>2 else ['3','4','5']
for name,o in R.items():
    print('##',name)
    for L in LV:
        if L not in o: continue
        d=o[L]; g=d['gold']; A=d['ans']; n=len(g); base=A[str((None,None))]; ok=[i for i in range(n) if base[i]==g[i]]
        keep=lambda c:sum(A[c][i]==g[i] for i in ok)/max(len(ok),1); chg=lambda c:sum(A[c][i]!=base[i] for i in range(n))/n
        c=lambda s,k:str((s,k))
        print(f' L{L} n={n} right {len(ok)}. controls: none changes {chg(c(None,None)):.3f}; last rule blocked at answer: right answers kept {keep(c("ans",f"rule_{int(L)-1}")):.2f}')
        print('   share of right answers kept. rows = stage, columns = set blocked')
        keys=[f'rule_{j}' for j in range(int(L))]+['rules','start','rest','prev','prev1']
        print('        '+''.join(k.replace('rule_','r').rjust(7) for k in keys))
        for s in [0,1,2,3,4,5,'lat','ans']:
            print(f'   {str(s):4s} '+''.join((f'{keep(c(s,k)):.2f}' if c(s,k) in A else '-').rjust(7) for k in keys))
        at=d['attn']; Li=int(L); print('   attention mass, mean over layers: rows = pass; columns r0.. start rest earlier self sink')
        for s,row in enumerate(at):
            m=[sum(row[l][j] for l in range(12))/12 for j in range(Li+5)]
            print(f'   p{s}   '+' '.join(f'{x:.2f}' for x in m))
