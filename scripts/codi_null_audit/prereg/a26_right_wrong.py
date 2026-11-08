"""Reads b23 output. Usage: a26_right_wrong.py <json> <rc dir>. Exact read of each step on items the model got right vs wrong, fixed cells; selectivity; does last rule(decoded value) give the model's own answer."""
import json,sys,math
sys.path.insert(0,sys.argv[2])
from rc import apply
R=json.load(open(sys.argv[1]))
def wil(k,n):
    if not n: return (0,0)
    z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return (round(c-h,2),round(c+h,2))
for name,o in R.items():
    print('##',name)
    for L,d in o.items():
        V=d['V'];ans=d['ans'];gold=d['gold'];n=len(V);T=len(V[0]);right=[i for i in range(n) if ans[i]==gold[i]];wrong=[i for i in range(n) if ans[i]!=gold[i]]
        print(f' L{L} n={n} right {len(right)} wrong {len(wrong)} fit {d["n_fit"]} targets v_0..v_{T-1}; v_{T-2} before last rule, v_{T-1} answer')
        for c,x in d['cells'].items():
            if c.split('|')[1] not in ('0','2','4','12'): continue
            F=x['full'];H=x['half'];S=x['shuf']
            ex=lambda P,I,j:sum(P[i][j]==V[i][j] for i in I)/max(len(I),1)
            js=range(1,T)
            print(f'   {c:9s} all '+' '.join(f'{ex(F,range(n),j):.2f}' for j in js)+' | right '+' '.join(f'{ex(F,right,j):.2f}' for j in js)+' | wrong '+' '.join(f'{ex(F,wrong,j):.2f}' for j in js)+' | shuf '+' '.join(f'{ex(S,range(n),j):.2f}' for j in js)+' | fit4k all '+' '.join(f'{ex(H,range(n),j):.2f}' for j in js))
        for c in (['h4|2','h4|0','h6|2'] if 'h4|2' in d['cells'] else ['t1|2','colon|2','t1|12']):
            F=d['cells'][c]['full'];j=T-2
            pa=[apply(tuple(d['last_rule'][i]),F[i][j]) for i in range(n)]
            km=sum(pa[i]==ans[i] for i in right); kw=sum(pa[i]==ans[i] for i in wrong)
            # chance: pair another item's decoded value with this item's last rule, wrong items only
            sh=sum(apply(tuple(d['last_rule'][i]),F[wrong[(q+1)%len(wrong)]][j])==ans[i] for q,i in enumerate(wrong)) if wrong else 0
            print(f'   last rule(decoded {c}) = model answer: right {km}/{len(right)}={km/max(len(right),1):.2f} {wil(km,len(right))}; wrong {kw}/{len(wrong)}={kw/max(len(wrong),1):.2f} {wil(kw,len(wrong))}; wrong, other item decoded {sh/max(len(wrong),1):.2f}; wrong answers equal to gold-path-free check')
