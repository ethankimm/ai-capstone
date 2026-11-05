import re, json
RP=re.compile(r"if it is even, halve it; otherwise add (\d+)$")
RT=re.compile(r"if it is at least (\d+), subtract (\d+); otherwise add (\d+)$")
RD=re.compile(r"if it is below (\d+), double it; otherwise subtract (\d+)$")
def parse(q):
    m=re.match(r"Start with (\d+)\. (.*) What is the final number\?$",q.strip())
    x0=int(m.group(1)); body=m.group(2)
    parts=re.split(r"Rule \d+: ",body)[1:]
    rules=[]
    for p in parts:
        p=p.strip().rstrip('.')
        a=RP.match(p)
        if a: rules.append(('P',0,int(a.group(1)),0)); continue
        a=RT.match(p)
        if a: rules.append(('T',int(a.group(1)),int(a.group(3)),int(a.group(2)))); continue
        a=RD.match(p)
        if a: rules.append(('D',int(a.group(1)),0,int(a.group(2)))); continue
        raise ValueError(p)
    return x0,rules
def branch(rule,s):
    k,t,a,b=rule
    if k=='P': return s%2==0
    if k=='T': return s>=t
    return s<t
def apply(rule,s,br=None):
    k,t,a,b=rule
    if br is None: br=branch(rule,s)
    if k=='P': return s//2 if br else s+a
    if k=='T': return s-b if br else s+a
    return 2*s if br else s-b
def walk(x0,rules):
    s=x0; out=[s]; brs=[]
    for r in rules:
        b=branch(r,s); s=apply(r,s,b); out.append(s); brs.append(b)
    return out,brs
def flip_outcomes(x0,rules):
    """final value if the branch at rule j is flipped once, all later rules applied correctly from the wrong value"""
    res=[]
    v,brs=walk(x0,rules)
    for j,r in enumerate(rules):
        s=apply(r,v[j],not brs[j])
        for r2 in rules[j+1:]:
            s=apply(r2,s)
        res.append(s)
    return res
def skip_outcomes(x0,rules):
    res=[]
    for j in range(len(rules)):
        s=x0
        for i,r in enumerate(rules):
            if i==j: continue
            s=apply(r,s)
        res.append(s)
    return res
def path_flip_outcomes(x0,rules):
    """branch flipped at j but later branch DECISIONS kept as in the true walk (a sum-of-selected-deltas error)"""
    res=[]
    v,brs=walk(x0,rules)
    for j in range(len(rules)):
        s=x0
        for i,r in enumerate(rules):
            s=apply(r,s,(not brs[i]) if i==j else brs[i])
        res.append(s)
    return res
