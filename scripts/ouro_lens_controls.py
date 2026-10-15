"""Post-hoc controls for `ouro_logit_lens.py` (run 20261015-024317_recurrent_depth_logit-lens-ouro-1.4b),
from its predictions.jsonl (end-of-loop lens digit log-probs at the suffix's ' ' per item):
is the first intermediate's leading digit there because Ouro computed it, or because the same
digit leads a number in the question? Splits the step1-vs-donor comparison by whether step1's
leading digit also leads a question number, and runs the same comparison for a question
number's leading digit (copy control).

  python scripts/ouro_lens_controls.py <run dir> <xmech_questions.jsonl>
"""
import json,re,random,sys
R=sys.argv[1]
qs={}
for l in open(sys.argv[2]):
    d=json.loads(l); qs[d["key"]]=d
rows=[json.loads(l) for l in open(f"{R}/predictions.jsonl")]
NUM=re.compile(r"\d[\d,]*(?:\.\d+)?")
def leads(q):
    out=set()
    for n in NUM.findall(q):
        s=re.sub(r"[^\d]","",n).lstrip("0")
        if s: out.add(int(s[0]))
    return out
rng=random.Random(1)
def stat(rs,key,pos="sfx3"):
    out=[]
    for u in range(4):
        w=[r[f"lens_digit_logp_{pos}_end_of_loop"][u][key(r)] > r[f"lens_digit_logp_{pos}_end_of_loop"][u][r["donor_digit"]] for r in rs]
        out.append(round(sum(w)/len(w),3))
    return out
S=lambda r:r["lead_step1"]; A=lambda r:r["lead_answer"]
for r in rows: r["qleads"]=leads(qs[r["key"]]["question"])
differ=[r for r in rows if r["lead_step1"]!=r["lead_answer"]]
novel=[r for r in differ if r["lead_step1"] not in r["qleads"]]
copy=[r for r in differ if r["lead_step1"] in r["qleads"]]
print("n differ",len(differ),"step1 lead novel",len(novel),"copyable",len(copy))
print("step1 beats donor, end of loop 1..4  differ  :",stat(differ,S))
print("                                     novel   :",stat(novel,S))
print("                                     copyable:",stat(copy,S))
print("answer beats donor                   differ  :",stat(differ,A))
cc=[]
for r in differ:
    c=[d for d in r["qleads"] if d not in (r["lead_step1"],r["lead_answer"],r["donor_digit"])]
    if c: r["qctrl"]=rng.choice(c); cc.append(r)
print("question-number lead digit beats donor (n=%d):"%len(cc), stat(cc,lambda r:r["qctrl"]))
print("frac donors that are question-number leads:",round(sum(r["donor_digit"] in r["qleads"] for r in differ)/len(differ),3))
clean=[r for r in novel if r["donor_digit"] not in r["qleads"]]
print("step1 novel & donor not in question (n=%d):"%len(clean), stat(clean,S))
cc2=[r for r in cc if r["donor_digit"] not in r["qleads"]]
print("q-number lead vs donor not in question (n=%d):"%len(cc2), stat(cc2,lambda r:r["qctrl"]))
corr=[r for r in novel if r["ouro_correct"]]
print("novel & ouro correct (n=%d): step1"%len(corr), stat(corr,S), "answer:",stat(corr,A))
an=[r for r in differ if r["lead_answer"] not in r["qleads"]]
print("answer lead novel (n=%d): answer"%len(an), stat(an,A))
