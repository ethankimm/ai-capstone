"""Where are the values? Decoding across tokens, latents and layers. Usage: b20_grid.py <this dir> <device> <output dir>. SMOKE=1: a few items.
Sites: t13..t1 (last 13 question positions, t1 = <bot>), h1..h6 (latent passes), eot/The/answer/is/colon (answer tokens). Layers 0 (input) to 12.
Targets: v_0 (start) to v_L, per level, so nothing mixes across levels. Fit on train items, scored on valid.
lin  : ridge to digit one-hots and the value, every (site, layer).
lens : final layer norm then the unembedding, restricted to the tokens 0-99, every (site, layer). No fitting.
mlp  : one hidden layer of 512, chosen cells. Shows whether linear probes understate what is there.
curve: linear and MLP at 1000, 2000 and 4000 fit items on one cell, to see whether fit size limits exact reads.
Negative control: in start-last, t13 closes the last rule, before the start value, so no running value can be there."""
import sys, json, time, os; sys.path.insert(0,sys.argv[1])
from harness import *
import numpy as np
from rc import *
DEV=sys.argv[2]; OUT=sys.argv[3]; SMOKE=int(os.environ.get('SMOKE','0'))
FIT=300 if SMOKE else 4000; LV=[4] if SMOKE else [3,4,5]
SL=BASE+'/codi_runs_l1to7w_startlast'; DL=BASE+'/synthetic_data_v2_l1to7w_startlast/depth2/'; DF=BASE+'/synthetic_data_v2_l1to7w/depth2/'
MODELS=[('startlast CODI',f'{SL}/codi/run/checkpoint-17820',6,True),('lr3 relay',CK['lr3'],6,False),('startlast CODI-0',f'{SL}/codi0/run/checkpoint-17820',0,True)]
if SMOKE: MODELS=MODELS[:1]
NT=13; DECN=['eot','The','answer','is','colon']
def names(nl): return [f't{NT-i}' for i in range(NT)]+[f'h{s+1}' for s in range(nl)]+DECN
MLPC=[('t1',l) for l in (4,6,8,10,12)]+[('colon',l) for l in (4,6,8,10,12)]+[(f'h{s}',l) for s in (1,2,3,4,6) for l in (4,6,8,10,12)]
if SMOKE: MLPC=[('h3',12)]
@torch.no_grad()
def collect(m,tok,rows_,nl,dec):
    S=len(names(nl)); body=body_of(m); X=np.zeros((len(rows_),S,13,768),np.float16); order=sorted(range(len(rows_)),key=lambda i:len(rows_[i]['question']))
    for i in range(0,len(order),32):
        ii=order[i:i+32]; b=encode([dict(question=rows_[j]['question'].strip(),steps=['x'],answer='0') for j in ii],tok,m,DEV)
        mask=b['encoder_attention_mask'].long(); pos=(mask.cumsum(-1)-1).clamp(min=0)
        o=body(input_ids=b['encoder_input_ids'],use_cache=True,attention_mask=mask,position_ids=pos,output_hidden_states=True)
        cols=[torch.stack(o.hidden_states,2)[:,-NT:]]; past=o.past_key_values; npos=mask.sum(-1,keepdim=True); z=m.prj(o.last_hidden_state[:,-1:,:]) if nl else None
        for s in range(nl):
            mask=torch.cat((mask,torch.ones(mask.size(0),1,dtype=mask.dtype,device=mask.device)),1)
            o=body(inputs_embeds=z,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos,output_hidden_states=True)
            cols.append(torch.stack(o.hidden_states,2)); past=o.past_key_values; npos=npos+1; z=m.prj(o.last_hidden_state[:,-1:,:])
        emb=m.get_embd(m.codi,m.model_name)(dec)[None].expand(mask.size(0),-1,-1); T=emb.size(1)
        mask=torch.cat((mask,torch.ones(mask.size(0),T,dtype=mask.dtype,device=mask.device)),1)
        o=body(inputs_embeds=emb,use_cache=True,past_key_values=past,attention_mask=mask,position_ids=npos+torch.arange(T,device=mask.device)[None],output_hidden_states=True)
        cols.append(torch.stack(o.hidden_states,2)); X[ii]=torch.cat(cols,1).to('cpu',torch.float16).numpy()
        del b,o,cols
        if DEV=='mps': torch.mps.empty_cache()
    return X
def targets(rows_):
    V=np.array([walk(*parse(r['question']))[0] for r in rows_]); assert V.min()>=0 and V.max()<=99
    n,T=V.shape; Y=np.zeros((n,21*T),np.float32)
    for j in range(T):
        Y[np.arange(n),21*j+V[:,j]//10]=1; Y[np.arange(n),21*j+10+V[:,j]%10]=1; Y[:,21*j+20]=(V[:,j]-50)/50
    return V,Y
def score(P,v):
    assert np.isfinite(P).all(); ex=[];w2=[]
    for j in range(v.shape[1]):
        t=P[:,21*j:21*j+10].argmax(1); o=P[:,21*j+10:21*j+20].argmax(1)
        ex.append(round(float(np.mean(10*t+o==v[:,j])),3)); w2.append(round(float(np.mean(np.abs(np.rint(P[:,21*j+20]*50+50)-v[:,j])<=2)),3))
    return ex,w2
def lin(Xa,Ya,Va,Xb,vb):
    Xa=Xa.astype(np.float64); Xb=Xb.astype(np.float64); mu,sd=Xa.mean(0),Xa.std(0)+1e-6; Xa=(Xa-mu)/sd; Xb=(Xb-mu)/sd; ym=Ya.mean(0); Yc=(Ya-ym).astype(np.float64); c=int(len(Xa)*.85)
    def fit(X,Y):
        w,V=np.linalg.eigh(X.T@X); XY=V.T@(X.T@Y); return lambda l:V@(XY/(np.maximum(w,0)+l)[:,None])
    f=fit(Xa[:c],Yc[:c]); best=max((1e1,1e2,1e3,1e4,1e5),key=lambda l:float(np.mean(score(Xa[c:]@f(l)+ym,Va[c:])[0])))
    return score(Xb@fit(Xa,Yc)(best)+ym,vb)
def mlp(Xa,Ya,Va,Xb,vb,epochs=40):
    Xa=Xa.astype(np.float32); Xb=Xb.astype(np.float32); mu,sd=Xa.mean(0),Xa.std(0)+1e-6
    A=torch.tensor((Xa-mu)/sd); Bv=torch.tensor((Xb-mu)/sd); Y=torch.tensor(Ya); T=vb.shape[1]; c=int(len(A)*.85); torch.manual_seed(0)
    net=torch.nn.Sequential(torch.nn.Linear(768,512),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(512,21*T)); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-2)
    best=-1;state=None
    for e in range(epochs):
        net.train(); p=torch.randperm(c)
        for i in range(0,c,256):
            k=p[i:i+256]; o=net(A[k]); y=Y[k]; loss=0
            for j in range(T):
                loss=loss+F.cross_entropy(o[:,21*j:21*j+10],y[:,21*j:21*j+10].argmax(1))+F.cross_entropy(o[:,21*j+10:21*j+20],y[:,21*j+10:21*j+20].argmax(1))+F.mse_loss(o[:,21*j+20],y[:,21*j+20])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): s=float(np.mean(score(net(A[c:]).numpy(),Va[c:])[0]))
        if s>best: best=s; state={k:v.clone() for k,v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): return score(net(Bv).numpy(),vb)
def measure(ck,nl,last):
    D=DL if last else DF; m,tok,_=load(ck,nl,DEV); m.eval(); dec=torch.tensor([m.eot_id]+tok.encode('The answer is:'),device=DEV)
    num={n:tok.encode(f"The answer is: {n}")[-1] for n in range(100)}; W=m.codi.get_base_model().lm_head.weight[[num[n] for n in range(100)]].detach().float().cpu().numpy()
    ln=body_of(m).ln_f; g=ln.weight.detach().float().cpu().numpy(); be=ln.bias.detach().float().cpu().numpy()
    def lens(H,l,v):
        H=H.astype(np.float32)
        if l<12: H=(H-H.mean(-1,keepdims=True))/np.sqrt(H.var(-1,keepdims=True)+1e-5)*g+be
        p=(H@W.T).argmax(1); return [round(float(np.mean(p==v[:,j])),3) for j in range(v.shape[1])]
    va=json.load(open(D+'valid.json')); N=names(nl); out={}
    ids=tok(va[-1]['question'].strip()+'\n').input_ids; print('last question tokens',[tok.decode([t]) for t in ids[-(NT-1):]]+['<bot>'],flush=True)
    for L in LV:
        fit=rows(D+'train.json',per_level=FIT,seed=20,levels=[L]); ev=[r for r in va if int(r['level'])==L]
        ev=ev[:40] if SMOKE else ev; Xa=collect(m,tok,fit,nl,dec); Xb=collect(m,tok,ev,nl,dec); Va,Ya=targets(fit); Vb,_=targets(ev)
        o=dict(base=[round(float(np.mean(Vb[:,j]==np.bincount(Va[:,j]).argmax())),3) for j in range(Vb.shape[1])],lin={},lens={},mlp={},curve={},sites=N)
        for s,nm in enumerate(N):
            for l in range(13):
                o['lin'][f'{nm}|{l}']=lin(Xa[:,s,l],Ya,Va,Xb[:,s,l],Vb); o['lens'][f'{nm}|{l}']=lens(Xb[:,s,l],l,Vb)
        print(time.strftime('%H:%M'),f'L{L} linear done',flush=True)
        for nm,l in MLPC:
            if nm in N: o['mlp'][f'{nm}|{l}']=mlp(Xa[:,N.index(nm),l],Ya,Va,Xb[:,N.index(nm),l],Vb)
        cell=(N.index('h3') if nl else N.index('colon'),12)
        for n in ((100,200,300) if SMOKE else (1000,2000,4000)):
            o['curve'][n]=dict(lin=lin(Xa[:n,cell[0],cell[1]],Ya[:n],Va[:n],Xb[:,cell[0],cell[1]],Vb),mlp=mlp(Xa[:n,cell[0],cell[1]],Ya[:n],Va[:n],Xb[:,cell[0],cell[1]],Vb))
        out[L]=o; del Xa,Xb; print(time.strftime('%H:%M'),f'L{L} done',flush=True)
    del m
    if DEV=='mps': torch.mps.empty_cache()
    return out
PATH=OUT+('/b20_smoke.json' if SMOKE else '/b20_grid.json'); RES=json.load(open(PATH)) if os.path.exists(PATH) and not SMOKE else {}
for name,ck,nl,last in MODELS:
    if name in RES: continue
    t0=time.time(); RES[name]=measure(ck,nl,last); json.dump(RES,open(PATH,'w')); print('##',name,f'{time.time()-t0:.0f}s',flush=True)
print('DONE')
