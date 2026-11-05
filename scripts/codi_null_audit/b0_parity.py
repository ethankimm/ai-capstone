import sys; sys.path.insert(0,sys.argv[1])
from harness import *
torch.manual_seed(0)
T=BASE+'/synthetic_data_v2_l1to7w/depth2/train.json'
tr=rows(T,per_level=3,seed=1)
for name,ck,nl in [('stab',CK['stab']%17820,6),('stab0',CK['stab0']%17820,0)]:
    m,tok,ta=load(ck,nl); m.distill_loss_div_std=True
    b=encode(tr,tok,m,'cpu')
    # token-level alignment audit
    for i in (0,10,20):
        rp=b['ref_answer_position'][i].item(); mp=b['model_answer_position'][i].item()
        print(repr(tok.decode(b['ref_input_ids'][i,rp-1:rp+1].tolist())), repr(tok.decode(b['decoder_input_ids'][i,mp-1:mp+1].tolist())), 'gold',tr[i]['answer'])
    with torch.no_grad():
        up=m(**b)                                     # upstream CODI.forward (patched), eval mode
        out,mask,pos=question(m,b)
        past=out.past_key_values; z=m.prj(out.last_hidden_state[:,-1:,:])
        for s in range(nl):
            h,z,past,mask,pos=latent_step(m,z,past,mask,pos)
        o=decode(m,b,past,mask,pos); t=teacher(m,b)
        ce,dist,per=losses(m,b,o,t)
    print(name,'upstream ce %.6f distill %.6f ref %.6f | mine ce %.6f distill %.6f'%(up['ce_loss'],up['distill_loss'],up['ref_ce_loss'],ce,dist))
    print(' per-layer distill',[round(float(x),3) for x in per])
    print(' trained params in ckpt: prj first-layer weight norm',float(m.prj[1].weight.norm()),'bias',float(m.prj[1].bias.norm()), 'prj ln w mean',float(m.prj.ln.weight.mean()))
