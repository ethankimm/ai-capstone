"""Model adapters for core.run / core.route_split. Run with the CODI venv (torch 2.7, transformers 4.52):

    ../codi/.venv/bin/python scripts/necessity/<script>.py ...

Each adapter: begin(rows) -> (ctx, past, z [B, d]); latent(ctx, past, z, s) -> (past, z_next); pos(ctx, s) -> cache position of latent
pass s; finish(ctx, past, hide) -> answers; n_latents; batch_size.

- RuleChainCODI: our RuleChain CODI checkpoints (gated / sim / start-last / relay / stab), batched with pad-invariant positions,
  single-token numeric answer. Line-for-line t15_rc_route_split.py's run(), so t15 numbers reproduce.
- GSM8KCODI: released zen-E/CODI-gpt2, batch 1, greedy decode, extract_final_number (t12's run()).
- GridCODI: Dilgren & Wiegreffe connordilgren/gpt2-{prosqa,prontoqa}-codi via the grid repo's CODIInference, batch 1, greedy, text after
  "The answer is:" (logic_codi_lens_collapse.py's loop).
- GridCoconut: connordilgren/gpt2-{prosqa,prontoqa,gsm8k}-coconut. Question + <|start-latent|>; latent pass j takes the last hidden state
  (post ln_f, no projection) as input, as Coconut.forward splices it into the <|latent|> position; then <|end-latent|> and greedy decode,
  answer after "###". Incremental KV-cache form of coconut_common.run_passes (same positions, batch 1).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
BASE = HERE.parents[3]  # folder holding ai-capstone/, codi/, run folders
GRID_REPO = Path(os.environ.get('GRID_REPO', Path.home() / 'Projects/are-lrms-easily-interpretable'))
GRID_CKPT = BASE / 'grid_checkpoints'
GRID_PATHS = {
    ('prosqa', 'codi'): 'checkpoints/codi_trained_models/prosqa_gpt2_latent_baseline/gpt2/ep_40/lr_0.003/seed_11/pytorch_model.bin',
    ('prontoqa', 'codi'): 'checkpoints/codi_trained_models/prontoqa_gpt2_latent_baseline/gpt2/ep_40/lr_0.003/seed_11/pytorch_model.bin',
    ('prosqa', 'coconut'): 'checkpoints/prosqa-coconut/checkpoint_40', ('prontoqa', 'coconut'): 'checkpoints/prontoqa-coconut/checkpoint_36',
    ('prosqa', 'no_cot'): 'checkpoints/prosqa-no_cot/checkpoint_15', ('prontoqa', 'no_cot'): 'checkpoints/prontoqa-no_cot/checkpoint_22',
    ('gsm8k', 'no_cot'): 'checkpoints/gsm-no_cot/checkpoint_25',
    ('prosqa', 'cot'): 'checkpoints/prosqa-cot/checkpoint_25', ('prontoqa', 'cot'): 'checkpoints/prontoqa-cot/checkpoint_25',
    ('gsm8k', 'cot'): 'checkpoints/gsm-cot/checkpoint_25',
}


def grid_path(task, method):
    if (task, method) == ('gsm8k', 'coconut'):
        snap = next((Path.home() / '.cache/huggingface/hub/models--connordilgren--gpt2-gsm8k-coconut/snapshots').iterdir())
        return str(snap / 'checkpoint_33')
    return str(GRID_CKPT / GRID_PATHS[(task, method)])


def _offline():
    os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


# ---------------------------------------------------------------- RuleChain CODI (t15)
class RuleChainCODI:
    batch_size = 32
    n_latents = 6

    def __init__(self, model_key, device):
        sys.path.insert(0, str(REPO / 'scripts/codi_null_audit'))
        import harness  # noqa: chdirs into the CODI checkout and sets up imports
        from src.model import RuleGate
        from eval_codi import PadInvariant, question_position_ids
        self.h, self.RuleGate, self.PadInvariant, self.qpos = harness, RuleGate, PadInvariant, question_position_ids
        ck = dict(gated='codi_runs_l1to7w_startlast_gated/codi/run/checkpoint-17820', sim='codi_runs_l1to7w_startlast_sim/codi/run/checkpoint-16335',
                  startlast='codi_runs_l1to7w_startlast/codi/run/checkpoint-17820', stab='codi_runs_l1to7w_stab/codi/run/checkpoint-17820')
        path = harness.CK['lr3'] if model_key == 'relay' else str(BASE / ck[model_key])
        self.m, self.tok, _ = harness.load(path, self.n_latents, device); self.m.eval()
        self.dev = device; self.gate = model_key == 'gated'; self.body = harness.body_of(self.m)
        self.dec = torch.tensor([self.m.eot_id] + self.tok.encode('The answer is:'), device=device)
        self.num = {self.tok.encode(f"The answer is: {n}")[-1]: n for n in range(100)}; assert len(self.num) == 100

    def begin(self, rows):
        b = self.h.encode([dict(question=r['question'].strip(), steps=['x'], answer='0') for r in rows], self.tok, self.m, self.dev)
        ids = b['encoder_input_ids']; mask = b['encoder_attention_mask'].long(); W = ids.size(1)
        g = self.RuleGate(ids, mask) if self.gate else None
        o = self.body(input_ids=ids, use_cache=True, attention_mask=g.question(torch.float32) if self.gate else mask,
                      position_ids=self.qpos(mask))
        ctx = dict(W=W, g=g, pi=self.PadInvariant(mask), B=ids.size(0))
        return ctx, o.past_key_values, self.m.prj(o.last_hidden_state[:, -1:, :])[:, 0]

    def latent(self, ctx, past, z, s):
        if self.gate: ctx['pi'].mask[:, :ctx['W']] = ctx['g'].latent(s)
        o = self.body(inputs_embeds=z[:, None], use_cache=True, past_key_values=past, **ctx['pi'].next(1))
        return o.past_key_values, self.m.prj(o.last_hidden_state[:, -1:, :])[:, 0]

    def pos(self, ctx, s):
        return ctx['W'] + s

    def finish(self, ctx, past, hide=()):
        pi = ctx['pi']
        if self.gate: pi.mask[:, :ctx['W']] = ctx['g'].answer()
        for s in hide: pi.mask[:, ctx['W'] + s] = 0
        emb = self.m.get_embd(self.m.codi, self.m.model_name)(self.dec)[None].expand(ctx['B'], -1, -1)
        o = self.body(inputs_embeds=emb, use_cache=True, past_key_values=past, **pi.next(emb.size(1)))
        t = self.m.codi.get_base_model().lm_head(o.last_hidden_state[:, -1, :])[:, :50257].argmax(-1).tolist()
        return [self.num.get(x, -999) for x in t]


# ---------------------------------------------------------------- shared batch-1 greedy decoding
def _greedy(body, head, embed, first_emb, past, start_len, hide_pos, vocab, eos, max_new, dev):
    """Greedy decode from first_emb [1, 1, d]; hide_pos = absolute cache positions masked for every answer token."""
    toks = []; nxt = first_emb
    for t in range(max_new):
        kw = {}
        if hide_pos:
            m = torch.ones((1, start_len + t + 1), dtype=torch.long, device=dev)
            for p in hide_pos: m[:, p] = 0
            kw['attention_mask'] = m
        o = body(inputs_embeds=nxt, use_cache=True, past_key_values=past, **kw)
        past = o.past_key_values
        nid = int(head(o.last_hidden_state[0, -1])[:vocab].argmax())
        if nid == eos: break
        toks.append(nid); nxt = embed(torch.tensor([nid], device=dev)).unsqueeze(0)
    return toks


# ---------------------------------------------------------------- released GSM8K CODI (t12)
class GSM8KCODI:
    batch_size = 1
    n_latents = 6

    def __init__(self, device):
        _offline()
        sys.path.insert(0, str(REPO / 'scripts/codi_null_audit'))
        import harness  # noqa: CODI checkout on the path
        import transformers
        from src.model import DataArguments, ModelArguments, TrainingArguments
        from eval_codi import build_model
        from decode_patch_codi import encode_question
        from latentreasoning.eval.metrics import extract_final_number
        ck = str(Path.home() / '.cache/huggingface/hub/models--zen-E--CODI-gpt2/snapshots/fd641b3d3edc59e4f534b55588e906588c9e36bb')
        argv = (f"--output_dir /tmp/u --model_name_or_path gpt2 --seed 11 --model_max_length 512 --lora_r 128 --lora_alpha 32 --lora_init "
                f"--greedy True --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 "
                f"--inf_num_iterations 1 --remove_eos True --use_lora True --ckpt_dir {ck}").split()
        ma, da, ta = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses(args=argv)
        ta.print_ref_model_stats = False; ta.print_loss = False
        self.m, self.tok, _ = build_model(ma, ta, device=device, dtype=torch.float32); self.m.eval()
        self.enc, self.xnum, self.dev = encode_question, extract_final_number, device
        self.emb = self.m.get_embd(self.m.codi, self.m.model_name)

    def begin(self, rows):
        (r,) = rows
        ids, attn = self.enc(self.m, self.tok, r['question'].strip().replace('  ', ' '), self.dev); Q = ids.size(1)
        o = self.m.codi(input_ids=ids, use_cache=True, output_hidden_states=True, past_key_values=None, attention_mask=attn)
        return dict(Q=Q), o.past_key_values, self.m.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1))[:, 0]

    def latent(self, ctx, past, z, s):
        m = torch.ones((1, ctx['Q'] + s + 1), dtype=torch.long, device=self.dev)
        o = self.m.codi(inputs_embeds=z[:, None], use_cache=True, output_hidden_states=True, past_key_values=past, attention_mask=m)
        return o.past_key_values, self.m.prj(o.hidden_states[-1][:, -1, :].unsqueeze(1))[:, 0]

    def pos(self, ctx, s):
        return ctx['Q'] + s

    def finish(self, ctx, past, hide=()):
        out = self.emb(torch.tensor([self.m.eot_id], device=self.dev)).unsqueeze(0); toks = []; L0 = ctx['Q'] + self.n_latents
        for t in range(64):
            kw = {}
            if hide:
                m = torch.ones((1, L0 + t + 1), dtype=torch.long, device=self.dev)
                for h in hide: m[:, ctx['Q'] + h] = 0
                kw['attention_mask'] = m
            o = self.m.codi(inputs_embeds=out, use_cache=True, past_key_values=past, **kw)
            past = o.past_key_values; nid = torch.argmax(o.logits[:, -1, :self.m.codi.config.vocab_size - 1], dim=-1)
            if nid.item() == self.tok.eos_token_id: break
            toks.append(nid.item()); out = self.emb(nid).unsqueeze(1)
        return [self.xnum(self.tok.decode(toks, skip_special_tokens=True))]


# ---------------------------------------------------------------- Dilgren & Wiegreffe grid CODI (ProsQA / ProntoQA)
def norm_text(s):
    return (s or '').strip().rstrip('.').lower().replace(',', '')


class GridCODI:
    batch_size = 1
    DELIM = 'The answer is:'

    def __init__(self, task, device, n_latents=6, max_new=32):
        _offline(); sys.path.insert(0, str(GRID_REPO))
        from models.codi.codi_wrapper import CODIInference
        self.n_latents, self.max_new, self.dev = n_latents, max_new, device
        m = CODIInference(model_name_or_path='gpt2', num_latent=6, use_prj=True, prj_dim=768, use_lora=True, lora_r=128, lora_alpha=32, device=device)
        m.load_checkpoint(grid_path(task, 'codi')); self.m = m.to(device).eval().float(); self.tok = m.tokenizer
        self.embed = m.get_embd(m.codi, m.model_name); lm = m.codi.get_base_model()
        self.body, self.head = lm.transformer, lm.lm_head; self.vocab = m.codi.config.vocab_size - 1

    def qids(self, q):
        enc = self.tok(q.strip().replace('  ', ' '), return_tensors='pt', add_special_tokens=True)
        return torch.cat([enc['input_ids'], torch.tensor([[self.m.bot_id]])], 1)

    def begin(self, rows):
        (r,) = rows
        ids = self.qids(r['question']).to(self.dev)
        o = self.body(input_ids=ids, use_cache=True)
        return dict(Q=ids.size(1)), o.past_key_values, self.m.prj(o.last_hidden_state[:, -1:, :])[:, 0]

    def latent(self, ctx, past, z, s):
        o = self.body(inputs_embeds=z[:, None], use_cache=True, past_key_values=past)
        return o.past_key_values, self.m.prj(o.last_hidden_state[:, -1:, :])[:, 0]

    def pos(self, ctx, s):
        return ctx['Q'] + s

    def finish(self, ctx, past, hide=()):
        first = self.embed(torch.tensor([self.m.eot_id], device=self.dev)).unsqueeze(0)
        toks = _greedy(self.body, self.head, self.embed, first, past, ctx['Q'] + self.n_latents, [ctx['Q'] + h for h in hide],
                       self.vocab, self.tok.eos_token_id, self.max_new, self.dev)
        raw = self.tok.decode([t for t in toks if t < len(self.tok)], skip_special_tokens=True)
        return [norm_text(raw.split(self.DELIM)[-1]) if self.DELIM in raw else '']

    def zero_latent_answer(self, row):
        """question + <bot> + <eot> (off-distribution truncation, as 20261103-021023)."""
        ids = self.qids(row['question']).to(self.dev)
        o = self.body(input_ids=ids, use_cache=True)
        first = self.embed(torch.tensor([self.m.eot_id], device=self.dev)).unsqueeze(0)
        toks = _greedy(self.body, self.head, self.embed, first, o.past_key_values, ids.size(1), [], self.vocab, self.tok.eos_token_id,
                       self.max_new, self.dev)
        raw = self.tok.decode([t for t in toks if t < len(self.tok)], skip_special_tokens=True)
        return norm_text(raw.split(self.DELIM)[-1]) if self.DELIM in raw else ''


# ---------------------------------------------------------------- Dilgren & Wiegreffe grid Coconut / no-CoT
class GridCoconut:
    """method='coconut': latent loop as above. For method in ('no_cot', 'cot') use plain_answer() only (no latents)."""
    batch_size = 1
    n_latents = 6

    def __init__(self, task, device, method='coconut', max_new=48):
        _offline(); sys.path.insert(0, str(REPO / 'scripts'))
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.task, self.dev, self.max_new, self.method = task, device, max_new, method
        tok = AutoTokenizer.from_pretrained('openai-community/gpt2'); tok.pad_token = tok.eos_token
        if method == 'coconut':
            for t in ('<|start-latent|>', '<|end-latent|>', '<|latent|>'): tok.add_tokens(t)
        lm = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
        state = torch.load(grid_path(task, method), map_location='cpu')
        sd = {k.removeprefix('base_causallm.'): v for k, v in state.items() if k.startswith('base_causallm.')} or state
        lm.resize_token_embeddings(sd['transformer.wte.weight'].size(0))
        missing, unexpected = lm.load_state_dict(sd, strict=False)
        assert not [k for k in missing if 'attn.bias' not in k and 'masked_bias' not in k and k != 'lm_head.weight'], missing[:5]
        self.lm = lm.to(device).eval().float(); self.tok = tok; self.body, self.head = lm.transformer, lm.lm_head
        self.embed = lm.transformer.wte; self.vocab = sd['transformer.wte.weight'].size(0)
        if method == 'coconut':
            self.sl, self.el = tok.convert_tokens_to_ids('<|start-latent|>'), tok.convert_tokens_to_ids('<|end-latent|>')

    def qids(self, q):
        q = q if q.endswith('\n') else q + '\n'
        return self.tok.encode(q, add_special_tokens=True)

    def extract(self, raw):
        if '###' not in raw: return ''
        a = raw.split('###')[-1].strip()
        return a.replace(',', '') if self.task == 'gsm8k' else norm_text(a)

    def begin(self, rows):
        (r,) = rows
        ids = torch.tensor([self.qids(r['question']) + [self.sl]], device=self.dev)
        o = self.body(input_ids=ids, use_cache=True)
        return dict(Q=ids.size(1)), o.past_key_values, o.last_hidden_state[:, -1, :]

    def latent(self, ctx, past, z, s):
        o = self.body(inputs_embeds=z[:, None], use_cache=True, past_key_values=past)
        return o.past_key_values, o.last_hidden_state[:, -1, :]

    def pos(self, ctx, s):
        return ctx['Q'] + s

    def finish(self, ctx, past, hide=()):
        first = self.embed(torch.tensor([self.el], device=self.dev)).unsqueeze(0)
        toks = _greedy(self.body, self.head, self.embed, first, past, ctx['Q'] + self.n_latents, [ctx['Q'] + h for h in hide],
                       self.vocab, self.tok.eos_token_id, self.max_new, self.dev)
        return [self.extract(self.tok.decode(toks, skip_special_tokens=True))]

    @torch.no_grad()
    def plain_answer(self, row, n_latents=None):
        """No latents: no-CoT / CoT checkpoints (question then greedy), or this Coconut model with fewer passes (n_latents < 6: question,
        <start-latent>, n passes, <end-latent>)."""
        if self.method != 'coconut':
            ids = torch.tensor([self.qids(row['question'])], device=self.dev)
            o = self.body(input_ids=ids[:, :-1], use_cache=True)
            first = self.embed(ids[:, -1:])
            toks = _greedy(self.body, self.head, self.embed, first, o.past_key_values, ids.size(1) - 1, [], self.vocab,
                           self.tok.eos_token_id, self.max_new if self.method == 'no_cot' else 256, self.dev)
            return self.extract(self.tok.decode(toks, skip_special_tokens=True))
        ctx, past, z = self.begin([row])
        for s in range(n_latents):
            past, z = self.latent(ctx, past, z, s)
        first = self.embed(torch.tensor([self.el], device=self.dev)).unsqueeze(0)
        toks = _greedy(self.body, self.head, self.embed, first, past, ctx['Q'] + n_latents, [], self.vocab, self.tok.eos_token_id,
                       self.max_new, self.dev)
        return self.extract(self.tok.decode(toks, skip_special_tokens=True))
