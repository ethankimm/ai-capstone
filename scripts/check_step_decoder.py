#!/usr/bin/env python3
"""Checks the step decoder (scripts/codi_step_decoder.patch, TRAINING_PROTOCOL section 13) before training. Exit 1 on any failure.
  1. parser: on every train item, teacher_steps gives exactly steps[:-1], tokenised alone, each ending in EOS; at most 6 per item.
  2. loss: finite, above 0, one entry per latent that has a step.
  3. gradients: the step loss alone reaches the decoder, the projection and the LoRA weights of the first block, finite.
  4. mapping: an item with one step puts gradient on latent 0 only; two steps on latents 0 and 1.
  5. answer path: with dropout off, the answer logits are bit-identical with the decoder on and off, and the loss differs only by
     the step loss.
Usage (from a CODI checkout with the patch, its venv): python ../ai-capstone/scripts/check_step_decoder.py <train.json>"""
import json, os, sys
import torch, transformers
from peft import LoraConfig, TaskType
sys.path.insert(0, os.getcwd())
from src.model import CODI, DataArguments, ModelArguments, TrainingArguments, teacher_steps  # noqa: E402
DATA = sys.argv[1]; fails = []
def check(name, ok, info=""):
    print(("ok  " if ok else "FAIL"), name, info, flush=True)
    if not ok: fails.append(name)
rows = json.load(open(DATA))
tok = transformers.AutoTokenizer.from_pretrained("gpt2")
PAD, EOS = 50257, tok.eos_token_id
def teacher(r):
    cot = "\n".join(r["steps"][:-1]) + "\n"
    return tok(r["question"].strip() + "\n")["input_ids"] + tok(cot)["input_ids"] + tok(f"The answer is: {str(r['answer']).strip()}")["input_ids"] + [EOS]
# 1 parser
bad, most = 0, 0
for i in range(0, len(rows) if not os.environ.get("QUICK") else 2048, 512):
    chunk = rows[i:i + 512]; seqs = [teacher(r) for r in chunk]; n = max(map(len, seqs))
    got = teacher_steps(torch.tensor([s + [PAD] * (n - len(s)) for s in seqs]), EOS)
    for r, g in zip(chunk, got):
        want = [tok(s)["input_ids"] + [EOS] for s in r["steps"][:-1]]
        bad += g != want; most = max(most, len(g))
check("parser equals steps[:-1] on every item", bad == 0, f"{len(rows)} items, {bad} wrong, most steps {most}")
check("at most 6 steps", most <= 6)
# model
ma, da, ta = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses(args=[
    "--output_dir", "/tmp/u", "--model_name_or_path", "gpt2", "--lora_r", "128", "--lora_alpha", "32", "--lora_init", "--use_lora", "True", "--use_cpu", "True",
    "--num_latent", "6", "--use_prj", "True", "--prj_dim", "768", "--prj_no_ln", "False", "--prj_dropout", "0.0", "--remove_eos", "True", "--fix_attn_mask", "True",
    "--pad_invariant", "True", "--include_last_cot", "False", "--distill_loss_div_std", "True", "--step_decoder", "True", "--bf16", "False"])
ma.train = True; ta.bf16 = True  # the training dtype; the args above only avoid the Trainer's device check
lora = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=128, lora_alpha=32, lora_dropout=0.1, target_modules=["c_attn", "c_proj", "c_fc"], init_lora_weights=True)
torch.manual_seed(0)
m = CODI(ma, ta, lora)
BOT, EOT = m.bot_id, m.eot_id
by = {}
for r in rows: by.setdefault(int(r["level"]), []).append(r)
items = [by[L][i] for L in range(1, 8) for i in range(2)]
def pad_left(s, n): return [PAD] * (n - len(s)) + s
def pad_right(s, n, v=PAD): return s + [v] * (n - len(s))
def batch(items):
    src = [tok(r["question"].strip() + "\n")["input_ids"] + [BOT] for r in items]
    ans = [[EOT] + tok(f"The answer is: {str(r['answer']).strip()}")["input_ids"] + [EOS] for r in items]
    ref = [teacher(r) for r in items]
    refl = [[-100] * (len(tok(r["question"].strip() + "\n")["input_ids"])) + s[len(tok(r["question"].strip() + "\n")["input_ids"]):] for r, s in zip(items, ref)]
    pre = tok("The answer is:")["input_ids"]
    def ans_pos(s): return next(i for i in range(len(s)) if s[i:i + len(pre)] == pre) + len(pre)
    a, b, c = max(map(len, src)), max(map(len, ans)), max(map(len, ref))
    enc = torch.tensor([pad_left(s, a) for s in src]); ri = torch.tensor([pad_right(s, c) for s in ref])
    return dict(encoder_input_ids=enc, decoder_input_ids=torch.tensor([pad_right(s, b) for s in ans]), ref_input_ids=ri,
                labels=torch.tensor([pad_right(s, b, -100) for s in ans]), encoder_attention_mask=enc.ne(PAD), ref_attention_mask=ri.ne(PAD),
                ref_answer_position=torch.tensor([ans_pos(s) for s in ref]), model_answer_position=torch.tensor([ans_pos(s) for s in ans]),
                ref_labels=torch.tensor([pad_right(s, c, -100) for s in refl]))
b = batch(items)
# 2, 3 loss and gradients
saved = {}; orig = m.explain_loss
def spy(lat, st):
    saved["lat"], saved["steps"] = lat, st
    saved["loss"], per = orig(lat, st); return saved["loss"], per
m.explain_loss = spy
amp = lambda: torch.autocast("cpu", dtype=torch.bfloat16)  # as the Trainer runs it under --bf16
m.train()
with amp(): out = m(**b)
n_steps = max(len(s) for s in saved["steps"])
check("step loss finite and above 0", 0 < out["explain_loss"] < 100, f"{out['explain_loss']:.3f}")
check("one entry per latent with a step", len(out["explain_by_latent"]) == n_steps == 6, str([round(x, 2) for x in out["explain_by_latent"]]))
check("latents collected", len(saved["lat"]) == 7)
ps = dict(m.named_parameters())
lora_p = next(p for k, p in ps.items() if "h.0." in k and "lora_B" in k and "codi" in k)  # B: A has zero gradient at init, since B starts at 0
tg = {"decoder": m.decoder.transformer.h[0].mlp.c_fc.weight, "prj": next(m.prj.parameters()), "lora": lora_p}
gs = torch.autograd.grad(saved["loss"], list(tg.values()), retain_graph=True, allow_unused=True)
for k, g in zip(tg, gs):
    check(f"step loss reaches {k}", g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, "" if g is None else f"|g| {g.norm():.3g}")
# 4 mapping
m.eval()
two = next(r for r in rows if int(r["level"]) == 3); one = next(r for r in rows if int(r["level"]) == 2)
for r, want in ((one, 1), (two, 2)):
    st = teacher_steps(torch.tensor([teacher(r)]), EOS)
    lat = [torch.randn(1, 1, 768, requires_grad=True) for _ in range(7)]
    with amp(): l, per = orig(lat, st)
    g = torch.autograd.grad(l, lat, allow_unused=True)
    hit = [i for i, x in enumerate(g) if x is not None and x.abs().sum() > 0]
    check(f"{want} step(s) touch latents {list(range(want))}", hit == list(range(want)), str(hit))
# 5 answer path
with torch.no_grad(), amp():
    torch.manual_seed(1); on = m(**b)
    m.step_decoder = False; torch.manual_seed(1); off = m(**b); m.step_decoder = True
check("answer logits identical", torch.equal(on["logits"], off["logits"]))
d = float(on["loss"] - off["loss"]) - on["explain_loss"]
check("loss differs by the step loss only", abs(d) < 1e-3 * max(1, on["explain_loss"]), f"{d:.2e}")
print("FAILED: " + ", ".join(fails) if fails else "ALL OK"); sys.exit(1 if fails else 0)
