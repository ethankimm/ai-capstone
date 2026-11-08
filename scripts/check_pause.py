#!/usr/bin/env python3
"""Checks the pause-latent control (TRAINING_PROTOCOL section 15, scripts/codi_pause.patch) on a real start-last checkpoint,
CPU, fp32. Exit 1 on any failure.
  1. recurrence cut: with --pause_latents, the latent inputs are pause[0..5] whatever the question says; ungated they move.
  2. flag off: the latent inputs and answer logits are bit-identical to the unpatched checkout's (--ref-codi).
  3. train = eval: CODI.forward's answer logits equal the eval path's, under the flag.
  4. gradients: finite, reach `pause` and the LoRA weights; prj gets none (it feeds nothing).
  5. padding: an item alone and in a padded batch give the same latents and the same answer logits.
Usage (from anywhere; CODI_DIR = a CODI checkout with codi_pause.patch applied):
  CODI_DIR=<patched> python scripts/check_pause.py <checkpoint> <train json> [--ref-codi <unpatched checkout>]"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
R = Path(__file__).resolve().parent
CODI_DIR = os.environ.get("CODI_DIR", str(R.parents[1] / "codi"))
for k, v in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("TOKENIZERS_PARALLELISM", "false")): os.environ.setdefault(k, v)
os.chdir(CODI_DIR); sys.path[:0] = [CODI_DIR, str(R)]
import torch, transformers  # noqa: E402
from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402
import diagnose_codi_latents  # noqa: E402,F401 (patches torch.load for CPU)
from eval_codi import build_model, encode_and_think, tokenize_questions  # noqa: E402
sys.path.insert(0, str(R / "codi_null_audit"))
args = [a for a in sys.argv[1:]]
ref = args.pop(args.index("--ref-codi") + 1) if "--ref-codi" in args else None
if ref: args.remove("--ref-codi")
dump = args.pop(args.index("--dump") + 1) if "--dump" in args else None
if dump: args.remove("--dump")
CK, DATA = args[0], args[1]

def load(pause, patched=True):
    argv = (f"--output_dir /tmp/u --model_name_or_path gpt2 --seed 11 --model_max_length 512 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 64 --greedy True "
            f"--num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True "
            f"--use_lora True --fix_attn_mask True --pad_invariant True --include_last_cot False --ckpt_dir {CK}" + (" --pause_latents True" if pause else "")).split()
    ma, da, ta = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses(args=argv)
    ta.print_ref_model_stats = False; ta.print_loss = False
    m, tok, lr = build_model(ma, ta, device="cpu", dtype=torch.float32)
    bad = [k for k in lr.missing_keys if not k.startswith("codi.base_model.model.lm_head") and k != "pause"]
    assert not bad, bad[:5]
    m.training_args = ta; m.print_loss = False; m.eval(); ta.batch_size = 64; ta.inf_latent_iterations = 6
    return m, tok, ta

rows_ = json.load(open(DATA)); by = {}
for r in rows_: by.setdefault(int(r["level"]), []).append(r)
def batch(m, tok, ta, qs): return tokenize_questions(m, tok, [q.strip() + "\n" for q in qs], len(qs), ta, "cpu")[0]
def think(m, tok, ta, qs):
    with torch.no_grad(): p, pi, lat = encode_and_think(m, batch(m, tok, ta, qs), ta)
    dec = torch.tensor([m.eot_id] + tok.encode("The answer is:")); emb = m.get_embd(m.codi, m.model_name)(dec)[None].expand(len(qs), -1, -1)
    with torch.no_grad(): lg = m.codi(inputs_embeds=emb, use_cache=True, past_key_values=p, **pi.next(len(dec))).logits[:, -1]
    return lat, lg

if dump:  # subprocess mode for check 2: flag off
    m, tok, ta = load(False); qs = [by[L][i]["question"] for L in (1, 3, 7) for i in range(2)]
    lat, lg = think(m, tok, ta, qs); torch.save({"lat": [x.clone() for x in lat], "lg": lg}, dump); sys.exit(0)

torch.manual_seed(0); fails = []
def check(name, ok, info=""):
    print(("ok  " if ok else "FAIL"), name, info, flush=True)
    if not ok: fails.append(name)
m, tok, ta = load(True)
with torch.no_grad(): m.pause.add_(0.5 * torch.randn_like(m.pause))  # distinct rows, as after training
check("pause is a trainable fp32 parameter of shape [6, 768]", isinstance(m.pause, torch.nn.Parameter) and tuple(m.pause.shape) == (6, 768) and m.pause.dtype == torch.float32 and m.pause.requires_grad)
# 1 recurrence cut
qs = [by[L][i]["question"] for L in (1, 4, 7) for i in range(2)]
lat, _ = think(m, tok, ta, qs)
check("latent inputs are exactly pause[i], whatever the question", all(torch.equal(lat[i], m.pause[i].expand(len(qs), 1, -1)) for i in range(6)))
q = by[4][0]["question"]; q2 = q.replace("Start with", "Start with 7 and then ignore") if False else q.rsplit("Start with", 1)[0] + "Start with 99. What is the final number?"
a, _ = think(m, tok, ta, [q]); b, _ = think(m, tok, ta, [q2])
check("editing the question changes no latent input", all(torch.equal(x, y) for x, y in zip(a, b)))
m.pause_latents = False; ta.pause_latents = False
a2, _ = think(m, tok, ta, [q]); b2, _ = think(m, tok, ta, [q2])
check("positive control: flag off, the same edit moves a latent input", any(not torch.equal(x, y) for x, y in zip(a2, b2)))
check("flag off: latent 0 is prj of the <bot> state, not pause[0]", not torch.equal(a2[0], m.pause[0].expand(1, 1, -1)))
m.pause_latents = True; ta.pause_latents = True
# 2 flag off == unpatched
if ref:
    with tempfile.TemporaryDirectory() as td:
        f1, f2 = td + "/ref.pt", td + "/new.pt"
        for d, f in ((ref, f1), (CODI_DIR, f2)):
            subprocess.run([sys.executable, __file__, CK, DATA, "--dump", f], env={**os.environ, "CODI_DIR": d}, check=True)
        A, B = torch.load(f1), torch.load(f2)
        check("flag off: latent inputs and answer logits bit-identical to the unpatched checkout", all(torch.equal(x, y) for x, y in zip(A["lat"], B["lat"])) and torch.equal(A["lg"], B["lg"]))
else: print("skip flag-off identity (no --ref-codi)")
# 3 train = eval
from harness import encode  # noqa: E402
xs = [by[L][i] for L in (1, 2, 4, 7) for i in range(2)]
enc = encode([dict(question=x["question"].strip(), steps=x["steps"], answer=x["answer"]) for x in xs], tok, m, "cpu")
keys = ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels", "encoder_attention_mask", "ref_answer_position", "model_answer_position", "ref_attention_mask", "ref_labels")
with torch.no_grad():
    o = m(**{k: v for k, v in enc.items() if k in keys}, step=0, step_ratio=0.0)
    b = {"input_ids": enc["encoder_input_ids"], "attention_mask": enc["encoder_attention_mask"].long()}; p, pi, _ = encode_and_think(m, b, ta)
    dec = torch.tensor([m.eot_id] + tok.encode("The answer is:")); emb = m.get_embd(m.codi, m.model_name)(dec)[None].expand(len(xs), -1, -1)
    e_log = m.codi(inputs_embeds=emb, use_cache=True, past_key_values=p, **pi.next(len(dec))).logits[:, -1]
d = float((o["logits"][:, len(dec) - 1] - e_log).abs().max()); check("train and eval answer logits match", d < 1e-2, f"max diff {d:.2e}")
# 4 gradients
m.train(); m.zero_grad(); o = m(**{k: v for k, v in enc.items() if k in keys}, step=0, step_ratio=0.0); o["loss"].backward()
g = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
lora = [x.norm().item() for n, x in g.items() if "lora" in n]; gp = m.pause.grad
check("loss finite; gradient reaches every pause row and the LoRA weights", bool(torch.isfinite(o["loss"])) and gp is not None and bool((gp.abs().sum(1) > 0).all()) and max(lora) > 0 and all(torch.isfinite(x).all() for x in g.values()),
      f"loss {float(o['loss']):.3f} pause grad row norms {[round(float(r), 3) for r in gp.norm(dim=1)]}")
check("prj gets no gradient (it feeds nothing)", not any("prj" in n and x.abs().sum() > 0 for n, x in g.items()))
m.eval()
# 5 padding
qs = [by[1][0]["question"], by[3][0]["question"], by[7][0]["question"]]; A, LA = think(m, tok, ta, qs); mx = ml = 0
for i, q in enumerate(qs):
    s, ls = think(m, tok, ta, [q]); mx = max(mx, max(float((A[t][i] - s[t][0]).abs().max()) for t in range(6))); ml = max(ml, float((LA[i] - ls[0]).abs().max()))
check("padding: batched equals alone (latent inputs, answer logits)", mx < 1e-3 and ml < 1e-2, f"max diff {mx:.2e} / {ml:.2e}")
print("FAILED:" if fails else "ALL PASSED", fails); sys.exit(1 if fails else 0)
