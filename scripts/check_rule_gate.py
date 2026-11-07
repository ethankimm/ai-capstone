#!/usr/bin/env python3
"""Checks the rule gate (TRAINING_PROTOCOL section 12) on a real start-last checkpoint, CPU, fp32. Exit 1 on any failure.
  1. groups: every rule's tokens decode to that rule, the tail starts at " Start".
  2. leaks: changing rule j leaves every latent before pass j-1's output bit-identical, and leaves the keys and values of
     all other rules bit-identical. Ungated, the same edit moves z_0, which shows the test can see a leak.
  3. padding: an item alone and in a padded batch give the same latents.
  4. train = eval: CODI.forward's answer logits equal the eval path's.
  5. gradients: finite, and reach the projection through the latents.
Usage (from the CODI checkout, its venv): python ../ai-capstone/scripts/check_rule_gate.py <checkpoint> <data json>"""
import json, re, sys
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R / "codi_null_audit"))
from harness import *  # noqa: E402,F403 (chdirs to the CODI checkout)
from eval_codi import encode_and_think, tokenize_questions  # noqa: E402
from src.model import RuleGate  # noqa: E402
CK, DATA = sys.argv[1], sys.argv[2]
torch.manual_seed(0); fails = []
def check(name, ok, info=""):
    print(("ok  " if ok else "FAIL"), name, info, flush=True)
    if not ok: fails.append(name)
m, tok, ta = load(CK, 6, "cpu"); m.eval(); ta.batch_size = 64; ta.inf_latent_iterations = 6
rows_ = json.load(open(DATA)); by = {}
for r in rows_: by.setdefault(int(r["level"]), []).append(r)
def batch(qs):
    return tokenize_questions(m, tok, [q.strip() + "\n" for q in qs], len(qs), ta, "cpu")[0]
def zs(qs, gate):
    ta.rule_gate = gate; m.rule_gate = gate
    with torch.no_grad(): p, pi, lat = encode_and_think(m, batch(qs), ta)
    return lat, p, pi
def kv(p):
    return [(a, b) for a, b in (p if isinstance(p, tuple) else p.to_legacy_cache())]
# 1 groups
for L in (1, 3, 7):
    q = by[L][0]["question"].strip(); b = batch([q]); g = RuleGate(b["input_ids"], b["attention_mask"]); ids = b["input_ids"][0]
    parts = [tok.decode(ids[g.g[0] == j].tolist()).strip() for j in range(1, L + 1)]
    want = re.findall(r"Rule \d+: [^.]*\.", q)
    check(f"groups L{L}", parts == want and int(g.n) == L and tok.decode(ids[g.g[0] == 0][:2].tolist()).strip() == "Start with" and int(g.sink.sum()) == 1, parts[0][:30] if parts else "")
# 2 leaks
def edit(q, j):
    """Change a constant of rule j, keeping the token count."""
    parts = re.split(r"(?=Rule \d+:)", q); k = j; s = parts[k]
    body, sep, rest = s.partition(" Start with")   # the last rule's segment also holds the start value, which is not part of the rule
    n = re.findall(r"\d+", body.split(":", 1)[1])[-1]; new = str(int(n) + 1 if int(n) % 10 != 9 else int(n) - 1)
    head, tail = body.rsplit(n, 1); parts[k] = head + new + tail + sep + rest; out = "".join(parts)
    assert len(tok(out).input_ids) == len(tok(q).input_ids) and out != q; return out
L = 5; ok_all = True; counts = {}
for r in by[L][:12]:
    q = r["question"].strip(); a, pa, _ = zs([q], True); W = len(tok(q + "\n").input_ids) + 1; g = RuleGate(*(lambda b: (b["input_ids"], b["attention_mask"]))(batch([q])))
    for j in range(1, L + 1):
        e = edit(q, j); b_, pb, _ = zs([e], True)
        same = [bool(torch.equal(a[i], b_[i])) for i in range(6)]
        want = [i <= j - 1 or j == L for i in range(6)]   # z_i is the input to pass i; rule j enters at pass j-1, so z_j is the first it can move
        kvsame = all(torch.equal(x[..., (g.g[0] != j).nonzero().flatten()[:], :][..., :, :][:, :, :, :], y[..., (g.g[0] != j).nonzero().flatten()[:], :]) for (x, _), (y, _) in zip(kv(pa), kv(pb)))
        moved = any(not s for s in same) or j == L
        if not (all(s for s, w in zip(same, want) if w) and moved): ok_all = False; print("  leak?", j, same)
        counts[j] = counts.get(j, 0) + (not all(same))
        if not kvsame: ok_all = False; print("  kv leak", j)
check("no leak: latents before a rule's pass are identical, other rules' keys unchanged", ok_all, f"items where a latent moved, by rule edited: {counts}")
check("rules 1 to L-1 do reach the latents; the last rule does not", all(counts[j] > 0 for j in range(1, L)) and counts[L] == 0)
q = by[L][0]["question"].strip(); e = edit(q, 3); a, _, _ = zs([q], False); b_, _, _ = zs([e], False)
check("positive control: ungated, the same edit moves z_0", not torch.equal(a[0], b_[0]), f"max diff {float((a[0]-b_[0]).abs().max()):.3f}")
a, _, _ = zs([q], True); e = q.replace("Start with", "Start  with") if False else re.sub(r"Start with (\d+)", lambda x: f"Start with {int(x.group(1)) % 90 + 1}", q)
b_, _, _ = zs([e], True); check("start value reaches z_0", not torch.equal(a[0], b_[0]))
# 3 padding
qs = [by[1][0]["question"], by[3][0]["question"], by[7][0]["question"]]; A, _, _ = zs(qs, True); mx = 0
for i, q in enumerate(qs):
    s, _, _ = zs([q], True); mx = max(mx, max(float((A[t][i] - s[t][0]).abs().max()) for t in range(6)))
check("padding: batched equals alone", mx < 1e-3, f"max diff {mx:.2e}")
# 4 train = eval
xs = [by[L][i] for L in (1, 2, 4, 7) for i in range(2)]; enc = encode([dict(question=x["question"].strip(), steps=x["steps"], answer=x["answer"]) for x in xs], tok, m, "cpu")
ta.rule_gate = True; m.rule_gate = True; m.training_args.print_ref_model_stats = False
with torch.no_grad():
    o = m(**{k: v for k, v in enc.items() if k in ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels", "encoder_attention_mask", "ref_answer_position", "model_answer_position", "ref_attention_mask", "ref_labels")}, step=0, step_ratio=0.0)
    b = {"input_ids": enc["encoder_input_ids"], "attention_mask": enc["encoder_attention_mask"].long()}; p, pi, _ = encode_and_think(m, b, ta)
    dec = torch.tensor([m.eot_id] + tok.encode("The answer is:")); emb = m.get_embd(m.codi, m.model_name)(dec)[None].expand(len(xs), -1, -1)
    e_log = m.codi(inputs_embeds=emb, use_cache=True, past_key_values=p, **pi.next(len(dec))).logits[:, -1]
t_log = o["logits"][:, len(dec) - 1]
d = float((t_log - e_log).abs().max()); check("train and eval answer logits match", d < 1e-2, f"max diff {d:.2e}")
# 5 gradients
m.train(); m.zero_grad(); o = m(**{k: v for k, v in enc.items() if k in ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels", "encoder_attention_mask", "ref_answer_position", "model_answer_position", "ref_attention_mask", "ref_labels")}, step=0, step_ratio=0.0)
o["loss"].backward(); gn = [p.grad.norm().item() for n, p in m.named_parameters() if p.grad is not None and "prj" in n]
allg = [p.grad for p in m.parameters() if p.grad is not None]
check("loss finite, gradients finite, projection gets gradient", bool(torch.isfinite(o["loss"])) and all(torch.isfinite(x).all() for x in allg) and max(gn) > 0, f"loss {float(o['loss']):.3f} prj grad {max(gn):.3f}")
print("FAILED:" if fails else "ALL PASSED", fails); sys.exit(1 if fails else 0)
