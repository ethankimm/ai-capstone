#!/usr/bin/env python3
"""Pre-launch check for CODI's padding handling (TRAINING_PROTOCOL.md section 5.2). Run INSIDE the CODI venv, from a
CODI checkout with scripts/codi_streaming.patch, codi_synthetic.patch and codi_model_fixes.patch applied. CPU is fine:

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/check_codi_padding.py \
      --data_json /workspace/synthetic_data_v2/depth2/valid.json [--device cpu]

Builds CODI on pretrained GPT-2 (random LoRA B weights, so the adapters are active) and, for items of different levels
(so questions of very different lengths share a left-padded batch), checks that each item gives the same result alone as
inside the padded batch:
  1. training forward (src/model.py): student decoder logits;
  2. eval (scripts/eval_codi.py): logits at the distillation position after <eot> "The answer is:";
  3. eval greedy generation (generate_batches): decoded strings;
and that 4. training and eval agree on the logits at the distillation position for the same item.
Checks 1-3 must pass with --pad_invariant True and are expected to FAIL with False (upstream behaviour), which shows the
test detects the problem. Exit 0 only if every pad_invariant=True check passes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from peft import LoraConfig, TaskType  # noqa: E402
from src.model import CODI, DataArguments, ModelArguments, TrainingArguments  # noqa: E402

from eval_codi import encode_and_think, generate_batches, tokenize_questions  # noqa: E402

ATOL = 1e-3


def build(pad_invariant: bool, device: str, seed: int):
    torch.manual_seed(seed)
    margs = ModelArguments(model_name_or_path="gpt2", lora_r=128, lora_alpha=32, lora_init=True, train=False)
    targs = TrainingArguments(output_dir=tempfile.mkdtemp(), num_latent=6, inf_latent_iterations=6, use_prj=True,
                              prj_dim=768, prj_dropout=0.0, fix_attn_mask=True, pad_invariant=pad_invariant,
                              remove_eos=True, distill_loss_div_std=True, greedy=True, print_loss=False,
                              bf16=False, use_cpu=(device == "cpu"), report_to=[])
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=128, lora_alpha=32, lora_dropout=0.1,
                      target_modules=["c_attn", "c_proj", "c_fc"], init_lora_weights=True)
    model = CODI(margs, targs, lora)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02)
    model = model.float().to(device).eval()
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained("gpt2", padding_side="left", use_fast=False)
    tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.pad_token_id = model.pad_token_id
    return model, tok, targs


def train_features(row: dict, tok, model) -> dict:
    """codi_synthetic.patch + CODI train.py preprocess for one item (remove_eos True, include_last_cot False)."""
    src = tok.encode(row["question"].strip() + "\n")
    cot = tok.encode("\n".join(row["steps"][:-1]) + "\n")
    ans = tok.encode(f"The answer is: {str(row['answer']).strip()}") + [tok.eos_token_id]
    prompt = tok.encode("The answer is:")
    ref = src + cot + ans
    dec = [model.eot_id] + ans

    def after_prompt(seq):
        for i in range(len(seq) - len(prompt) + 1):
            if seq[i:i + len(prompt)] == prompt:
                return i + len(prompt)
        raise ValueError("prompt not found")
    return {"enc": src + [model.bot_id], "dec": dec, "ref": ref, "ref_labels": [-100] * len(src) + ref[len(src):],
            "ref_pos": after_prompt(ref), "dec_pos": after_prompt(dec)}


def collate(feats: list[dict], pad: int, device: str) -> dict:
    def right(seqs, val):
        width = max(len(s) for s in seqs)
        return torch.tensor([s + [val] * (width - len(s)) for s in seqs])

    def left(seqs, val):
        width = max(len(s) for s in seqs)
        return torch.tensor([[val] * (width - len(s)) + s for s in seqs])
    enc = left([f["enc"] for f in feats], pad)
    ref = right([f["ref"] for f in feats], pad)
    batch = {"encoder_input_ids": enc, "decoder_input_ids": right([f["dec"] for f in feats], pad),
             "ref_input_ids": ref, "labels": right([f["dec"] for f in feats], -100),
             "encoder_attention_mask": enc.ne(pad), "ref_attention_mask": ref.ne(pad),
             "ref_answer_position": torch.tensor([f["ref_pos"] for f in feats]),
             "model_answer_position": torch.tensor([f["dec_pos"] for f in feats]),
             "ref_labels": right([f["ref_labels"] for f in feats], -100)}
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def eval_logits(model, tok, targs, questions, device) -> torch.Tensor:
    """Logits at the distillation position (after <eot> "The answer is:"), via eval_codi's helpers."""
    dec = [model.eot_id] + tok.encode("The answer is:")
    embed = model.get_embd(model.codi, model.model_name)
    batch = tokenize_questions(model, tok, questions, len(questions), targs, device)[0]
    past, pi, _ = encode_and_think(model, batch, targs)
    emb = embed(torch.tensor(dec, device=device))[None].expand(len(questions), -1, -1)
    out = model.codi(inputs_embeds=emb, use_cache=True, past_key_values=past, **(pi.next(len(dec)) if pi else {"attention_mask": None}))
    return out.logits[:, -1]


def run(pad_invariant: bool, rows: list[dict], device: str, seed: int) -> dict:
    model, tok, targs = build(pad_invariant, device, seed)
    feats = [train_features(r, tok, model) for r in rows]
    questions = [r["question"].strip() + "\n" for r in rows]
    res = {}
    with torch.no_grad():
        batched = model(**collate(feats, model.pad_token_id, device))["logits"]
        diffs, train_at_answer = [], []
        for b, f in enumerate(feats):
            alone = model(**collate([f], model.pad_token_id, device))["logits"][0]
            n = len(f["dec"])
            diffs.append((batched[b, :n] - alone[:n]).abs().max().item())
            train_at_answer.append(alone[f["dec_pos"] - 1])  # logits predicting the answer token
        res["1 training forward: batch vs alone"] = max(diffs)
        eb = eval_logits(model, tok, targs, questions, device)
        ea = torch.stack([eval_logits(model, tok, targs, [q], device)[0] for q in questions])
        res["2 eval distillation position: batch vs alone"] = (eb - ea).abs().max().item()
        res["4 training vs eval at the distillation position"] = (torch.stack(train_at_answer) - ea).abs().max().item()
    dargs = DataArguments()
    dargs.batch_size = len(questions)
    gb = generate_batches(model, tok, questions, dargs, targs)
    dargs.batch_size = 1
    g1 = generate_batches(model, tok, questions, dargs, targs)
    res["3 eval generation: batch vs alone (items differing)"] = float(sum(x != y for x, y in zip(gb, g1)))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--levels", default="1,4,8,12")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = json.loads(Path(a.data_json).read_text())
    rows = [next(r for r in data if int(r["level"]) == lv) for lv in map(int, a.levels.split(","))]
    print("question tokens per item:", [len(r["question"].split()) for r in rows], "(words)")
    ok = True
    for pad_invariant in (True, False):
        res = run(pad_invariant, rows, a.device, a.seed)
        print(f"\npad_invariant={pad_invariant}")
        for k, v in res.items():
            passed = v <= ATOL
            print(f"  {'PASS' if passed else 'FAIL'}  {k}: {v:.2e}")
            if pad_invariant:
                ok &= passed
    print("\nRESULT:", "OK (pad_invariant=True is padding-invariant and train/eval-consistent)" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
