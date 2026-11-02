#!/usr/bin/env python3
"""Do CODI's latents collapse on GSM8K too? Logit lens + latent-collapse check on the authors' released GPT-2 CODI
(`zen-E/CODI-gpt2` @ fd641b3, 43.7% GSM8K), the GSM8K counterpart of the RuleChain runs
20261102-233815 (logit lens) and its collapse addendum.

Batch size 1, so there is no padding: upstream's unmasked latent passes and cache-derived positions are then exactly
what the model was trained with (the released model was trained without our pad_invariant fix). Per item: question
(test.py formatting) + <bot>, 6 latent passes (z_1 = prj(h_<bot>), z_{j+1} = prj(h of pass j)), then
"<eot> The answer is:" teacher-forced.

Reports
- teacher-forced accuracy of the answer's first token at ":" (sanity: should be near the model's 41-44%).
- logit lens at every layer of each latent pass (final LayerNorm + tied unembedding; layer 12 is already normed):
  top-1 token == first token of intermediate value v_s (with or without a leading space), as a [latent position x step]
  matrix, plus "any of the item's intermediate values" and "is a number".
- collapse: within-item cosine between latent positions per layer, raw and centred (minus the per-layer mean over
  items and positions), cross-item cosine, share of variance between latent positions.

Run from the CODI checkout in its venv:
  .venv/bin/python ../ai-capstone/scripts/gsm8k_codi_lens_collapse.py --ckpt_dir <snapshot of zen-E/CODI-gpt2> \\
      --output_dir /tmp/u --model_name_or_path gpt2 --seed 11 --model_max_length 512 --lora_r 128 --lora_alpha 32 \\
      --lora_init --batch_size 1 --greedy True --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False \\
      --prj_dropout 0.0 --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402

from eval_codi import build_model  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

if not torch.cuda.is_available():
    _torch_load = torch.load
    torch.load = lambda *a, **kw: _torch_load(*a, **{"map_location": "cpu", **kw})


@dataclass
class Args:
    n: int = field(default=0, metadata={"help": "0 = full test split (1319)"})
    slug: str = field(default="gsm8k-released-lens-collapse")
    stage: str = field(default="full_run")
    hardware: str = field(default="local")
    checkpoint_label: Optional[str] = field(default="hf:zen-E/CODI-gpt2@fd641b3")
    no_log: bool = field(default=False)


def first_tokens(tokenizer, value: str) -> set[int]:
    v = value.replace(",", "")
    if v.endswith(".0"):
        v = v[:-2]
    return {tokenizer.encode(s)[0] for s in (v, " " + v) if tokenizer.encode(s)}


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, Args))
    model_args, data_args, training_args, a = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, load_result = build_model(model_args, training_args, device=device, dtype=torch.float32)
    print(f"loaded: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}", flush=True)
    k = training_args.inf_latent_iterations
    lm = model.codi.get_base_model()
    body, ln_f, head = lm.transformer, lm.transformer.ln_f, lm.lm_head
    embed = model.get_embd(model.codi, model.model_name)
    vocab = model.codi.config.vocab_size - 1
    dec = torch.tensor([model.eot_id] + tokenizer.encode("The answer is:"), device=device)
    is_num = torch.zeros(vocab, dtype=torch.bool)
    for t in range(min(vocab, len(tokenizer))):  # ids past the tokenizer (added <bot>/<eot> rows) are not numbers
        s = tokenizer.decode([t]).strip()
        is_num[t] = bool(s) and s.replace(".", "", 1).isdigit()

    exs = load_gsm8k_aug(split="test", n=None, seed=None)
    if a.n:
        exs = exs[:a.n]
    t0 = time.perf_counter()
    H = []          # [N, k, 13, d] latent-pass hidden states (fp16)
    top1 = []       # [N, k, 13] top-1 token per latent position and layer
    tf_ok = []
    for i, ex in enumerate(exs):
        q = ex.question.strip().replace("  ", " ")
        ids = torch.tensor([tokenizer.encode(q) + [model.bot_id]], device=device)
        with torch.no_grad():
            out = body(input_ids=ids, use_cache=True)
            past = out.past_key_values
            z = model.prj(out.last_hidden_state[:, -1:, :])
            hs = []
            for _ in range(k):
                out = body(inputs_embeds=z, use_cache=True, past_key_values=past, output_hidden_states=True)
                past = out.past_key_values
                hs.append(torch.stack([h[0, -1] for h in out.hidden_states]))  # [13, d]
                z = model.prj(out.last_hidden_state[:, -1:, :])
            hs = torch.stack(hs)  # [k, 13, d]
            normed = torch.cat([ln_f(hs[:, :12]), hs[:, 12:]], 1)
            top1.append(head(normed)[..., :vocab].argmax(-1).cpu())
            H.append(hs.half().cpu())
            out = model.codi(inputs_embeds=embed(dec)[None], use_cache=True, past_key_values=past)
            pred = out.logits[0, -1, :vocab].argmax().item()
        tf_ok.append(pred in first_tokens(tokenizer, ex.answer))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(exs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    H = torch.stack(H).float()        # [N, k, 13, d]
    top1 = torch.stack(top1)          # [N, k, 13]
    N = len(exs)
    acc = sum(tf_ok) / N
    print(f"teacher-forced first-token accuracy {acc:.3f}", flush=True)

    vals = [ex.intermediate_values for ex in exs]
    targets = [[first_tokens(tokenizer, v) for v in vs] for vs in vals]
    max_steps = 6
    # lens matrix at each layer: [latent position p, step s] -> fraction (items with >= s steps)
    lens = {}
    for layer in range(13):
        mat = []
        for p in range(k):
            row = []
            for s in range(max_steps):
                idx = [i for i in range(N) if len(targets[i]) > s]
                row.append(round(sum(int(top1[i, p, layer]) in targets[i][s] for i in idx) / max(len(idx), 1), 4))
            mat.append(row)
        anyv = [round(sum(any(int(top1[i, p, layer]) in t for t in targets[i]) for i in range(N)) / N, 4) for p in range(k)]
        num = [round(is_num[top1[:, p, layer]].float().mean().item(), 4) for p in range(k)]
        lens[layer] = {"pos_by_step": mat, "any_intermediate": anyv, "is_number": num}
    best = max(range(13), key=lambda l: sum(lens[l]["any_intermediate"]))
    print(f"logit lens, best layer {best}: any-intermediate by latent position {lens[best]['any_intermediate']}, "
          f"is-number {lens[best]['is_number']}", flush=True)
    for p, row in enumerate(lens[best]["pos_by_step"]):
        print(f"  latent {p + 1}: step1..6 {row}", flush=True)
    toks = {f"z{p + 1}@{best}": [tokenizer.decode([t]) for t in top1[:8, p, best].tolist()] for p in range(k)}
    print("first 8 items' top-1 tokens per latent at that layer:", toks, flush=True)

    collapse = {}
    for layer in range(13):
        X = H[:, :, layer]
        xr = torch.nn.functional.normalize(X, dim=-1)
        Xc = X - X.mean((0, 1), keepdim=True)
        xc = torch.nn.functional.normalize(Xc, dim=-1)
        Cr = torch.einsum("nid,njd->ij", xr, xr) / N
        Cc = torch.einsum("nid,njd->ij", xc, xc) / N
        off = ~torch.eye(k, dtype=torch.bool)
        y = xc[:, 2]
        cc = y @ y.T
        tot = Xc.pow(2).sum(-1).mean()
        within = (Xc - Xc.mean(1, keepdim=True)).pow(2).sum(-1).mean()
        collapse[layer] = {"raw_within_item_cos": round(Cr[off].mean().item(), 4),
                           "centred_within_item_cos": round(Cc[off].mean().item(), 4),
                           "centred_cos_matrix": [[round(x, 3) for x in r] for r in Cc.tolist()],
                           "centred_cross_item_cos": round(((cc.sum() - N) / (N * N - N)).item(), 4),
                           "share_variance_between_positions": round((within / tot).item(), 4)}
        print(f"layer {layer:2d}: within-item cos raw {collapse[layer]['raw_within_item_cos']:.3f} "
              f"centred {collapse[layer]['centred_within_item_cos']:.3f} | share var between positions "
              f"{collapse[layer]['share_variance_between_positions']:.3f}", flush=True)
    print("centred cos matrix at layer 12:", collapse[12]["centred_cos_matrix"], flush=True)

    if a.no_log:
        return
    records = [{"idx": ex.idx, "question": ex.question, "gold_answer": ex.answer, "tf_first_token_correct": ok,
                "intermediate_values": vals[i],
                "top1_by_latent_at_best_layer": [tokenizer.decode([t]) for t in top1[i, :, best].tolist()]}
               for i, (ex, ok) in enumerate(zip(exs, tf_ok))]
    record = RunRecord(
        run_id=new_run_id("codi", a.slug), mechanism="codi", stage=a.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=a.checkpoint_label or model_args.ckpt_dir,
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=N, seed=None),
        metrics={"final_answer_accuracy": round(acc, 4), "compute_steps": k,
                 "sec_per_example": (time.perf_counter() - t0) / N,
                 "extra": {"accuracy_kind": "teacher-forced first token of the answer after 'The answer is:' (not the "
                                            "paper metric)", "best_lens_layer": best, "logit_lens": lens,
                           "collapse": collapse}},
        hyperparams={"latent_count": k, "batch_size": 1, "masking": "upstream (no padding at batch 1)", "device": device},
        seed=None, hardware=f"{a.hardware} / {device}",
        notes="Released GSM8K CODI: logit lens and latent collapse check.",
    )
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id}")


if __name__ == "__main__":
    main()
