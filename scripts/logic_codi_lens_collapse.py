#!/usr/bin/env python3
"""Do CODI's latents collapse on ProsQA / ProntoQA? Latent-collapse statistics + logit lens + latent truncation on the
Dilgren & Wiegreffe released GPT-2 CODI checkpoints (`connordilgren/gpt2-{prosqa,prontoqa}-codi`), the logic-task
counterpart of the GSM8K run 20261102-235641 (not collapsed) and the RuleChain run 20261102-233815 (collapsed).

Our Zhang-swap runs (20261011-061753 / -062104) found these models ignore their latents (random swap changes 0% of
answers). Kevin's hypothesis: if latents are not needed, the recurrently fed state should barely change between
latents, i.e. collapse. This tests it.

Loads with the grid repo's `CODIInference` (as `swap_random_codi.py`), batch size 1 (no padding), fp32.
Per item: question + <bot>, 6 latent passes (z_1 = prj(h_<bot>), z_{j+1} = prj(h of pass j)), then <eot> and greedy
generation; the answer is the text after "The answer is:" (same extraction as `swap_random_codi.py`). The same item is
also decoded with 0 latents (question + <bot> + <eot>; off-distribution truncation, secondary).

Reports, per latent position and layer (0 = embeddings .. 12 = post-ln_f):
- collapse: within-item cosine between latent positions, raw and centred (minus the per-layer mean over items and
  positions), cross-item centred cosine, share of variance between latent positions (same code as the GSM8K run).
- logit lens top-1 token: share that is a word from the item's own gold reasoning steps (first BPE token of a step
  word, with leading space, excluding a small stoplist), vs share that is a word appearing only in the question
  (distractor facts).

  ../codi/.venv/bin/python scripts/logic_codi_lens_collapse.py --dataset prosqa \\
      --checkpoint_path <dir>/gpt2-prosqa-codi/pytorch_model.bin \\
      --grid_repo ~/Projects/are-lrms-easily-interpretable --data_dir ~/Projects/are-lrms-easily-interpretable/data
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

DELIM = "The answer is:"
DATA_FILES = {"prontoqa": "prontoqa_test.json", "prosqa": "prosqa_test.json"}
STOP = {"is", "a", "an", "are", "every", "each", "the", "not", "or", "and", "true", "false", "of"}


@dataclass
class Args:
    checkpoint_path: str = field(metadata={"help": "grid pytorch_model.bin"})
    grid_repo: str = field(metadata={"help": "checkout of are-lrms-easily-interpretable"})
    data_dir: str = field(metadata={"help": "are-lrms-easily-interpretable/data"})
    dataset: str = field(default="prosqa")
    num_latents: int = field(default=6)
    n: int = field(default=0, metadata={"help": "0 = full test split"})
    max_new_tokens: int = field(default=48)
    device: str = field(default="cpu")
    slug: Optional[str] = field(default=None)
    stage: str = field(default="full_run")
    hardware: str = field(default="local Apple Silicon")
    no_log: bool = field(default=False)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[A-Za-z]+", text.lower()) if w not in STOP}


def norm(s: Optional[str]) -> str:
    return (s or "").strip().rstrip(".").lower()


def main() -> None:
    (a,) = transformers.HfArgumentParser((Args,)).parse_args_into_dataclasses()
    assert a.dataset in DATA_FILES
    argv = " ".join(sys.argv)
    sys.path.insert(0, str(Path(a.grid_repo).expanduser()))
    from models.codi.codi_wrapper import CODIInference  # noqa: E402

    model = CODIInference(model_name_or_path="gpt2", num_latent=a.num_latents, use_prj=True, prj_dim=768,
                          use_lora=True, lora_r=128, lora_alpha=32, device=a.device)
    model.load_checkpoint(a.checkpoint_path)
    model = model.to(a.device).eval().float()
    tok = model.tokenizer
    dev = torch.device(a.device)
    embed = model.get_embd(model.codi, model.model_name)
    lm = model.codi.get_base_model()
    body, ln_f, head = lm.transformer, lm.transformer.ln_f, lm.lm_head
    vocab = model.codi.config.vocab_size - 1
    k = a.num_latents

    first_tok_cache: dict[str, int] = {}

    def ft(w: str) -> int:
        if w not in first_tok_cache:
            first_tok_cache[w] = tok.encode(" " + w)[0]
        return first_tok_cache[w]

    @torch.no_grad()
    def decode(pkv) -> str:
        nxt = embed(torch.tensor([model.eot_id], device=dev)).unsqueeze(0)
        gen = []
        for _ in range(a.max_new_tokens):
            out = body(inputs_embeds=nxt, use_cache=True, past_key_values=pkv)
            pkv = out.past_key_values
            t = int(head(out.last_hidden_state[0, -1])[:vocab].argmax())
            if t == tok.eos_token_id:
                break
            gen.append(t)
            nxt = embed(torch.tensor([t], device=dev)).unsqueeze(0)
        raw = tok.decode([t for t in gen if t < len(tok)], skip_special_tokens=True)
        return raw.split(DELIM)[-1].strip() if DELIM in raw else ""

    rows = json.loads((Path(a.data_dir) / DATA_FILES[a.dataset]).read_text())
    if a.n:
        rows = rows[:a.n]
    N = len(rows)
    H, top1, ok, ok0, records = [], [], [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        enc = tok(r["question"].strip().replace("  ", " "), return_tensors="pt", add_special_tokens=True)
        ids = torch.cat([enc["input_ids"], torch.tensor([[model.bot_id]])], 1).to(dev)
        with torch.no_grad():
            out = body(input_ids=ids, use_cache=True)
            pred0 = decode(out.past_key_values)  # 0 latents; the cache is mutated, so the question is re-encoded below
            out = body(input_ids=ids, use_cache=True)
            pkv = out.past_key_values
            z = model.prj(out.last_hidden_state[:, -1:, :])
            hs = []
            for _ in range(k):
                out = body(inputs_embeds=z, use_cache=True, past_key_values=pkv, output_hidden_states=True)
                pkv = out.past_key_values
                hs.append(torch.stack([h[0, -1] for h in out.hidden_states]))
                z = model.prj(out.last_hidden_state[:, -1:, :])
            hs = torch.stack(hs)  # [k, 13, d]
            normed = torch.cat([ln_f(hs[:, :12]), hs[:, 12:]], 1)
            t1 = head(normed)[..., :vocab].argmax(-1).cpu()
            pred = decode(pkv)
        H.append(hs.half().cpu())
        top1.append(t1)
        ok.append(norm(pred) == norm(r["answer"]))
        ok0.append(norm(pred0) == norm(r["answer"]))
        records.append({"idx": i, "gold_answer": r["answer"], "pred": pred, "correct": ok[-1],
                        "pred_0_latents": pred0, "correct_0_latents": ok0[-1]})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{N} ({time.perf_counter() - t0:.0f}s) acc {sum(ok) / len(ok):.3f} "
                  f"acc0 {sum(ok0) / len(ok0):.3f}", flush=True)
    elapsed = time.perf_counter() - t0
    H = torch.stack(H).float()
    top1 = torch.stack(top1)
    acc, acc0 = sum(ok) / N, sum(ok0) / N
    print(f"accuracy 6 latents {acc:.3f} | 0 latents {acc0:.3f}", flush=True)

    # logit lens: step-word vs question-only-word hits
    step_t = [{ft(w) for w in words(" ".join(r["steps"]))} for r in rows]
    q_only_t = [{ft(w) for w in words(r["question"]) - words(" ".join(r["steps"]) + " " + r["answer"])} - step_t[i]
                for i, r in enumerate(rows)]
    lens = {}
    for layer in range(13):
        sw = [round(sum(int(top1[i, p, layer]) in step_t[i] for i in range(N)) / N, 4) for p in range(k)]
        qw = [round(sum(int(top1[i, p, layer]) in q_only_t[i] for i in range(N)) / N, 4) for p in range(k)]
        lens[layer] = {"step_word": sw, "question_only_word": qw}
    best = max(range(13), key=lambda l: sum(lens[l]["step_word"]))
    print(f"logit lens best layer {best}: step-word by latent {lens[best]['step_word']}, "
          f"question-only-word {lens[best]['question_only_word']}", flush=True)
    print(f"layer 12 step-word {lens[12]['step_word']}, question-only {lens[12]['question_only_word']}", flush=True)
    examples_top = {f"z{p + 1}@{L}": [tok.decode([t]) for t in top1[:6, p, L].tolist()]
                    for L in (best, 12) for p in range(k)}
    print("first 6 items' top-1 tokens:", examples_top, flush=True)

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
        # two-way decomposition of the uncentred energy sum ||X||^2 = constant + position + item + interaction
        m = X.mean((0, 1))
        P = X.mean(0) - m                       # [k, d] position effect shared by all items
        I = X.mean(1) - m                       # [N, d] item effect shared by all positions
        R = X - m - P[None] - I[:, None]        # item-specific change across positions
        tot_u = X.pow(2).sum(-1).mean()
        dec = {"constant": m.pow(2).sum(), "position": P.pow(2).sum(-1).mean(), "item": I.pow(2).sum(-1).mean(),
               "item_x_position": R.pow(2).sum(-1).mean()}
        xr2 = xr[:, 2]
        raw_cross = ((xr2 @ xr2.T).sum() - N) / (N * N - N)
        collapse[layer] = {"energy_share": {kk: round((v / tot_u).item(), 4) for kk, v in dec.items()},
                           "raw_cross_item_cos": round(raw_cross.item(), 4),
                           "raw_within_item_cos": round(Cr[off].mean().item(), 4),
                           "centred_within_item_cos": round(Cc[off].mean().item(), 4),
                           "centred_cos_matrix": [[round(x, 3) for x in row] for row in Cc.tolist()],
                           "centred_cross_item_cos": round(((cc.sum() - N) / (N * N - N)).item(), 4),
                           "share_variance_between_positions": round((within / tot).item(), 4)}
        print(f"layer {layer:2d}: within-item cos raw {collapse[layer]['raw_within_item_cos']:.3f} "
              f"centred {collapse[layer]['centred_within_item_cos']:.3f} | share var between positions "
              f"{collapse[layer]['share_variance_between_positions']:.3f} | raw cross-item cos "
              f"{collapse[layer]['raw_cross_item_cos']:.3f} | energy {collapse[layer]['energy_share']}", flush=True)
    print("centred cos matrix at layer 12:", collapse[12]["centred_cos_matrix"], flush=True)

    if a.no_log:
        return
    slug = a.slug or f"{a.dataset}-released-lens-collapse"
    record = RunRecord(
        run_id=new_run_id("codi", slug), mechanism="codi", stage=a.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=f"hf:connordilgren/gpt2-{a.dataset}-codi/pytorch_model.bin",
                        n_params=sum(p.numel() for p in model.parameters())),
        dataset=DatasetInfo(name=a.dataset, split="test", n_examples=N, seed=None),
        metrics={"final_answer_accuracy": round(acc, 4), "compute_steps": k, "sec_per_example": elapsed / N,
                 "early_termination_necessity": round(acc - acc0, 4),
                 "extra": {"accuracy_0_latents": round(acc0, 4), "best_lens_layer": best, "logit_lens": lens,
                           "collapse": collapse}},
        hyperparams={"latent_count": k, "batch_size": 1, "dtype": "float32", "greedy": True,
                     "max_new_tokens": a.max_new_tokens, "device": a.device, "loader": "grid CODIInference"},
        seed=None, hardware=f"{a.hardware} / {a.device}",
        notes=f"Released {a.dataset} CODI: latent collapse + logit lens + 0-latent truncation.",
    )
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id}")


if __name__ == "__main__":
    main()
