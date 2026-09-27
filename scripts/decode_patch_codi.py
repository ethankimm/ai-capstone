#!/usr/bin/env python3
"""CODI decodability + causal-patching -- the faithfulness-spine contrast case to
recurrent_depth's step-supervised run (results/20260918-214656_recurrent_depth_stepsup-*).

Reuses `scripts/eval_codi.py`'s model loading exactly (same released checkpoint, same
paper protocol). Two things eval_codi.py doesn't do:

1. Decoding: at each of the 6 loop iterations, `outputs.logits[:, -1, :]` computed in
   that SAME forward pass IS the paper's own decode method (Sec 5.1: "projecting its
   last hidden state into vocabulary space via the model's word embeddings") -- GPT-2
   ties wte/lm_head and LoRA doesn't touch either, so `logits = lm_head(hidden_state)`
   is exactly that projection, already computed, no extra work.

2. Causal patching: swap one thought z_i (post-projection -- the actual value fed
   forward, per generate_batches) for a donor example's z_i computed from the donor's
   own independent context, keep the recipient's own KV cache/trajectory otherwise
   unchanged, let iterations i+1..6 and the final answer proceed from the patched
   value. Two controls: a latent from a random unrelated (example, ANY iteration)
   pair, and a random unrelated (example, LIVE iteration only) pair -- separating "any
   perturbation moves the answer" from "this specific counterfactual content moves the
   answer", and checking whether restricting the control to in-distribution (live)
   activations changes anything.

v2 (this run) vs the pilot (`results/20260919-073312_codi_decode-patch-pilot/`):
full 1319-example test set instead of the 200-slice; a held-out split so the
iteration<->step mapping is fit on half A and all headline numbers (decoding AND
patch-pair selection) are computed on half B only (the pilot fit and evaluated on the
same 200 examples -- flagged as circular in its own caveats); a train-corpus
most-common-value baseline instead of a same-slice one; patch pairs scaled to ~350 and
stratified toward the single strongest decode iteration instead of spread evenly;
both control variants (any-iteration / live-iteration-only); paired exact McNemar test
and Wilson CIs on the headline proportions instead of raw fractions alone.

Run inside the CODI venv, from the CODI checkout (same convention as eval_codi.py):

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/decode_patch_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug decode-patch-full --stage full_run --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --full_test True --mapping_split True --n_patch_pairs 350
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`

from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402
from eval_codi import build_model  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

ORIG_VOCAB = 50257  # GPT-2's real vocab; indices >= this are the added pad/bot/eot


@dataclass
class DecodeArguments:
    slug: str = field(default="decode-patch-full")
    stage: str = field(default="full_run")
    eval_n: int = field(default=200)  # only used if full_test=False
    eval_seed: int = field(default=0)
    full_test: bool = field(default=True, metadata={"help": "decode all 1319 test examples instead of the 200-slice"})
    mapping_split: bool = field(default=True, metadata={"help": "fit best_iter_for_step on half A, report/patch on half B"})
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    n_patch_pairs: int = field(default=350)
    patch_iter_focus: Optional[int] = field(default=None, metadata={"help": "defaults to best_iter_for_step[1] from half A"})
    patch_focus_frac: float = field(default=0.65)
    train_baseline_n: int = field(default=20000)
    max_new_tokens: int = field(default=64)


# ---- generation primitives (unchanged from the pilot) -----------------------------------

def encode_question(model, tokenizer, question: str, device: str):
    batch = tokenizer([question], return_tensors="pt", padding="longest")
    bot = torch.tensor([[model.bot_id]], dtype=torch.long)
    input_ids = torch.cat([batch["input_ids"], bot], dim=1).to(device)
    attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1).to(device)
    return input_ids, attn


@torch.no_grad()
def run_thoughts(model, tokenizer, question: str, device: str, n_iters: int,
                  override_input_at: dict[int, torch.Tensor] | None = None,
                  include_latent0: bool = False):
    """Encode `question`, run `n_iters` loop iterations (1-indexed: iter 1..n_iters).
    `override_input_at`: {iter_index: post_proj_tensor} -- if iter i is a key, the
    computed post-proj latent that would normally feed iteration i is replaced by the
    given tensor before running that iteration's forward pass (a causal patch).
    Returns (past_key_values, records, final_post_proj_latent_for_next_step).
    records[i-1] = {"iter": i, "logits": [vocab], "post": tensor} for the *output* of
    iteration i (the i-th continuous thought z_i) -- UNLESS `include_latent0=True`, in
    which case an extra {"iter": 0, ...} record for the pre-loop encode-pass hidden
    state (the value that feeds iteration 1, read out via the same lm_head projection,
    no extra forward pass needed) is prepended, and every other record shifts by one
    position -- so any caller relying on positional `records[i-1]` indexing rather than
    filtering by `rec["iter"]` MUST NOT pass `include_latent0=True`. Default False keeps
    every existing caller's behavior identical; only `decode_patch_codi.py`'s own decode
    loop opts in (this is the reference repo's "latent 0" probe position, never scored
    here before -- see `20260923-*` notes for why it matters for decodability)."""
    input_ids, attn = encode_question(model, tokenizer, question, device)
    outputs = model.codi(input_ids=input_ids, use_cache=True, output_hidden_states=True,
                          past_key_values=None, attention_mask=attn)
    pkv = outputs.past_key_values
    latent = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    if model.use_prj:
        latent = model.prj(latent)

    records = []
    if include_latent0:
        latent0_logits = outputs.logits[:, -1, :].float().cpu()
        records.append({"iter": 0, "logits": latent0_logits, "post": latent})
    for it in range(1, n_iters + 1):
        feed = latent
        if override_input_at is not None and it in override_input_at:
            feed = override_input_at[it]
            if callable(feed):  # f(live latent) -> latent to feed, e.g. a subspace patch
                feed = feed(latent)
        outputs = model.codi(inputs_embeds=feed, use_cache=True, output_hidden_states=True, past_key_values=pkv)
        pkv = outputs.past_key_values
        pre_hidden = outputs.hidden_states[-1][:, -1, :]
        logits = outputs.logits[:, -1, :].float().cpu()
        post = model.prj(pre_hidden.unsqueeze(1)) if model.use_prj else pre_hidden.unsqueeze(1)
        records.append({"iter": it, "logits": logits, "post": post})
        latent = post
    return pkv, records, latent


@torch.no_grad()
def decode_answer(model, tokenizer, pkv, latent, device: str, max_new_tokens: int):
    """Continue from the 6th thought's post-proj latent through eot + greedy decode.
    Batch size 1. Mirrors eval_codi.py's generate_batches tail."""
    eot_emb = model.get_embd(model.codi, model.model_name)(
        torch.tensor([model.eot_id], dtype=torch.long, device=device)
    ).unsqueeze(0)
    output = eot_emb
    pred_tokens: list[int] = []
    for _ in range(max_new_tokens):
        out = model.codi(inputs_embeds=output, output_hidden_states=False, attention_mask=None,
                          use_cache=True, past_key_values=pkv)
        pkv = out.past_key_values
        logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]
        next_id = torch.argmax(logits, dim=-1)  # shape [1] (batch=1) -- do NOT squeeze(-1) here,
        # that collapses the size-1 batch dim to a 0-d scalar and breaks the embed+unsqueeze below
        if next_id.item() == tokenizer.eos_token_id:
            break
        pred_tokens.append(next_id.item())
        output = model.get_embd(model.codi, model.model_name)(next_id).unsqueeze(1).to(device)
    return tokenizer.decode(pred_tokens, skip_special_tokens=True)


def topk_token_strings(tokenizer, logits_1d: torch.Tensor, k: int) -> list[str]:
    masked = logits_1d.clone()
    masked[ORIG_VOCAB:] = -float("inf")  # never decode to pad/bot/eot as "the answer"
    top = torch.topk(masked, k).indices.tolist()
    return [tokenizer.decode([t]).strip() for t in top]


def num_match(candidates: list[str], gold: str) -> bool:
    gold = gold.strip().lstrip("+")
    for c in candidates:
        c = c.strip()
        if c == gold:
            return True
        try:
            if float(c.replace(",", "")) == float(gold.replace(",", "")):
                return True
        except ValueError:
            continue
    return False


# ---- stats helpers (pure python -- no scipy/statsmodels dependency in the CODI venv) ----

def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = x / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs b, c (b+c = n_discordant),
    X ~ Binomial(b+c, 0.5) under H0. Equivalent to statsmodels' mcnemar(exact=True)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_le_k = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p_le_k)


def rate(hits: dict, it: int, s: int) -> float:
    c, n = hits[it][s]
    return c / n if n else 0.0


def build_matrix(decode_records: list[dict], n_latents: int, max_step: int):
    """Unconditional top1/top5/top10 hit-count matrices over the given decode_records,
    for iterations 1..n_latents only -- the matched-iteration mapping fit and
    `patch_iter_focus` selection are defined over these 6 loop positions, unchanged by
    `include_latent0`. A `per_iter` entry with `"iter": 0` (present when the decode loop
    was run with `include_latent0=True`) is intentionally skipped here; it only feeds
    the ANY-iteration metrics (`any_iter_hit` et al.), which iterate `per_iter` directly
    and are not restricted to 1..n_latents."""
    hits1 = {i: {s: [0, 0] for s in range(1, max_step + 1)} for i in range(1, n_latents + 1)}
    hits5 = {i: {s: [0, 0] for s in range(1, max_step + 1)} for i in range(1, n_latents + 1)}
    hits10 = {i: {s: [0, 0] for s in range(1, max_step + 1)} for i in range(1, n_latents + 1)}
    for r in decode_records:
        for s_idx, gold_val in enumerate(r["step_values"], start=1):
            for pi in r["per_iter"]:
                it = pi["iter"]
                if it == 0:
                    continue
                hits1[it][s_idx][1] += 1
                hits5[it][s_idx][1] += 1
                hits10[it][s_idx][1] += 1
                if num_match([pi["top1"]], gold_val):
                    hits1[it][s_idx][0] += 1
                if num_match(pi["top5"], gold_val):
                    hits5[it][s_idx][0] += 1
                if num_match(pi["top10"], gold_val):
                    hits10[it][s_idx][0] += 1
    return hits1, hits5, hits10


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DecodeArguments))
    model_args, data_args, training_args, dec_args = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")

    if dec_args.full_test:
        examples = load_gsm8k_aug(split="test", n=None, seed=None)
    else:
        examples = load_gsm8k_aug(split="test", n=dec_args.eval_n, seed=dec_args.eval_seed)
    n_latents = training_args.inf_latent_iterations
    print(f"decoding {len(examples)} examples")

    # ---- Task 1: decode every example -----------------------------------------------------
    t0 = time.perf_counter()
    decode_records = []
    out_dir_placeholder = Path("/tmp/codi_decode_patch_progress.jsonl")
    with out_dir_placeholder.open("w") as progress_f:
        for ex in examples:
            question = ex.question.strip().replace("  ", " ")
            pkv, thoughts, latent = run_thoughts(model, tokenizer, question, device, n_latents,
                                                  include_latent0=True)
            raw_output = decode_answer(model, tokenizer, pkv, latent, device, dec_args.max_new_tokens)
            pred = extract_final_number(raw_output)
            correct = is_correct(pred, ex.answer)
            steps = ex.intermediate_values
            per_iter = []
            for rec in thoughts:
                logits = rec["logits"][0]
                top1 = topk_token_strings(tokenizer, logits, 1)
                top5 = topk_token_strings(tokenizer, logits, 5)
                top10 = topk_token_strings(tokenizer, logits, 10)
                per_iter.append({"iter": rec["iter"], "top1": top1[0], "top5": top5, "top10": top10})
            rec_out = {
                "idx": ex.idx, "n_steps": len(steps), "step_values": steps,
                "gold_answer": ex.answer, "raw_output": raw_output, "predicted_answer": pred,
                "correct": correct, "per_iter": per_iter,
            }
            decode_records.append(rec_out)
            progress_f.write(json.dumps({"idx": ex.idx, "correct": correct}) + "\n")
            if len(decode_records) % 100 == 0:
                progress_f.flush()
                print(f"decoded {len(decode_records)}/{len(examples)}", flush=True)
    decode_elapsed = time.perf_counter() - t0
    accuracy = sum(r["correct"] for r in decode_records) / len(decode_records)
    print(f"decode pass done in {decode_elapsed:.1f}s, final_answer_accuracy={accuracy:.3f}")

    # ---- held-out split: half A fits the mapping, half B reports headline numbers ---------
    if dec_args.mapping_split:
        half_a = [r for r in decode_records if r["idx"] % 2 == 0]
        half_b = [r for r in decode_records if r["idx"] % 2 == 1]
    else:
        half_a = decode_records
        half_b = decode_records
    print(f"mapping_split={dec_args.mapping_split}: half_a n={len(half_a)}, half_b n={len(half_b)}")

    max_step = max((r["n_steps"] for r in decode_records), default=0)
    # full-population matrix, for the descriptive structural finding (odd/even iterations) --
    # not used for any mapping fit or metric that gets compared against a baseline.
    hits_full_1, hits_full_5, hits_full_10 = build_matrix(decode_records, n_latents, max_step)
    matrix_top1_full = {it: {s: round(rate(hits_full_1, it, s), 4) for s in range(1, max_step + 1)} for it in range(1, n_latents + 1)}
    matrix_top5_full = {it: {s: round(rate(hits_full_5, it, s), 4) for s in range(1, max_step + 1)} for it in range(1, n_latents + 1)}
    matrix_top10_full = {it: {s: round(rate(hits_full_10, it, s), 4) for s in range(1, max_step + 1)} for it in range(1, n_latents + 1)}

    # fit on half A only
    hits_a_1, hits_a_5, hits_a_10 = build_matrix(half_a, n_latents, max_step)
    best_iter_for_step = {}
    for s in range(1, max_step + 1):
        best_it = max(range(1, n_latents + 1), key=lambda it: rate(hits_a_1, it, s))
        best_iter_for_step[s] = best_it
    live_iterations = sorted(set(best_iter_for_step.values()))
    iter_to_primary_step = {}
    for s in sorted(best_iter_for_step):
        it = best_iter_for_step[s]
        iter_to_primary_step.setdefault(it, s)  # lowest step index that maps to this iteration
    print(f"best_iter_for_step (fit on half A): {best_iter_for_step}, live_iterations={live_iterations}")

    # report on half B only, using half A's mapping
    hits_b_1, hits_b_5, hits_b_10 = build_matrix(half_b, n_latents, max_step)
    matched_c1 = sum(hits_b_1[best_iter_for_step[s]][s][0] for s in best_iter_for_step)
    matched_n1 = sum(hits_b_1[best_iter_for_step[s]][s][1] for s in best_iter_for_step)
    matched_c5 = sum(hits_b_5[best_iter_for_step[s]][s][0] for s in best_iter_for_step)
    matched_c10 = sum(hits_b_10[best_iter_for_step[s]][s][0] for s in best_iter_for_step)
    overall_matched_top1 = matched_c1 / matched_n1 if matched_n1 else 0.0
    overall_matched_top5 = matched_c5 / matched_n1 if matched_n1 else 0.0
    overall_matched_top10 = matched_c10 / matched_n1 if matched_n1 else 0.0
    matched_top1_ci = wilson_ci(matched_c1, matched_n1)
    print(f"HELD-OUT matched-iteration top1={overall_matched_top1:.4f} (95% CI {matched_top1_ci}) "
          f"top5={overall_matched_top5:.4f} top10={overall_matched_top10:.4f}  n={matched_n1}")

    # paper's own metric, correct-answers-only, on half B with half A's mapping
    paper_metric = {}
    for target_steps in (1, 2, 3):
        subset = [r for r in half_b if r["correct"] and r["n_steps"] == target_steps]
        if not subset:
            paper_metric[target_steps] = {"value": None, "n": 0}
            continue
        hits = 0
        for r in subset:
            # look up by "iter" key, not position -- `per_iter[0]` is iter 0 (the
            # pre-loop encode-pass readout) when this run's decode loop was called with
            # `include_latent0=True`, not iter 1, so a positional `[best_iter_for_step[s] - 1]`
            # index would silently grab the wrong iteration's candidates.
            all_hit = all(
                num_match(next(pi for pi in r["per_iter"] if pi["iter"] == best_iter_for_step[s])["top5"], v)
                for s, v in enumerate(r["step_values"], start=1)
            )
            hits += all_hit
        paper_metric[target_steps] = {"value": hits / len(subset), "n": len(subset)}
    print("paper-style metric (held-out, correct-only, all-steps-in-top5):", paper_metric)

    # ---- ANY-iteration variant: does the gold value appear in ANY iteration's top-k, not just
    # the one `best_iter_for_step` assigns to that step? Doesn't depend on the half-A mapping at
    # all (no circularity), so it's computed on the same half-B population directly. Iterates
    # `rec["per_iter"]` directly (not the 1..n_latents-only `build_matrix`), so when the decode
    # loop below is called with `include_latent0=True` this automatically also covers iter 0
    # (the pre-loop encode-pass readout, "latent 0" in the reference repo's own probe script --
    # never scored here before this addition).
    def any_iter_hit(rec, gold_val, k):
        field = {1: "top1", 5: "top5", 10: "top10"}[k]
        return any(num_match(pi[field] if k > 1 else [pi[field]], gold_val) for pi in rec["per_iter"])

    any_hit1 = any_hit5 = any_hit10 = any_n = 0
    for r in half_b:
        for gold_val in r["step_values"]:
            any_n += 1
            any_hit1 += any_iter_hit(r, gold_val, 1)
            any_hit5 += any_iter_hit(r, gold_val, 5)
            any_hit10 += any_iter_hit(r, gold_val, 10)
    any_iter_top1 = any_hit1 / any_n if any_n else 0.0
    any_iter_top5 = any_hit5 / any_n if any_n else 0.0
    any_iter_top10 = any_hit10 / any_n if any_n else 0.0
    any_iter_top1_ci = wilson_ci(any_hit1, any_n)
    print(f"ANY-ITERATION top1={any_iter_top1:.4f} (95% CI {any_iter_top1_ci}) top5={any_iter_top5:.4f} "
          f"top10={any_iter_top10:.4f}  n={any_n}"
          f"  (vs. matched top1={overall_matched_top1:.4f} top5={overall_matched_top5:.4f} top10={overall_matched_top10:.4f})")

    any_iter_paper_metric = {}
    any_iter_paper_metric_top10 = {}
    for target_steps in (1, 2, 3):
        subset = [r for r in half_b if r["correct"] and r["n_steps"] == target_steps]
        if not subset:
            any_iter_paper_metric[target_steps] = {"value": None, "n": 0}
            any_iter_paper_metric_top10[target_steps] = {"value": None, "n": 0}
            continue
        hits5 = sum(all(any_iter_hit(r, v, 5) for v in r["step_values"]) for r in subset)
        hits10 = sum(all(any_iter_hit(r, v, 10) for v in r["step_values"]) for r in subset)
        any_iter_paper_metric[target_steps] = {"value": hits5 / len(subset), "n": len(subset)}
        any_iter_paper_metric_top10[target_steps] = {"value": hits10 / len(subset), "n": len(subset)}
    print("paper-style metric, ANY-ITERATION top5 (held-out, correct-only, all-steps-in-any-top5):",
          any_iter_paper_metric)
    print("paper-style metric, ANY-ITERATION top10:", any_iter_paper_metric_top10)

    # ---- CHECKED-STEPS variant: CODI's own paper (Shen et al., Sec 3.5) excludes the final
    # CoT step from distillation supervision ("this behavior would undermine the quality of
    # the target hidden activations"), and Table 3's step-count buckets very likely count only
    # those checked (non-final) steps, not the raw total chain length -- bucket by
    # `n_steps - 1` and drop the final step's gold value before requiring all-hit, instead of
    # requiring every value in the full chain including the one the model was never trained to
    # represent. Combined with ANY-iteration (now including latent-0 above), this is the
    # corrected reproduction of Table 3.
    checked_paper_metric = {}
    for checked_steps in (1, 2, 3):
        subset = [r for r in half_b if r["correct"] and r["n_steps"] - 1 == checked_steps]
        if not subset:
            checked_paper_metric[checked_steps] = {"value_top5": None, "value_top10": None, "n": 0}
            continue
        vals_per_r = [r["step_values"][:-1] for r in subset]
        hits5 = sum(all(any_iter_hit(r, v, 5) for v in vals) for r, vals in zip(subset, vals_per_r))
        hits10 = sum(all(any_iter_hit(r, v, 10) for v in vals) for r, vals in zip(subset, vals_per_r))
        checked_paper_metric[checked_steps] = {
            "value_top5": hits5 / len(subset), "value_top10": hits10 / len(subset), "n": len(subset),
        }
    print("paper-style metric, CHECKED STEPS (n_steps-1, drop final step) x ANY-iteration incl. latent-0:",
          checked_paper_metric)

    # ---- train-corpus baseline (mirrors recurrent_depth's "most common step value in train") --
    train_sample = load_gsm8k_aug(split="train", n=dec_args.train_baseline_n, seed=0)
    from collections import Counter
    value_counts = Counter(v for ex in train_sample for v in ex.intermediate_values)
    mode_value, mode_count = value_counts.most_common(1)[0]
    baseline_hits = sum(
        1 for r in half_b for s, gold_val in enumerate(r["step_values"], start=1)
        if s in best_iter_for_step and num_match([mode_value], gold_val)
    )
    baseline_n = sum(1 for r in half_b for s in range(1, r["n_steps"] + 1) if s in best_iter_for_step)
    train_baseline_rate = baseline_hits / baseline_n if baseline_n else 0.0
    print(f"train-corpus baseline: mode_value={mode_value!r} (count {mode_count}/{sum(value_counts.values())} "
          f"step-positions in {len(train_sample)} train examples), matched-population rate={train_baseline_rate:.4f} "
          f"(n={baseline_n}) -- compare to matched top1={overall_matched_top1:.4f}")

    # ---- Task 2: causal patching, on half B only, stratified toward the strongest iteration --
    patch_iter_focus = dec_args.patch_iter_focus if dec_args.patch_iter_focus is not None else best_iter_for_step[1]
    print(f"patch_iter_focus={patch_iter_focus}")

    rng = random.Random(0)
    half_b_examples = {r["idx"] for r in half_b}
    examples_by_idx = {ex.idx: ex for ex in examples}
    by_steps: dict[int, list] = {}
    for ex in examples:
        if ex.idx not in half_b_examples:
            continue
        by_steps.setdefault(len(ex.intermediate_values), []).append(ex)

    focus_pairs, other_pairs = [], []
    for n_steps, exs in by_steps.items():
        if n_steps < 1 or n_steps > max_step or n_steps not in best_iter_for_step:
            continue
        for a in range(len(exs)):
            for b in range(len(exs)):
                if a == b:
                    continue
                ea, eb = exs[a], exs[b]
                for s in range(1, n_steps + 1):
                    if ea.intermediate_values[s - 1] != eb.intermediate_values[s - 1]:
                        pair = (ea, eb, s)
                        if best_iter_for_step[s] == patch_iter_focus:
                            focus_pairs.append(pair)
                        else:
                            other_pairs.append(pair)
        if len(focus_pairs) >= dec_args.n_patch_pairs * 2 and len(other_pairs) >= dec_args.n_patch_pairs:
            break
    rng.shuffle(focus_pairs)
    rng.shuffle(other_pairs)
    focus_quota = int(dec_args.n_patch_pairs * dec_args.patch_focus_frac)
    n_focus = min(focus_quota, len(focus_pairs))
    n_other = min(dec_args.n_patch_pairs - n_focus, len(other_pairs))
    pairs = focus_pairs[:n_focus] + other_pairs[:n_other]
    rng.shuffle(pairs)
    print(f"selected {len(pairs)} patch pairs ({n_focus} at focus iter {patch_iter_focus}, {n_other} at other live iters)")

    latent_cache: dict[int, list] = {}
    def get_thought_records(ex):
        if ex.idx not in latent_cache:
            q = ex.question.strip().replace("  ", " ")
            _, recs, _ = run_thoughts(model, tokenizer, q, device, n_latents)
            latent_cache[ex.idx] = recs
        return latent_cache[ex.idx]

    def control_donor(live_only: bool):
        pool = live_iterations if live_only else list(range(1, n_latents + 1))
        rand_ex = rng.choice(list(half_b_examples))
        rand_ex = examples_by_idx[rand_ex]
        rand_it = rng.choice(pool)
        recs = get_thought_records(rand_ex)
        return rand_ex, rand_it, recs[rand_it - 1]["post"]

    patch_records = []
    t1 = time.perf_counter()
    for pi, (recipient, donor, s) in enumerate(pairs):
        it = best_iter_for_step[s]
        q_r = recipient.question.strip().replace("  ", " ")

        pkv_base, thoughts_base, latent_base = run_thoughts(model, tokenizer, q_r, device, n_latents)
        out_base = decode_answer(model, tokenizer, pkv_base, latent_base, device, dec_args.max_new_tokens)

        donor_recs = get_thought_records(donor)
        donor_post = donor_recs[it - 1]["post"]
        pkv_p, thoughts_p, latent_p = run_thoughts(model, tokenizer, q_r, device, n_latents,
                                                     override_input_at={it: donor_post})
        out_patched = decode_answer(model, tokenizer, pkv_p, latent_p, device, dec_args.max_new_tokens)

        rand_ex_any, rand_it_any, rand_post_any = control_donor(live_only=False)
        pkv_ca, thoughts_ca, latent_ca = run_thoughts(model, tokenizer, q_r, device, n_latents,
                                                        override_input_at={it: rand_post_any})
        out_control_any = decode_answer(model, tokenizer, pkv_ca, latent_ca, device, dec_args.max_new_tokens)

        rand_ex_live, rand_it_live, rand_post_live = control_donor(live_only=True)
        pkv_cl, thoughts_cl, latent_cl = run_thoughts(model, tokenizer, q_r, device, n_latents,
                                                        override_input_at={it: rand_post_live})
        out_control_live = decode_answer(model, tokenizer, pkv_cl, latent_cl, device, dec_args.max_new_tokens)

        def readout_move(thoughts_variant, target_val):
            # target_val is the specific gold value the injected activation is
            # supposed to represent; None if that's undefined (e.g. control drew a
            # structurally-dead iteration with no corresponding step).
            if it >= n_latents or target_val is None:
                return None
            base_r = topk_token_strings(tokenizer, thoughts_base[it]["logits"][0], 1)[0]
            var_r = topk_token_strings(tokenizer, thoughts_variant[it]["logits"][0], 1)[0]
            return num_match([var_r], target_val) and not num_match([base_r], target_val)

        def value_for_iter(ex, iteration):
            step = iter_to_primary_step.get(iteration)
            if step is None or step > len(ex.intermediate_values):
                return None
            return ex.intermediate_values[step - 1]

        # real patch: target value is the donor's value at the ACTUAL matched step s
        # (not re-derived via iter_to_primary_step, which can point to a different step
        # when >1 step maps to the same live iteration -- s is already known exactly).
        readout_moved_real = readout_move(thoughts_p, donor.intermediate_values[s - 1])
        readout_moved_control_any = readout_move(thoughts_ca, value_for_iter(rand_ex_any, rand_it_any))
        readout_moved_control_live = readout_move(thoughts_cl, value_for_iter(rand_ex_live, rand_it_live))

        pred_base = extract_final_number(out_base)
        pred_patched = extract_final_number(out_patched)
        pred_control_any = extract_final_number(out_control_any)
        pred_control_live = extract_final_number(out_control_live)
        patch_records.append({
            "recipient_idx": recipient.idx, "donor_idx": donor.idx, "step": s, "iter": it,
            "is_focus_iter": it == patch_iter_focus,
            "recipient_gold": recipient.answer, "recipient_value_at_step": recipient.intermediate_values[s - 1],
            "donor_value_at_step": donor.intermediate_values[s - 1],
            "answer_base": pred_base, "answer_patched": pred_patched,
            "answer_control_any": pred_control_any, "answer_control_live": pred_control_live,
            "answer_changed_by_patch": pred_patched != pred_base,
            "answer_changed_by_control_any": pred_control_any != pred_base,
            "answer_changed_by_control_live": pred_control_live != pred_base,
            "readout_moved_toward_donor_real": readout_moved_real,
            "readout_moved_toward_donor_control_any": readout_moved_control_any,
            "readout_moved_toward_donor_control_live": readout_moved_control_live,
        })
        if (pi + 1) % 25 == 0:
            with open("/tmp/codi_patch_progress.jsonl", "w") as f:
                for r in patch_records:
                    f.write(json.dumps(r, default=str) + "\n")
            elapsed = time.perf_counter() - t1
            print(f"patched {pi + 1}/{len(pairs)} ({elapsed:.0f}s elapsed)", flush=True)

    n_pairs = len(patch_records)

    def frac(key):
        return sum(r[key] for r in patch_records) / n_pairs if n_pairs else 0.0

    def wilson_for(key):
        x = sum(r[key] for r in patch_records)
        return x, n_pairs, wilson_ci(x, n_pairs)

    frac_patch, frac_ctrl_any, frac_ctrl_live = frac("answer_changed_by_patch"), frac("answer_changed_by_control_any"), frac("answer_changed_by_control_live")

    def mcnemar_for(key_a, key_b):
        b = sum(1 for r in patch_records if r[key_a] and not r[key_b])
        c = sum(1 for r in patch_records if r[key_b] and not r[key_a])
        return b, c, mcnemar_exact_p(b, c)

    b_any, c_any, p_any = mcnemar_for("answer_changed_by_patch", "answer_changed_by_control_any")
    b_live, c_live, p_live = mcnemar_for("answer_changed_by_patch", "answer_changed_by_control_live")

    def readout_frac(key):
        vals = [r[key] for r in patch_records if r[key] is not None]
        return (sum(vals) / len(vals) if vals else None), len(vals)

    ro_real, n_ro_real = readout_frac("readout_moved_toward_donor_real")
    ro_ctrl_any, n_ro_ctrl_any = readout_frac("readout_moved_toward_donor_control_any")
    ro_ctrl_live, n_ro_ctrl_live = readout_frac("readout_moved_toward_donor_control_live")

    focus_records = [r for r in patch_records if r["is_focus_iter"]]
    other_records = [r for r in patch_records if not r["is_focus_iter"]]
    def sub_frac(records, key):
        return (sum(r[key] for r in records) / len(records)) if records else None

    print(f"n_pairs={n_pairs}  answer_changed: patch={frac_patch:.3f} (x={wilson_for('answer_changed_by_patch')}), "
          f"control_any={frac_ctrl_any:.3f}, control_live={frac_ctrl_live:.3f}")
    print(f"McNemar patch-vs-control_any: b={b_any} c={c_any} p={p_any:.4f}")
    print(f"McNemar patch-vs-control_live: b={b_live} c={c_live} p={p_live:.4f}")
    print(f"readout_moved_toward_donor: real={ro_real} (n={n_ro_real}), control_any={ro_ctrl_any} (n={n_ro_ctrl_any}), "
          f"control_live={ro_ctrl_live} (n={n_ro_ctrl_live})")
    print(f"focus-iter ({patch_iter_focus}) n={len(focus_records)}: answer_changed patch={sub_frac(focus_records, 'answer_changed_by_patch')}, "
          f"control_any={sub_frac(focus_records, 'answer_changed_by_control_any')}")
    print(f"other-iters n={len(other_records)}: answer_changed patch={sub_frac(other_records, 'answer_changed_by_patch')}, "
          f"control_any={sub_frac(other_records, 'answer_changed_by_control_any')}")

    # ---- Log the run ------------------------------------------------------------------------
    metrics = {
        "final_answer_accuracy": accuracy,
        "unparseable_rate": sum(r["predicted_answer"] is None for r in decode_records) / len(decode_records),
        "compute_steps": n_latents,
        "sec_per_example": decode_elapsed / len(decode_records),
        "decoding_accuracy": overall_matched_top1,
        "intervention_accuracy": ro_real,
        "extra": {
            "held_out_split": dec_args.mapping_split,
            "half_a_n": len(half_a), "half_b_n": len(half_b),
            "decoding_accuracy_matched_top1": overall_matched_top1,
            "decoding_accuracy_matched_top1_wilson_ci": list(matched_top1_ci),
            "decoding_accuracy_matched_top5": overall_matched_top5,
            "decoding_accuracy_matched_top10": overall_matched_top10,
            "decoding_accuracy_matched_n": matched_n1,
            "decoding_matrix_top1_by_iter_step_FULL_POPULATION": matrix_top1_full,
            "decoding_matrix_top5_by_iter_step_FULL_POPULATION": matrix_top5_full,
            "decoding_matrix_top10_by_iter_step_FULL_POPULATION": matrix_top10_full,
            "best_iter_for_step_fit_on_half_A": best_iter_for_step,
            "live_iterations": live_iterations,
            "paper_style_metric_held_out_correct_only_by_step_count": paper_metric,
            "decoding_accuracy_any_iteration_top1": any_iter_top1,
            "decoding_accuracy_any_iteration_top1_wilson_ci": list(any_iter_top1_ci),
            "decoding_accuracy_any_iteration_top5": any_iter_top5,
            "decoding_accuracy_any_iteration_top10": any_iter_top10,
            "decoding_accuracy_any_iteration_n": any_n,
            "paper_style_metric_any_iteration_top5_by_step_count": any_iter_paper_metric,
            "paper_style_metric_any_iteration_top10_by_step_count": any_iter_paper_metric_top10,
            "decode_loop_includes_latent0": True,
            "paper_style_metric_checked_steps_n_minus_1_drop_final_any_iteration": checked_paper_metric,
            "train_baseline_mode_value": mode_value,
            "train_baseline_mode_count": mode_count,
            "train_baseline_sample_n": len(train_sample),
            "train_baseline_rate_matched_population": train_baseline_rate,
            "patch_iter_focus": patch_iter_focus,
            "patch_focus_frac_requested": dec_args.patch_focus_frac,
            "n_patch_pairs": n_pairs,
            "n_patch_pairs_focus": len(focus_records),
            "n_patch_pairs_other": len(other_records),
            "answer_changed_by_patch": frac_patch,
            "answer_changed_by_control_any": frac_ctrl_any,
            "answer_changed_by_control_live": frac_ctrl_live,
            "answer_changed_wilson_ci_patch": list(wilson_ci(sum(r["answer_changed_by_patch"] for r in patch_records), n_pairs)),
            "answer_changed_wilson_ci_control_any": list(wilson_ci(sum(r["answer_changed_by_control_any"] for r in patch_records), n_pairs)),
            "answer_changed_wilson_ci_control_live": list(wilson_ci(sum(r["answer_changed_by_control_live"] for r in patch_records), n_pairs)),
            "mcnemar_patch_vs_control_any": {"b": b_any, "c": c_any, "p_exact": p_any},
            "mcnemar_patch_vs_control_live": {"b": b_live, "c": c_live, "p_exact": p_live},
            "readout_moved_toward_donor_real": ro_real,
            "readout_moved_toward_donor_control_any": ro_ctrl_any,
            "readout_moved_toward_donor_control_live": ro_ctrl_live,
            "answer_changed_patch_focus_iter": sub_frac(focus_records, "answer_changed_by_patch"),
            "answer_changed_control_any_focus_iter": sub_frac(focus_records, "answer_changed_by_control_any"),
            "answer_changed_patch_other_iters": sub_frac(other_records, "answer_changed_by_patch"),
            "answer_changed_control_any_other_iters": sub_frac(other_records, "answer_changed_by_control_any"),
            "n_trainable_params": n_trainable,
            "pilot_run_id": "20260919-073312_codi_decode-patch-pilot",
        },
    }

    record = RunRecord(
        run_id=new_run_id("codi", dec_args.slug),
        mechanism="codi",
        stage=dec_args.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=dec_args.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=dec_args.eval_seed),
        metrics=metrics,
        hyperparams={"inf_latent_iterations": n_latents, "greedy": training_args.greedy,
                     "num_latent": training_args.num_latent, "n_patch_pairs": dec_args.n_patch_pairs,
                     "patch_iter_focus": patch_iter_focus, "mapping_split": dec_args.mapping_split},
        seed=None,
        hardware=f"{dec_args.hardware} / {torch.cuda.get_device_name(0)}",
        notes="CODI decoding (full test set, held-out mapping split) + causal patching "
              "(scaled, stratified, two control variants, McNemar + Wilson CIs). "
              "Follow-up to the pilot run.",
    )
    manifest = record.save(predictions=decode_records)
    out_dir = manifest.parent
    with (out_dir / "patch_pairs.jsonl").open("w") as f:
        for r in patch_records:
            f.write(json.dumps(r, default=str) + "\n")
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
