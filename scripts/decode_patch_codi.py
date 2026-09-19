#!/usr/bin/env python3
"""CODI decodability + causal-patching pilot -- the faithfulness-spine contrast case to
recurrent_depth's step-supervised run (results/20260918-214656_recurrent_depth_stepsup-*).

Reuses `scripts/eval_codi.py`'s model loading exactly (same released checkpoint, same
paper protocol). Two things eval_codi.py doesn't do:

1. Decoding: at each of the 6 loop iterations, `outputs.logits[:, -1, :]` computed in
   that SAME forward pass IS the paper's own decode method (Sec 5.1: "projecting its
   last hidden state into vocabulary space via the model's word embeddings") -- GPT-2
   ties wte/lm_head and LoRA doesn't touch either, so `logits = lm_head(hidden_state)`
   is exactly that projection, already computed, no extra work. We report both the
   paper's own metric (top-5, conditioned on correct final answer, by step count -- a
   sanity check against their 97.1/83.9/75.0) and the metric comparable to
   recurrent_depth's 31.3% (top-1 and top-5, UNCONDITIONAL on correctness, broken down
   by iteration x step-count, to find the iteration<->step mapping empirically -- the
   paper's own case study suggests placeholder tokens interleave with real ones, not a
   clean 1:1 diagonal like recurrent_depth's loop).

2. Causal patching: swap one thought z_i (post-projection -- the actual value fed
   forward, per generate_batches) for a donor example's z_i computed from the donor's
   own independent context, keep the recipient's own KV cache/trajectory otherwise
   unchanged, let iterations i+1..6 and the final answer proceed from the patched
   value. Control: patch with a latent from a random unrelated (example, iteration)
   pair instead of a real donor at the matched step, to separate "any perturbation
   moves the answer" from "this specific counterfactual content moves the answer".

Run inside the CODI venv, from the CODI checkout (same convention as eval_codi.py):

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/decode_patch_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug decode-patch-pilot --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --n_patch_pairs 30
"""
from __future__ import annotations

import json
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
    slug: str = field(default="decode-patch-pilot")
    stage: str = field(default="pilot")
    eval_n: int = field(default=200)
    eval_seed: int = field(default=0)
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    n_patch_pairs: int = field(default=30)
    max_new_tokens: int = field(default=64)


def encode_question(model, tokenizer, question: str, device: str):
    batch = tokenizer([question], return_tensors="pt", padding="longest")
    bot = torch.tensor([[model.bot_id]], dtype=torch.long)
    input_ids = torch.cat([batch["input_ids"], bot], dim=1).to(device)
    attn = torch.cat([batch["attention_mask"], torch.ones_like(bot)], dim=1).to(device)
    return input_ids, attn


@torch.no_grad()
def run_thoughts(model, tokenizer, question: str, device: str, n_iters: int,
                  override_input_at: dict[int, torch.Tensor] | None = None):
    """Encode `question`, run `n_iters` loop iterations (1-indexed: iter 1..n_iters).
    `override_input_at`: {iter_index: post_proj_tensor} -- if iter i is a key, the
    computed post-proj latent that would normally feed iteration i is replaced by the
    given tensor before running that iteration's forward pass (a causal patch).
    Returns (past_key_values, records, final_post_proj_latent_for_next_step).
    records[i-1] = {"iter": i, "logits": [vocab], "post": tensor} for the *output* of
    iteration i (the i-th continuous thought z_i)."""
    input_ids, attn = encode_question(model, tokenizer, question, device)
    outputs = model.codi(input_ids=input_ids, use_cache=True, output_hidden_states=True,
                          past_key_values=None, attention_mask=attn)
    pkv = outputs.past_key_values
    latent = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    if model.use_prj:
        latent = model.prj(latent)

    records = []
    for it in range(1, n_iters + 1):
        feed = latent
        if override_input_at is not None and it in override_input_at:
            feed = override_input_at[it]
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


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DecodeArguments))
    model_args, data_args, training_args, dec_args = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")

    examples = load_gsm8k_aug(split="test", n=dec_args.eval_n, seed=dec_args.eval_seed)
    n_latents = training_args.inf_latent_iterations

    # ---- Task 1: decode all 200 examples ------------------------------------------------
    t0 = time.perf_counter()
    decode_records = []
    for ex in examples:
        question = ex.question.strip().replace("  ", " ")
        pkv, thoughts, latent = run_thoughts(model, tokenizer, question, device, n_latents)
        raw_output = decode_answer(model, tokenizer, pkv, latent, device, dec_args.max_new_tokens)
        pred = extract_final_number(raw_output)
        correct = is_correct(pred, ex.answer)
        steps = ex.intermediate_values
        per_iter = []
        for rec in thoughts:
            logits = rec["logits"][0]
            top1 = topk_token_strings(tokenizer, logits, 1)
            top5 = topk_token_strings(tokenizer, logits, 5)
            per_iter.append({"iter": rec["iter"], "top1": top1[0], "top5": top5})
        decode_records.append({
            "idx": ex.idx, "n_steps": len(steps), "step_values": steps,
            "gold_answer": ex.answer, "raw_output": raw_output, "predicted_answer": pred,
            "correct": correct, "per_iter": per_iter,
        })
        if len(decode_records) % 25 == 0:
            print(f"decoded {len(decode_records)}/{len(examples)}", flush=True)
    decode_elapsed = time.perf_counter() - t0
    accuracy = sum(r["correct"] for r in decode_records) / len(decode_records)
    print(f"decode pass done in {decode_elapsed:.1f}s, final_answer_accuracy={accuracy:.3f}")

    # ---- Task 1 analysis: iteration x step-index confusion matrix (unconditional) -------
    max_step = max((r["n_steps"] for r in decode_records), default=0)
    hits_uncond_top1 = {i: {s: [0, 0] for s in range(1, max_step + 1)} for i in range(1, n_latents + 1)}
    hits_uncond_top5 = {i: {s: [0, 0] for s in range(1, max_step + 1)} for i in range(1, n_latents + 1)}
    for r in decode_records:
        for s_idx, gold_val in enumerate(r["step_values"], start=1):
            for pi in r["per_iter"]:
                it = pi["iter"]
                hits_uncond_top1[it][s_idx][1] += 1
                hits_uncond_top5[it][s_idx][1] += 1
                if num_match([pi["top1"]], gold_val):
                    hits_uncond_top1[it][s_idx][0] += 1
                if num_match(pi["top5"], gold_val):
                    hits_uncond_top5[it][s_idx][0] += 1

    def rate(hits, it, s):
        c, n = hits[it][s]
        return c / n if n else 0.0

    matrix_top1 = {it: {s: round(rate(hits_uncond_top1, it, s), 3) for s in range(1, max_step + 1)} for it in range(1, n_latents + 1)}
    matrix_top5 = {it: {s: round(rate(hits_uncond_top5, it, s), 3) for s in range(1, max_step + 1)} for it in range(1, n_latents + 1)}
    # empirical best iteration per step index (top-1 unconditional match rate), for patching target selection
    best_iter_for_step = {}
    for s in range(1, max_step + 1):
        best_it = max(range(1, n_latents + 1), key=lambda it: rate(hits_uncond_top1, it, s))
        best_iter_for_step[s] = best_it

    # unconditional overall top1/top5 (all iters, all steps, all examples) -- the number
    # comparable to recurrent_depth's 31.3%
    total_c1 = sum(hits_uncond_top1[it][s][0] for it in hits_uncond_top1 for s in hits_uncond_top1[it])
    total_n1 = sum(hits_uncond_top1[it][s][1] for it in hits_uncond_top1 for s in hits_uncond_top1[it])
    total_c5 = sum(hits_uncond_top5[it][s][0] for it in hits_uncond_top5 for s in hits_uncond_top5[it])
    overall_uncond_top1 = total_c1 / total_n1 if total_n1 else 0.0
    overall_uncond_top5 = total_c5 / total_n1 if total_n1 else 0.0

    # "matched" variant: only the empirically-best iteration per step (best_iter_for_step),
    # not averaged over the full 6-iter x n-step grid -- CODI has structurally dead cells
    # (placeholder iterations that never decode to a number, and step counts beyond what 6
    # latents can encode at all), which dilute the unconditional-over-everything number
    # above into something NOT comparable to recurrent_depth's 31.3% (which has no such
    # dead cells). This is the number to actually put next to it.
    matched_c1 = sum(hits_uncond_top1[best_iter_for_step[s]][s][0] for s in best_iter_for_step)
    matched_n1 = sum(hits_uncond_top1[best_iter_for_step[s]][s][1] for s in best_iter_for_step)
    matched_c5 = sum(hits_uncond_top5[best_iter_for_step[s]][s][0] for s in best_iter_for_step)
    overall_matched_top1 = matched_c1 / matched_n1 if matched_n1 else 0.0
    overall_matched_top5 = matched_c5 / matched_n1 if matched_n1 else 0.0

    # ---- Task 1 sanity check: paper's own metric (top-5, correct-answer-conditioned) ----
    paper_metric = {}
    for target_steps in (1, 2, 3):
        subset = [r for r in decode_records if r["correct"] and r["n_steps"] == target_steps]
        if not subset:
            paper_metric[target_steps] = None
            continue
        hits = 0
        for r in subset:
            # paper matches "top-5 intermediate results" against reference -- interpret as:
            # every gold step value appears in SOME iteration's top-5, using the empirical
            # best-iteration-per-step mapping found above.
            all_hit = all(num_match(r["per_iter"][best_iter_for_step[s] - 1]["top5"], v)
                           for s, v in enumerate(r["step_values"], start=1))
            hits += all_hit
        paper_metric[target_steps] = hits / len(subset)

    print("iteration x step top1 matrix:", json.dumps(matrix_top1))
    print("iteration x step top5 matrix:", json.dumps(matrix_top5))
    print("best_iter_for_step:", best_iter_for_step)
    print(f"overall unconditional top1={overall_uncond_top1:.3f} top5={overall_uncond_top5:.3f}")
    print(f"overall matched-iteration top1={overall_matched_top1:.3f} top5={overall_matched_top5:.3f} "
          f"(the number comparable to recurrent_depth's 31.3%)")
    print("paper-style metric (correct-only, all-steps-in-top5):", paper_metric)

    # ---- Task 2: causal patching ---------------------------------------------------------
    rng = random.Random(0)
    by_steps: dict[int, list] = {}
    for ex in examples:
        by_steps.setdefault(len(ex.intermediate_values), []).append(ex)

    pairs = []
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
                        pairs.append((ea, eb, s))
        if len(pairs) >= dec_args.n_patch_pairs * 3:
            break
    rng.shuffle(pairs)
    pairs = pairs[: dec_args.n_patch_pairs]
    print(f"selected {len(pairs)} patch pairs")

    # cache full 6-iteration decode of every example touched, to reuse post-proj latents as
    # donor values without recomputing per pair
    latent_cache: dict[int, list] = {}
    def get_thought_records(ex):
        if ex.idx not in latent_cache:
            q = ex.question.strip().replace("  ", " ")
            _, recs, _ = run_thoughts(model, tokenizer, q, device, n_latents)
            latent_cache[ex.idx] = recs
        return latent_cache[ex.idx]

    all_examples_by_idx = {ex.idx: ex for ex in examples}
    patch_records = []
    for recipient, donor, s in pairs:
        it = best_iter_for_step[s]
        q_r = recipient.question.strip().replace("  ", " ")

        # baseline: recipient's own unpatched trajectory
        pkv_base, thoughts_base, latent_base = run_thoughts(model, tokenizer, q_r, device, n_latents)
        out_base = decode_answer(model, tokenizer, pkv_base, latent_base, device, dec_args.max_new_tokens)

        # donor's post-proj z_it, from the donor's own independent context
        donor_recs = get_thought_records(donor)
        donor_post = donor_recs[it - 1]["post"]

        # real patch: recipient's own cache through iter it-1, donor's z_it as input to iter it
        pkv_p, thoughts_p, latent_p = run_thoughts(model, tokenizer, q_r, device, n_latents,
                                                     override_input_at={it: donor_post})
        out_patched = decode_answer(model, tokenizer, pkv_p, latent_p, device, dec_args.max_new_tokens)

        # control: unrelated random (example, iteration) post-proj latent, not the matched donor
        rand_ex = rng.choice(examples)
        rand_it = rng.randint(1, n_latents)
        rand_recs = get_thought_records(rand_ex)
        rand_post = rand_recs[rand_it - 1]["post"]
        pkv_c, thoughts_c, latent_c = run_thoughts(model, tokenizer, q_r, device, n_latents,
                                                     override_input_at={it: rand_post})
        out_control = decode_answer(model, tokenizer, pkv_c, latent_c, device, dec_args.max_new_tokens)

        # read-out shift check: iteration it+1's top1 (if it exists), before/after patch
        readout_shift = None
        if it < n_latents:
            base_r = topk_token_strings(tokenizer, thoughts_base[it]["logits"][0], 1)[0]
            patch_r = topk_token_strings(tokenizer, thoughts_p[it]["logits"][0], 1)[0]
            readout_shift = {"base_next_readout": base_r, "patched_next_readout": patch_r,
                              "donor_value": donor.intermediate_values[s - 1],
                              "moved_toward_donor": num_match([patch_r], donor.intermediate_values[s - 1]) and not num_match([base_r], donor.intermediate_values[s - 1])}

        pred_base = extract_final_number(out_base)
        pred_patched = extract_final_number(out_patched)
        pred_control = extract_final_number(out_control)
        patch_records.append({
            "recipient_idx": recipient.idx, "donor_idx": donor.idx, "step": s, "iter": it,
            "recipient_gold": recipient.answer, "recipient_value_at_step": recipient.intermediate_values[s - 1],
            "donor_value_at_step": donor.intermediate_values[s - 1],
            "answer_base": pred_base, "answer_patched": pred_patched, "answer_control": pred_control,
            "answer_changed_by_patch": pred_patched != pred_base,
            "answer_changed_by_control": pred_control != pred_base,
            "readout_shift": readout_shift,
        })

    n_pairs = len(patch_records)
    frac_answer_changed_patch = sum(r["answer_changed_by_patch"] for r in patch_records) / n_pairs if n_pairs else 0.0
    frac_answer_changed_control = sum(r["answer_changed_by_control"] for r in patch_records) / n_pairs if n_pairs else 0.0
    readouts_with_shift = [r["readout_shift"] for r in patch_records if r["readout_shift"] is not None]
    frac_readout_moved_toward_donor = (
        sum(rs["moved_toward_donor"] for rs in readouts_with_shift) / len(readouts_with_shift)
        if readouts_with_shift else None
    )
    print(f"patch pairs n={n_pairs}: answer changed by real patch = {frac_answer_changed_patch:.3f}, "
          f"by control patch = {frac_answer_changed_control:.3f}, "
          f"next-readout moved toward donor = {frac_readout_moved_toward_donor}")

    # ---- Log the run ----------------------------------------------------------------------
    metrics = {
        "final_answer_accuracy": accuracy,
        "unparseable_rate": sum(r["predicted_answer"] is None for r in decode_records) / len(decode_records),
        "compute_steps": n_latents,
        "sec_per_example": decode_elapsed / len(decode_records),
        "decoding_accuracy": overall_matched_top1,  # canonical top-level key, per record-run SKILL.md
        "intervention_accuracy": frac_readout_moved_toward_donor,  # canonical top-level key
        "extra": {
            "decoding_accuracy_matched_top1": overall_matched_top1,
            "decoding_accuracy_matched_top5": overall_matched_top5,
            "decoding_accuracy_unconditional_top1": overall_uncond_top1,
            "decoding_accuracy_unconditional_top5": overall_uncond_top5,
            "decoding_matrix_top1_by_iter_step": matrix_top1,
            "decoding_matrix_top5_by_iter_step": matrix_top5,
            "best_iter_for_step": best_iter_for_step,
            "paper_style_metric_correct_only_by_step_count": paper_metric,
            "intervention_accuracy_answer_changed_by_patch": frac_answer_changed_patch,
            "intervention_accuracy_answer_changed_by_control": frac_answer_changed_control,
            "intervention_readout_moved_toward_donor": frac_readout_moved_toward_donor,
            "n_patch_pairs": n_pairs,
            "n_trainable_params": n_trainable,
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
                     "num_latent": training_args.num_latent, "n_patch_pairs": dec_args.n_patch_pairs},
        seed=None,
        hardware=f"{dec_args.hardware} / {torch.cuda.get_device_name(0)}",
        notes="CODI decoding (logit-lens on pre-projection hidden state, all 6 iterations) "
              "+ causal patching pilot (post-projection latent swap vs. random control).",
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
