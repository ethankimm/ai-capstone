#!/usr/bin/env python3
"""CODI diagnostics for the causal-patching null (results/20260919-080349_codi_decode-patch-full):
does the focus position (iteration 2, the strongest decode position) matter AT ALL
independent of content (ablation), and does the model even attend to it when answering
(attention check)? Reuses the exact 227 focus-iteration pairs from that run's
patch_pairs.jsonl so the ablation conditions are paired with the already-known
real-donor / control-any / control-live outcomes on the SAME recipients.

Run inside the CODI venv, from the CODI checkout (same convention as decode_patch_codi.py):

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/ablate_attention_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug ablate-attn --stage full_run --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --n_attn_examples 80 --mean_sample_n 300
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
from decode_patch_codi import (  # noqa: E402
    encode_question, run_thoughts, decode_answer, topk_token_strings,
    wilson_ci, mcnemar_exact_p,
)

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

PRIOR_RUN = "20260919-080349_codi_decode-patch-full"


@dataclass
class DiagArguments:
    slug: str = field(default="ablate-attn")
    stage: str = field(default="full_run")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    mean_sample_n: int = field(default=300)
    n_attn_examples: int = field(default=80)
    max_new_tokens: int = field(default=64)


@torch.no_grad()
def mean_latent_at_iter(model, tokenizer, examples, device, n_latents, target_iter, sample_n, seed=0):
    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)
    pool = pool[:sample_n]
    acc = None
    for ex in pool:
        q = ex.question.strip().replace("  ", " ")
        _, recs, _ = run_thoughts(model, tokenizer, q, device, n_latents)
        v = recs[target_iter - 1]["post"].detach().float()
        acc = v.clone() if acc is None else acc + v
    model_dtype = next(model.parameters()).dtype
    return (acc / len(pool)).to(device=device, dtype=model_dtype)


@torch.no_grad()
def attention_to_positions(model, tokenizer, question: str, device: str, n_latents: int,
                            focus_iter: int, max_new_tokens: int):
    """Honest (unpatched) generation with output_attentions=True during the answer phase.
    Returns per-generated-token attention mass (averaged over layers and heads) to:
    question-token positions, each thought position 1..n_latents, the eot position,
    and previously-generated answer-token positions. Position bookkeeping: the initial
    forward processes `L0` positions (question + bot); iteration i's thought occupies
    absolute position `L0 + i - 1`; eot occupies `L0 + n_latents`; the k-th generated
    answer token occupies `L0 + n_latents + 1 + k`."""
    input_ids, attn = encode_question(model, tokenizer, question, device)
    L0 = input_ids.shape[1]
    outputs = model.codi(input_ids=input_ids, use_cache=True, output_hidden_states=True,
                          past_key_values=None, attention_mask=attn)
    pkv = outputs.past_key_values
    latent = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    if model.use_prj:
        latent = model.prj(latent)
    for it in range(1, n_latents + 1):
        outputs = model.codi(inputs_embeds=latent, use_cache=True, output_hidden_states=True, past_key_values=pkv)
        pkv = outputs.past_key_values
        pre_hidden = outputs.hidden_states[-1][:, -1, :]
        latent = model.prj(pre_hidden.unsqueeze(1)) if model.use_prj else pre_hidden.unsqueeze(1)

    eot_emb = model.get_embd(model.codi, model.model_name)(
        torch.tensor([model.eot_id], dtype=torch.long, device=device)
    ).unsqueeze(0)
    output = eot_emb
    thought_pos = {i: L0 + i - 1 for i in range(1, n_latents + 1)}
    focus_pos = thought_pos[focus_iter]
    eot_pos = L0 + n_latents

    per_token_mass = []  # list of dicts: {question, focus_thought, other_thoughts, eot, prior_answer}
    pred_tokens: list[int] = []
    for step_i in range(max_new_tokens):
        out = model.codi(inputs_embeds=output, output_hidden_states=False, attention_mask=None,
                          use_cache=True, past_key_values=pkv, output_attentions=True)
        pkv = out.past_key_values
        logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]
        next_id = torch.argmax(logits, dim=-1)
        if next_id.item() == tokenizer.eos_token_id:
            break

        if out.attentions is not None:
            # attentions: tuple[n_layers] of [batch, heads, q_len=1, k_len] -- mean over layers & heads
            layer_stack = torch.stack([a[0].mean(dim=0).squeeze(0).float() for a in out.attentions])  # [layers, k_len]
            mass = layer_stack.mean(dim=0).cpu()  # [k_len], averaged over layers and heads, sums to ~1
            k_len = mass.shape[0]
            q_mass = mass[:L0].sum().item()
            focus_mass = mass[focus_pos].item() if focus_pos < k_len else 0.0
            other_thought_mass = sum(mass[p].item() for i, p in thought_pos.items() if i != focus_iter and p < k_len)
            eot_mass = mass[eot_pos].item() if eot_pos < k_len else 0.0
            prior_answer_mass = mass[eot_pos + 1:].sum().item() if k_len > eot_pos + 1 else 0.0
            per_token_mass.append({
                "step": step_i, "question": q_mass, "focus_thought": focus_mass,
                "other_thoughts": other_thought_mass, "eot": eot_mass, "prior_answer": prior_answer_mass,
                "k_len": k_len,
            })

        pred_tokens.append(next_id.item())
        output = model.get_embd(model.codi, model.model_name)(next_id).unsqueeze(1).to(device)

    raw_output = tokenizer.decode(pred_tokens, skip_special_tokens=True)
    return raw_output, per_token_mass


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DiagArguments))
    model_args, data_args, training_args, diag_args = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")
    n_latents = training_args.inf_latent_iterations

    prior_dir = Path(__file__).resolve().parent.parent / "results" / PRIOR_RUN
    manifest = json.loads((prior_dir / "manifest.json").read_text())
    extra = manifest["metrics"]["extra"]
    best_iter_for_step = {int(k): v for k, v in extra["best_iter_for_step_fit_on_half_A"].items()}
    focus_iter = extra["patch_iter_focus"]
    focus_pairs = [json.loads(l) for l in (prior_dir / "patch_pairs.jsonl").open() if json.loads(l)["is_focus_iter"]]
    print(f"loaded {len(focus_pairs)} focus-iteration (iter={focus_iter}) pairs from {PRIOR_RUN}")

    all_test = load_gsm8k_aug(split="test", n=None, seed=None)
    by_idx = {ex.idx: ex for ex in all_test}
    half_b = [ex for ex in all_test if ex.idx % 2 == 1]  # same split rule as decode_patch_codi.py

    # ---- Task 1: ablation, on the exact same 227 focus-iter pairs as the prior run ----------
    t0 = time.perf_counter()
    print(f"computing mean latent at iter {focus_iter} over {diag_args.mean_sample_n} half-B examples...")
    mean_vec = mean_latent_at_iter(model, tokenizer, half_b, device, n_latents, focus_iter, diag_args.mean_sample_n)
    zero_vec = torch.zeros_like(mean_vec)
    print(f"mean/zero vectors ready ({time.perf_counter() - t0:.1f}s)")

    ablate_records = []
    t1 = time.perf_counter()
    for pi, fp in enumerate(focus_pairs):
        recipient = by_idx[fp["recipient_idx"]]
        s = fp["step"]
        it = fp["iter"]
        q_r = recipient.question.strip().replace("  ", " ")

        pkv_b, thoughts_b, latent_b = run_thoughts(model, tokenizer, q_r, device, n_latents)
        out_b = decode_answer(model, tokenizer, pkv_b, latent_b, device, diag_args.max_new_tokens)
        base_readout = topk_token_strings(tokenizer, thoughts_b[it]["logits"][0], 1)[0] if it < n_latents else None

        pkv_z, thoughts_z, latent_z = run_thoughts(model, tokenizer, q_r, device, n_latents, override_input_at={it: zero_vec})
        out_z = decode_answer(model, tokenizer, pkv_z, latent_z, device, diag_args.max_new_tokens)
        readout_z = topk_token_strings(tokenizer, thoughts_z[it]["logits"][0], 1)[0] if it < n_latents else None

        pkv_m, thoughts_m, latent_m = run_thoughts(model, tokenizer, q_r, device, n_latents, override_input_at={it: mean_vec})
        out_m = decode_answer(model, tokenizer, pkv_m, latent_m, device, diag_args.max_new_tokens)
        readout_m = topk_token_strings(tokenizer, thoughts_m[it]["logits"][0], 1)[0] if it < n_latents else None

        pred_b, pred_z, pred_m = extract_final_number(out_b), extract_final_number(out_z), extract_final_number(out_m)
        ablate_records.append({
            "recipient_idx": fp["recipient_idx"], "step": s, "iter": it,
            "answer_base": pred_b, "answer_zero": pred_z, "answer_mean": pred_m,
            "answer_changed_by_zero": pred_z != pred_b, "answer_changed_by_mean": pred_m != pred_b,
            "readout_changed_by_zero": (readout_z != base_readout) if base_readout is not None else None,
            "readout_changed_by_mean": (readout_m != base_readout) if base_readout is not None else None,
            # carry over the prior run's already-computed conditions on this SAME unit for paired stats
            "answer_changed_by_patch": fp["answer_changed_by_patch"],
            "answer_changed_by_control_any": fp["answer_changed_by_control_any"],
        })
        if (pi + 1) % 50 == 0:
            print(f"ablated {pi + 1}/{len(focus_pairs)} ({time.perf_counter() - t1:.0f}s elapsed)", flush=True)
    ablate_elapsed = time.perf_counter() - t1
    n_ab = len(ablate_records)

    def frac(key):
        return sum(r[key] for r in ablate_records) / n_ab if n_ab else 0.0

    def mcnemar_for(records, key_a, key_b):
        b = sum(1 for r in records if r[key_a] and not r[key_b])
        c = sum(1 for r in records if r[key_b] and not r[key_a])
        return b, c, mcnemar_exact_p(b, c)

    frac_zero, frac_mean = frac("answer_changed_by_zero"), frac("answer_changed_by_mean")
    frac_patch, frac_ctrl = frac("answer_changed_by_patch"), frac("answer_changed_by_control_any")
    b_zp, c_zp, p_zp = mcnemar_for(ablate_records, "answer_changed_by_zero", "answer_changed_by_patch")
    b_mp, c_mp, p_mp = mcnemar_for(ablate_records, "answer_changed_by_mean", "answer_changed_by_patch")
    b_zc, c_zc, p_zc = mcnemar_for(ablate_records, "answer_changed_by_zero", "answer_changed_by_control_any")

    print(f"n={n_ab}: answer_changed  patch={frac_patch:.3f} control_any={frac_ctrl:.3f} "
          f"zero_ablate={frac_zero:.3f} mean_ablate={frac_mean:.3f}")
    print(f"McNemar zero-vs-patch: b={b_zp} c={c_zp} p={p_zp:.4f}")
    print(f"McNemar mean-vs-patch: b={b_mp} c={c_mp} p={p_mp:.4f}")
    print(f"McNemar zero-vs-control_any: b={b_zc} c={c_zc} p={p_zc:.4f}")
    readout_z_frac = sum(1 for r in ablate_records if r["readout_changed_by_zero"]) / sum(1 for r in ablate_records if r["readout_changed_by_zero"] is not None)
    readout_m_frac = sum(1 for r in ablate_records if r["readout_changed_by_mean"]) / sum(1 for r in ablate_records if r["readout_changed_by_mean"] is not None)
    print(f"readout_changed_at_all: zero={readout_z_frac:.3f} mean={readout_m_frac:.3f}")

    # ---- Task 2: attention check, fresh honest sample from half B ---------------------------
    rng2 = random.Random(1)
    attn_pool = list(half_b)
    rng2.shuffle(attn_pool)
    attn_sample = attn_pool[:diag_args.n_attn_examples]

    t2 = time.perf_counter()
    all_token_mass = []
    n_ok = 0
    for i, ex in enumerate(attn_sample):
        q = ex.question.strip().replace("  ", " ")
        try:
            _, per_token_mass = attention_to_positions(model, tokenizer, q, device, n_latents, focus_iter, diag_args.max_new_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"attention extraction failed on idx={ex.idx}: {e!r}")
            continue
        if not per_token_mass:
            continue
        n_ok += 1
        all_token_mass.extend(per_token_mass)
        if (i + 1) % 20 == 0:
            print(f"attention-checked {i + 1}/{len(attn_sample)} ({time.perf_counter() - t2:.0f}s elapsed)", flush=True)
    attn_elapsed = time.perf_counter() - t2

    if all_token_mass:
        n_tok = len(all_token_mass)
        mean_question = sum(m["question"] for m in all_token_mass) / n_tok
        mean_focus = sum(m["focus_thought"] for m in all_token_mass) / n_tok
        mean_other_thoughts = sum(m["other_thoughts"] for m in all_token_mass) / n_tok
        mean_eot = sum(m["eot"] for m in all_token_mass) / n_tok
        mean_prior_answer = sum(m["prior_answer"] for m in all_token_mass) / n_tok
        # first-generated-token only (least confounded by growing prior-answer context)
        first_tok = [m for m in all_token_mass if m["step"] == 0]
        first_focus = sum(m["focus_thought"] for m in first_tok) / len(first_tok) if first_tok else None
    else:
        n_tok = 0
        mean_question = mean_focus = mean_other_thoughts = mean_eot = mean_prior_answer = first_focus = None

    print(f"attention mass (n_examples_ok={n_ok}, n_generated_tokens={n_tok}): "
          f"question={mean_question}, focus_thought(iter {focus_iter})={mean_focus}, "
          f"other_thoughts={mean_other_thoughts}, eot={mean_eot}, prior_answer={mean_prior_answer}, "
          f"first_token_only_focus_mass={first_focus}")

    # ---- Log the run -------------------------------------------------------------------------
    metrics = {
        "final_answer_accuracy": None,
        "unparseable_rate": None,
        "compute_steps": n_latents,
        "sec_per_example": None,
        "extra": {
            "prior_run_id": PRIOR_RUN,
            "focus_iter": focus_iter,
            "n_ablate_pairs": n_ab,
            "answer_changed_by_patch": frac_patch,
            "answer_changed_by_control_any": frac_ctrl,
            "answer_changed_by_zero_ablate": frac_zero,
            "answer_changed_by_mean_ablate": frac_mean,
            "mcnemar_zero_vs_patch": {"b": b_zp, "c": c_zp, "p_exact": p_zp},
            "mcnemar_mean_vs_patch": {"b": b_mp, "c": c_mp, "p_exact": p_mp},
            "mcnemar_zero_vs_control_any": {"b": b_zc, "c": c_zc, "p_exact": p_zc},
            "readout_changed_at_all_zero": readout_z_frac,
            "readout_changed_at_all_mean": readout_m_frac,
            "ablate_elapsed_sec": ablate_elapsed,
            "n_attn_examples_requested": diag_args.n_attn_examples,
            "n_attn_examples_ok": n_ok,
            "n_attn_generated_tokens": n_tok,
            "attention_mass_question": mean_question,
            "attention_mass_focus_thought": mean_focus,
            "attention_mass_other_thoughts": mean_other_thoughts,
            "attention_mass_eot": mean_eot,
            "attention_mass_prior_answer": mean_prior_answer,
            "attention_mass_focus_thought_first_token_only": first_focus,
            "attn_elapsed_sec": attn_elapsed,
            "mean_sample_n": diag_args.mean_sample_n,
        },
    }

    record = RunRecord(
        run_id=new_run_id("codi", diag_args.slug),
        mechanism="codi",
        stage=diag_args.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=diag_args.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=n_ab, seed=None),
        metrics=metrics,
        hyperparams={"inf_latent_iterations": n_latents, "focus_iter": focus_iter,
                     "n_ablate_pairs": n_ab, "n_attn_examples": diag_args.n_attn_examples,
                     "mean_sample_n": diag_args.mean_sample_n},
        seed=None,
        hardware=f"{diag_args.hardware} / {torch.cuda.get_device_name(0)}",
        notes=f"Ablation (zero/mean at focus iter, paired against {PRIOR_RUN}'s real/control patches) "
              f"+ attention-mass check (does answer generation attend to the focus thought position at all).",
    )
    manifest_out = record.save(predictions=ablate_records)
    out_dir = manifest_out.parent
    with (out_dir / "attention_mass.jsonl").open("w") as f:
        for m in all_token_mass:
            f.write(json.dumps(m) + "\n")
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
