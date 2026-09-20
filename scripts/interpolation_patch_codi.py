#!/usr/bin/env python3
"""CODI Task 3: smooth path interpolation patching. Follow-up to
`interchange_patch_placeholder_codi.py` (full donor-swap at z0/z3: content-bearing
donor doesn't steer the answer, unlike the explicit-CoT positive control). That result
is consistent with two very different mechanisms: (a) the placeholder slots really
don't carry portable content, or (b) a full alpha=1 donor swap knocks the recipient's
residual stream off-manifold before the donor's content can be read out at all, and the
null is an artifact of only ever testing the extreme endpoint. This script walks the
*continuous* path between them to tell those apart.

For (Recipient A, Donor B) pairs where the base model gets BOTH their answers right
and y_A != y_B, at a chosen latent position z_i (i in 0..5 -- z_i is the INPUT fed to
loop iteration i+1, same indexing as `interchange_patch_placeholder_codi.py`'s
NONDECODABLE_ITERS=(1,4)==z0,z3 / DECODABLE_ITERS=(3,5)==z2,z4):

  z_i(alpha) = (1 - alpha) * z_i^A + alpha * z_i^B,  alpha in [0.0, 0.1, ..., 1.0]

patch it into iteration i+1 of RECIPIENT A's forward pass (recipient's own question,
own KV cache -- only this one slot is replaced), run the remaining iterations, and read
the vocabulary distribution at the SAME single logits readout CODI's own decode method
uses: `outputs.logits[:, -1, :]` computed on the eot-embedding forward pass immediately
after the 6-iteration loop, before any autoregressive generation (this is `decode_answer`'s
first step, and per the paper's Sec 5.1 "projecting into vocabulary space via the word
embeddings" -- exactly the method `decode_patch_codi.py` already uses for the logit
lens). P(y_A)/P(y_B) = softmax probability on the first token of each example's
tokenized gold-answer string at that readout; H = Shannon entropy of the same
distribution (masked to the real GPT-2 vocab, i.e. excluding CODI's added pad/bot/eot
ids, same masking `topk_token_strings` uses for decode targets).

Classification (`interpolation_common.classify_trajectory`) and the trajectory plot
(`interpolation_common.plot_interpolation_curves`) are shared with the Coconut version
of this script so the two mechanisms are read the same way.

Reuses `run_thoughts` / `decode_answer` / `wilson_ci` from `decode_patch_codi.py` and
`sample_pairs` from `interchange_patch_placeholder_codi.py` (donor/recipient selection
by different gold final answers).

Run inside the CODI venv, from the CODI checkout (same convention as the other CODI
scripts). Needs matplotlib (`pip install matplotlib`, not a base project dependency):

  cd /workspace/codi && .venv/bin/python /workspace/ai-capstone/scripts/interpolation_patch_codi.py \\
      --ckpt_dir /workspace/codi_released --checkpoint_label "hf:zen-E/CODI-gpt2@fd641b3" \\
      --slug interpolation-pilot --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --model_name_or_path gpt2 --seed 11 --model_max_length 512 --bf16 \\
      --lora_r 128 --lora_alpha 32 --lora_init --greedy True \\
      --num_latent 6 --use_prj True --prj_dim 768 --prj_no_ln False --prj_dropout 0.0 \\
      --inf_latent_iterations 6 --inf_num_iterations 1 --remove_eos True --use_lora True \\
      --pool_n 400 --n_pairs 50 --positions 0,1,2,3,4,5
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
import torch.nn.functional as F
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())  # the CODI checkout: `src.model`

from src.model import DataArguments, ModelArguments, TrainingArguments  # noqa: E402
from eval_codi import build_model  # noqa: E402
from decode_patch_codi import run_thoughts, decode_answer, wilson_ci, ORIG_VOCAB  # noqa: E402
from interchange_patch_placeholder_codi import sample_pairs  # noqa: E402

from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.eval.metrics import extract_final_number, is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402
from interpolation_common import (  # noqa: E402
    ALPHAS, classify_trajectory, classification_summary, plot_interpolation_curves,
)

NONDECODABLE_POSITIONS = (0, 3)  # z0, z3 -- load-bearing under ablation, null under single donor-swap
DECODABLE_POSITIONS = (2, 4)     # z2, z4 -- best logit-lens decode positions


@dataclass
class InterpArguments:
    slug: str = field(default="interpolation-pilot")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: Optional[str] = field(default=None)
    pool_n: int = field(default=400, metadata={"help": "candidate pool size before filtering to base-correct examples"})
    pool_seed: int = field(default=0)
    n_pairs: int = field(default=50, metadata={"help": "held-out (recipient, donor) pairs, both base-correct, different gold answers"})
    pair_seed: int = field(default=0)
    positions: str = field(default="0,1,2,3,4,5", metadata={"help": "comma-separated z_i positions to sweep (0..num_latent-1)"})
    max_new_tokens: int = field(default=64)


@torch.no_grad()
def get_zs_and_pred(model, tokenizer, ex, device, n_latents, cache: dict, max_new_tokens: int) -> dict:
    if ex.idx in cache:
        return cache[ex.idx]
    q = ex.question.strip().replace("  ", " ")
    _, _, z0 = run_thoughts(model, tokenizer, q, device, 0)  # pre-loop latent, cheap (encode only)
    pkv, thoughts, latent = run_thoughts(model, tokenizer, q, device, n_latents)
    zs = {0: z0}
    for i in range(1, n_latents):
        zs[i] = thoughts[i - 1]["post"]
    raw = decode_answer(model, tokenizer, pkv, latent, device, max_new_tokens)
    pred = extract_final_number(raw)
    cache[ex.idx] = {"zs": zs, "pred": pred, "correct": is_correct(pred, ex.answer)}
    return cache[ex.idx]


@torch.no_grad()
def readout_at_alpha(model, tokenizer, recipient_question: str, device, n_latents: int,
                      iter_idx: int, z_alpha) -> tuple[torch.Tensor, str]:
    """Runs the recipient's full loop with z_i(alpha) patched into iteration `iter_idx`,
    then does the SAME single forward pass `decode_answer` starts with (eot embedding,
    before any generated token) and returns its raw logits -- the "final answer logits
    decoded right after the N-thought loop"."""
    pkv, thoughts, latent = run_thoughts(model, tokenizer, recipient_question, device, n_latents,
                                          override_input_at={iter_idx: z_alpha})
    eot_emb = model.get_embd(model.codi, model.model_name)(
        torch.tensor([model.eot_id], dtype=torch.long, device=device)
    ).unsqueeze(0)
    out = model.codi(inputs_embeds=eot_emb, output_hidden_states=False, attention_mask=None,
                      use_cache=True, past_key_values=pkv)
    logits = out.logits[0, -1, :].float().cpu()
    return logits


def answer_token_id(tokenizer, answer: str) -> int:
    ids = tokenizer(answer.strip(), add_special_tokens=False)["input_ids"]
    return ids[0]


def prob_and_entropy(logits: torch.Tensor, id_a: int, id_b: int) -> tuple[float, float, float]:
    masked = logits.clone()
    masked[ORIG_VOCAB:] = -float("inf")  # never let CODI's added pad/bot/eot ids carry probability mass
    probs = F.softmax(masked, dim=-1)
    p_a, p_b = probs[id_a].item(), probs[id_b].item()
    nz = probs[probs > 0]
    h = -(nz * nz.log()).sum().item()
    return p_a, p_b, h


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, InterpArguments))
    model_args, data_args, training_args, ia = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = "cuda"

    model, tokenizer, load_result = build_model(model_args, training_args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {model_args.ckpt_dir}: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")

    n_latents = training_args.inf_latent_iterations
    positions = [int(p) for p in ia.positions.split(",")]
    for p in positions:
        assert 0 <= p < n_latents, f"position {p} out of range for n_latents={n_latents}"

    pool = load_gsm8k_aug(split="test", n=ia.pool_n, seed=ia.pool_seed)
    print(f"pool n={len(pool)}, evaluating base correctness + caching z0..z{n_latents - 1}")

    cache: dict = {}
    t0 = time.perf_counter()
    for ex in pool:
        get_zs_and_pred(model, tokenizer, ex, device, n_latents, cache, ia.max_new_tokens)
    correct_pool = [ex for ex in pool if cache[ex.idx]["correct"]]
    print(f"base-correct pool: {len(correct_pool)}/{len(pool)} ({time.perf_counter() - t0:.0f}s)")

    pairs = sample_pairs(correct_pool, ia.n_pairs, random.Random(ia.pair_seed))
    print(f"selected {len(pairs)} (recipient, donor) pairs with different gold answers")

    # ---- alpha sweep -------------------------------------------------------------------------
    all_records = []
    curves: dict[int, dict[str, list[float]]] = {}
    labels_by_position: dict[int, list[str]] = {p: [] for p in positions}
    t1 = time.perf_counter()
    n_done = 0
    total = len(pairs) * len(positions)
    for pos in positions:
        iter_idx = pos + 1
        sum_pa = [0.0] * len(ALPHAS)
        sum_pb = [0.0] * len(ALPHAS)
        sum_h = [0.0] * len(ALPHAS)
        for recipient, donor in pairs:
            q_r = recipient.question.strip().replace("  ", " ")
            z_a = cache[recipient.idx]["zs"][pos]
            z_b = cache[donor.idx]["zs"][pos]
            id_a = answer_token_id(tokenizer, recipient.answer)
            id_b = answer_token_id(tokenizer, donor.answer)
            pa_seq, pb_seq, h_seq = [], [], []
            for alpha in ALPHAS:
                z_alpha = (1.0 - alpha) * z_a + alpha * z_b
                logits = readout_at_alpha(model, tokenizer, q_r, device, n_latents, iter_idx, z_alpha)
                p_a, p_b, h = prob_and_entropy(logits, id_a, id_b)
                pa_seq.append(p_a)
                pb_seq.append(p_b)
                h_seq.append(h)
            for i in range(len(ALPHAS)):
                sum_pa[i] += pa_seq[i]
                sum_pb[i] += pb_seq[i]
                sum_h[i] += h_seq[i]
            label = classify_trajectory(list(ALPHAS), pa_seq, pb_seq, h_seq)
            labels_by_position[pos].append(label)
            all_records.append({
                "position": pos, "recipient_idx": recipient.idx, "donor_idx": donor.idx,
                "recipient_gold": recipient.answer, "donor_gold": donor.answer,
                "alphas": list(ALPHAS), "p_yA": pa_seq, "p_yB": pb_seq, "entropy": h_seq,
                "classification": label,
            })
            n_done += 1
            if n_done % 25 == 0:
                print(f"{n_done}/{total} (pos={pos}) done ({time.perf_counter() - t1:.0f}s)", flush=True)
        n_p = len(pairs)
        curves[pos] = {
            "mean_pa": [v / n_p for v in sum_pa],
            "mean_pb": [v / n_p for v in sum_pb],
            "mean_h": [v / n_p for v in sum_h],
        }

    # ---- classification summary -----------------------------------------------------------
    all_labels = [r["classification"] for r in all_records]
    overall_summary = classification_summary(all_labels)
    per_position_summary = {pos: classification_summary(labels_by_position[pos]) for pos in positions}
    nondec_in_scope = [p for p in positions if p in NONDECODABLE_POSITIONS]
    dec_in_scope = [p for p in positions if p in DECODABLE_POSITIONS]
    nondec_summary = classification_summary([l for p in nondec_in_scope for l in labels_by_position[p]])
    dec_summary = classification_summary([l for p in dec_in_scope for l in labels_by_position[p]])
    print(f"OVERALL classification: {overall_summary['pct']}")
    print(f"non-decodable positions {nondec_in_scope}: {nondec_summary['pct']}")
    print(f"decodable positions {dec_in_scope}: {dec_summary['pct']}")

    # ---- log the run (also writes interpolation_curves.png + interpolation_pairs.jsonl) ---
    metrics = {
        "compute_steps": n_latents,
        "intervention_accuracy": nondec_summary["pct"]["smooth_transition"] if nondec_in_scope else None,
        "extra": {
            "positions": positions,
            "nondecodable_positions_in_scope": nondec_in_scope,
            "decodable_positions_in_scope": dec_in_scope,
            "n_pairs": len(pairs),
            "alphas": list(ALPHAS),
            "overall_classification": overall_summary,
            "per_position_classification": per_position_summary,
            "nondecodable_classification": nondec_summary,
            "decodable_classification": dec_summary,
            "n_trainable_params": None,
            "related_runs": [
                "20260919-184323_codi_decode-patch-full-eval",
                "20260919-192228_codi_early-termination-ablate-all",
                "20260920-031925_codi_interchange-placeholder-pilot",
            ],
        },
    }

    record = RunRecord(
        run_id=new_run_id("codi", ia.slug),
        mechanism="codi",
        stage=ia.stage,
        model=ModelInfo(backbone="gpt2", checkpoint=ia.checkpoint_label or model_args.ckpt_dir, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(pool), seed=ia.pool_seed),
        metrics=metrics,
        hyperparams={
            "inf_latent_iterations": n_latents, "num_latent": training_args.num_latent,
            "n_pairs": ia.n_pairs, "positions": positions, "n_alphas": len(ALPHAS),
        },
        seed=ia.pair_seed,
        hardware=f"{ia.hardware} / {torch.cuda.get_device_name(0)}",
        notes="Smooth path interpolation patching (Task 3): 11-point alpha sweep between "
              "recipient and donor z_i, single-slot patch, reading the post-loop logits at "
              "each alpha. Classifies each pair/position trajectory as smooth_transition, "
              "off_manifold_collapse, step_function_invariance, or ambiguous.",
    )
    manifest = record.save(predictions=all_records)
    out_dir = manifest.parent
    with (out_dir / "interpolation_pairs.jsonl").open("w") as f:
        for r in all_records:
            f.write(json.dumps(r, default=str) + "\n")
    (out_dir / "eval_command.txt").write_text(argv + "\n")
    try:
        plot_interpolation_curves(
            curves, ALPHAS, str(out_dir / "interpolation_curves.png"),
            decodable_positions=DECODABLE_POSITIONS, nondecodable_positions=NONDECODABLE_POSITIONS,
            position_label=lambda p: f"z{p}", title="CODI: smooth path interpolation patching",
        )
        print(f"wrote {out_dir / 'interpolation_curves.png'}")
    except ImportError as e:
        print(f"matplotlib not available ({e}); skipped interpolation_curves.png -- "
              f"curves data is in interpolation_pairs.jsonl, plot offline with interpolation_common.plot_interpolation_curves")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
