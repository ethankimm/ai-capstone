#!/usr/bin/env python3
"""Coconut Task 3: smooth path interpolation patching -- the Coconut counterpart to
`interpolation_patch_codi.py` (see that script's docstring for the full motivation:
does a full alpha=1 donor swap fail to steer the answer because the slot isn't
representational, or because the jump knocks the recipient off-manifold before any
content gets read out?).

Execution difference from the CODI version (per this project's Task 3 spec): CODI
reads P(y_A)/P(y_B) off the "final answer logits decoded right after the N-thought
loop" (one readout after all 6 loop iterations finish, mirroring the paper's own
vocabulary-projection decode). Coconut has no separate final-answer decode step --
after its `num_latents` passes finish, the very next forward step generates the first
real output token, which for this checkpoint's GSM8K format is normally part of the
"### <answer>" delimiter, not the answer digit itself. This script tracks the
**answer token logits generated immediately following the continuous pass sequence**
exactly as stated: the single logits vector from that first post-latent forward step
(mirroring `coconut_common.finish_and_decode`'s first block, without the
argmax+continuation loop), read against the first token of each example's gold-answer
string. That is a strictly harder readout than CODI's (there's no guarantee the
digit's probability mass is concentrated there rather than a few tokens later, past
"###"), which is itself informative: if this readout is uniformly near-floor for both
P(y_A) and P(y_B) regardless of alpha, that says the "answer signal" genuinely isn't
at this position for Coconut, independent of any interpolation-collapse question.

For (Recipient A, Donor B) pairs where the base model gets BOTH their answers right
and y_A != y_B, at a chosen pass index i (0..num_latents-1):

  z_i(alpha) = (1 - alpha) * z_i^A + alpha * z_i^B,  alpha in [0.0, 0.1, ..., 1.0]

spliced into RECIPIENT A's pass i via `override_at_pass` (own question, own KV cache,
only this one pass's spliced vector is replaced).

Coconut has no fixed known decodable/non-decodable split the way CODI's z0/z3 vs z2/z4
is (per `decode_patch_coconut.py`'s own finding: Coconut's passes are mostly decodable
data-dependently). Pass `--decodable_positions` / `--nondecodable_positions` (comma
lists) from that script's `most_decodable_passes` / `least_decodable_passes` output if
you want the same plot highlighting; left empty, every swept position is plotted
without a highlighted subgroup.

Classification (`interpolation_common.classify_trajectory`) and the trajectory plot
(`interpolation_common.plot_interpolation_curves`) are shared with the CODI script so
both mechanisms are read the same way.

Run inside a venv pinned to Coconut's own requirements (see `decode_patch_coconut.py`'s
docstring for why). Needs matplotlib (`pip install matplotlib`, not a base project
dependency):

  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/interpolation_patch_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/gsm-coconut/checkpoint_33 \\
      --data_dir /workspace/coconut_data \\
      --slug interpolation-pilot --stage pilot --hardware "RunPod RTX A5000 (secure)" \\
      --num_latents 6 --pool_n 400 --n_pairs 50 --positions 0,1,2,3,4,5
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coconut_common import (  # noqa: E402
    load_coconut, encode_question, run_passes, finish_and_decode, extract_answer_after_delimiter,
)
from decode_patch_coconut import load_examples, CoconutArguments  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.mechanisms.coconut import name as mechanism_name  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402
from interpolation_common import (  # noqa: E402
    ALPHAS, classify_trajectory, classification_summary, plot_interpolation_curves,
)


@dataclass
class InterpArguments(CoconutArguments):
    slug: str = field(default="interpolation-pilot")
    pool_n: int = field(default=400, metadata={"help": "candidate pool size before filtering to base-correct examples"})
    pool_seed: int = field(default=0)
    n_pairs: int = field(default=50)
    pair_seed: int = field(default=0)
    positions: str = field(default="0,1,2,3,4,5")
    decodable_positions: str = field(default="", metadata={"help": "comma list from decode_patch_coconut.py's most_decodable_passes, for plot highlighting only"})
    nondecodable_positions: str = field(default="", metadata={"help": "comma list from decode_patch_coconut.py's least_decodable_passes, for plot highlighting only"})


def _parse_positions(s: str) -> tuple[int, ...]:
    return tuple(int(p) for p in s.split(",") if p.strip() != "")


def sample_pairs(examples: list, n: int, rng: random.Random) -> list[tuple]:
    pool = list(examples)
    pairs, attempts, max_attempts = [], 0, n * 20
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        r, d = rng.sample(pool, 2)
        if r.answer.strip() != d.answer.strip():
            pairs.append((r, d))
    return pairs


@torch.no_grad()
def get_zs_and_pred(tokenizer, base_model, embedding, special_ids, ex, device, num_latents, max_new_tokens,
                     eos_id, cache: dict) -> dict:
    if ex.idx in cache:
        return cache[ex.idx]
    input_ids, attn = encode_question(tokenizer, special_ids, ex.question, num_latents, device)
    pass_records, inputs_embeds, kv_cache, ncr = run_passes(
        base_model, embedding, input_ids, attn, device, num_latents, latent_token_id=special_ids["latent"])
    zs = {p["pass"]: p["live_hidden"] for p in pass_records}
    raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, ncr, attn, device,
                             max_new_tokens, eos_id)
    pred = extract_answer_after_delimiter(raw)
    cache[ex.idx] = {"zs": zs, "pred": pred, "correct": is_correct(pred, ex.answer),
                      "input_ids": input_ids, "attn": attn}
    return cache[ex.idx]


@torch.no_grad()
def readout_at_alpha(base_model, embedding, tokenizer, special_ids, ex, device, num_latents,
                      pos: int, z_alpha) -> torch.Tensor:
    """Recomputes the recipient's pass sequence with z_i(alpha) spliced into pass `pos`,
    then does the single forward step that immediately follows the latent sequence --
    `coconut_common.finish_and_decode`'s first block, logits only, no generation."""
    input_ids, attn = encode_question(tokenizer, special_ids, ex.question, num_latents, device)
    pass_records, inputs_embeds, kv_cache, ncr = run_passes(
        base_model, embedding, input_ids, attn, device, num_latents,
        override_at_pass={pos: z_alpha}, latent_token_id=special_ids["latent"])
    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
    past_key_values = (
        [(k[:, :, :ncr[0], :], v[:, :, :ncr[0], :]) for k, v in kv_cache] if kv_cache else None
    )
    outputs = base_model(
        inputs_embeds=inputs_embeds[:, ncr[0]:ncr[1], :],
        attention_mask=attn[:, :ncr[1]],
        position_ids=position_ids[:, ncr[0]:ncr[1]],
        past_key_values=past_key_values,
    )
    return outputs.logits[0, -1, :].float().cpu()


def answer_token_id(tokenizer, answer: str) -> int:
    ids = tokenizer(answer.strip(), add_special_tokens=False)["input_ids"]
    return ids[0]


def prob_and_entropy(logits: torch.Tensor, id_a: int, id_b: int) -> tuple[float, float, float]:
    probs = F.softmax(logits, dim=-1)
    p_a, p_b = probs[id_a].item(), probs[id_b].item()
    nz = probs[probs > 0]
    h = -(nz * nz.log()).sum().item()
    return p_a, p_b, h


def main() -> None:
    parser = transformers.HfArgumentParser((InterpArguments,))
    (ia,) = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = ia.device

    tokenizer, base_model, embedding, special_ids = load_coconut(ia.model_id, ia.checkpoint_path, device)
    n_params = sum(p.numel() for p in base_model.parameters())
    eos_id = tokenizer.eos_token_id

    positions = _parse_positions(ia.positions)
    for p in positions:
        assert 0 <= p < ia.num_latents, f"position {p} out of range for num_latents={ia.num_latents}"
    decodable_positions = _parse_positions(ia.decodable_positions)
    nondecodable_positions = _parse_positions(ia.nondecodable_positions)

    pool = load_examples(ia.data_dir, full_test=False, eval_n=ia.pool_n, eval_seed=ia.pool_seed)
    print(f"pool n={len(pool)}, evaluating base correctness + caching pass live_hidden 0..{ia.num_latents - 1}")

    cache: dict = {}
    t0 = time.perf_counter()
    for ex in pool:
        get_zs_and_pred(tokenizer, base_model, embedding, special_ids, ex, device, ia.num_latents,
                         ia.max_new_tokens, eos_id, cache)
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
        sum_pa = [0.0] * len(ALPHAS)
        sum_pb = [0.0] * len(ALPHAS)
        sum_h = [0.0] * len(ALPHAS)
        for recipient, donor in pairs:
            z_a = cache[recipient.idx]["zs"][pos]
            z_b = cache[donor.idx]["zs"][pos]
            id_a = answer_token_id(tokenizer, recipient.answer)
            id_b = answer_token_id(tokenizer, donor.answer)
            pa_seq, pb_seq, h_seq = [], [], []
            for alpha in ALPHAS:
                z_alpha = (1.0 - alpha) * z_a + alpha * z_b
                logits = readout_at_alpha(base_model, embedding, tokenizer, special_ids, recipient, device,
                                           ia.num_latents, pos, z_alpha)
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
    nondec_in_scope = [p for p in positions if p in nondecodable_positions]
    dec_in_scope = [p for p in positions if p in decodable_positions]
    nondec_summary = classification_summary([l for p in nondec_in_scope for l in labels_by_position[p]]) if nondec_in_scope else None
    dec_summary = classification_summary([l for p in dec_in_scope for l in labels_by_position[p]]) if dec_in_scope else None
    print(f"OVERALL classification: {overall_summary['pct']}")
    if nondec_summary:
        print(f"non-decodable positions {nondec_in_scope}: {nondec_summary['pct']}")
    if dec_summary:
        print(f"decodable positions {dec_in_scope}: {dec_summary['pct']}")

    # ---- log the run (also writes interpolation_curves.png + interpolation_pairs.jsonl) ---
    metrics = {
        "compute_steps": ia.num_latents,
        "intervention_accuracy": nondec_summary["pct"]["smooth_transition"] if nondec_summary else None,
        "extra": {
            "positions": positions,
            "decodable_positions_in_scope": dec_in_scope,
            "nondecodable_positions_in_scope": nondec_in_scope,
            "n_pairs": len(pairs),
            "alphas": list(ALPHAS),
            "overall_classification": overall_summary,
            "per_position_classification": per_position_summary,
            "nondecodable_classification": nondec_summary,
            "decodable_classification": dec_summary,
            "n_trainable_params": n_params,
            "codi_analogue": "interpolation_patch_codi.py",
        },
    }

    record = RunRecord(
        run_id=new_run_id("coconut", ia.slug),
        mechanism=mechanism_name,
        stage=ia.stage,
        model=ModelInfo(backbone=ia.model_id, checkpoint=ia.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(pool), seed=ia.pool_seed),
        metrics=metrics,
        hyperparams={"num_latents": ia.num_latents, "n_pairs": ia.n_pairs, "positions": positions,
                     "n_alphas": len(ALPHAS)},
        seed=ia.pair_seed,
        hardware=f"{ia.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes="Smooth path interpolation patching (Task 3), Coconut counterpart to "
              "interpolation_patch_codi.py: 11-point alpha sweep between recipient and "
              "donor pass vectors, single-pass splice, reading the first post-latent "
              "logits at each alpha. Classifies each pair/position trajectory as "
              "smooth_transition, off_manifold_collapse, step_function_invariance, or ambiguous.",
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
            decodable_positions=decodable_positions, nondecodable_positions=nondecodable_positions,
            position_label=lambda p: f"pass{p}", title="Coconut: smooth path interpolation patching",
        )
        print(f"wrote {out_dir / 'interpolation_curves.png'}")
    except ImportError as e:
        print(f"matplotlib not available ({e}); skipped interpolation_curves.png -- "
              f"curves data is in interpolation_pairs.jsonl, plot offline with interpolation_common.plot_interpolation_curves")
    print(f"run_id={record.run_id} -> fill in {out_dir / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
