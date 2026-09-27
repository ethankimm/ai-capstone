#!/usr/bin/env python3
"""P1 (RESEARCH_PLAN §4): transfer-ladder levels 2-4 for Coconut -- the counterpart of
`patch_ladder_codi.py`, same donors/scoring (`ladder_common.py`). Sites are passes 0..5
(pass p = the vector spliced into the p-th <|latent|> slot); carriers default to passes 1
and 4, the sites that carry the value in the full-n E3 rerun
(`20260927-003331_coconut_minimal-pair-patch-full`).

Run inside the venv pinned to Coconut's own requirements.txt:

  cd /workspace/ai-capstone && .venv_coconut/bin/python scripts/patch_ladder_coconut.py \\
      --checkpoint_path /workspace/coconut_checkpoints/checkpoint_33 --data_dir /workspace/coconut_data \\
      --slug ladder-patch --stage full_run --hardware "RunPod RTX A5000 (secure)" --eval_n 0
"""
from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coconut_common import encode_question, extract_answer_after_delimiter, finish_and_decode, load_coconut, run_passes  # noqa: E402
from ladder_common import LEVELS, build_ladder_pairs, format_row, summarize, targets  # noqa: E402
from patch_minimal_pair_coconut import load_examples  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.eval.metrics import is_correct  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402


@dataclass
class LadderArguments:
    checkpoint_path: str = field(metadata={"help": "path to the downloaded checkpoint_33 file"})
    data_dir: str = field(metadata={"help": "dir with gsm_valid-gold-reasoning-trace_test.json"})
    model_id: str = field(default="openai-community/gpt2")
    slug: str = field(default="ladder-patch")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    checkpoint_label: str = field(default="hf:connordilgren/gpt2-gsm8k-coconut@checkpoint_33")
    num_latents: int = field(default=6)
    eval_n: int = field(default=0, metadata={"help": "0 = every gold-trace example (1194)"})
    eval_seed: int = field(default=0)
    n_recipients: int = field(default=1000)
    carriers: str = field(default="1,4")
    pair_seed: int = field(default=0)
    reps: int = field(default=200)
    max_new_tokens: int = field(default=48)
    smoke_n: int = field(default=0, metadata={"help": ">0: first n examples only, don't log"})
    device: str = field(default="cuda")


def main() -> None:
    parser = transformers.HfArgumentParser((LadderArguments,))
    (la,) = parser.parse_args_into_dataclasses()
    argv = " ".join(sys.argv)
    device = la.device

    tokenizer, base_model, embedding, special_ids = load_coconut(la.model_id, la.checkpoint_path, device)
    n_params = sum(p.numel() for p in base_model.parameters())
    eos_id = tokenizer.eos_token_id
    sites = list(range(la.num_latents))
    carriers = [int(s) for s in la.carriers.split(",")]
    conditions = {"all_slot": sites, "carriers": carriers, **{f"pass{p}": [p] for p in carriers}}

    examples = load_examples(la.data_dir, la.eval_n, la.eval_seed)
    if la.smoke_n:
        examples = examples[:la.smoke_n]
    print(f"n_examples={len(examples)}, conditions={conditions}")

    def answer_for(question: str, override_at_pass=None):
        input_ids, attn = encode_question(tokenizer, special_ids, question, la.num_latents, device)
        pass_records, inputs_embeds, kv_cache, ncr = run_passes(
            base_model, embedding, input_ids, attn, device, la.num_latents,
            override_at_pass=override_at_pass, latent_token_id=special_ids["latent"])
        raw = finish_and_decode(base_model, embedding, tokenizer, inputs_embeds, kv_cache, ncr, attn,
                                 device, la.max_new_tokens, eos_id)
        return extract_answer_after_delimiter(raw), pass_records

    t0 = time.perf_counter()
    base, correct, z_cache = {}, {}, {}
    for ex in examples:
        pred, recs = answer_for(ex.question)
        base[ex.idx], correct[ex.idx] = pred, is_correct(pred, ex.answer)
        z_cache[ex.idx] = {p: recs[p]["live_hidden"] for p in sites}
    decode_elapsed = time.perf_counter() - t0
    accuracy = sum(correct.values()) / len(correct)
    print(f"decode pass {decode_elapsed:.0f}s, base accuracy={accuracy:.3f}")

    pairs = build_ladder_pairs(examples, correct, random.Random(la.pair_seed), la.n_recipients)
    print(f"ladder recipients (L2, L3 and L4 donors available): {len(pairs)}")

    rows = []
    t1 = time.perf_counter()
    for pi, p in enumerate(pairs):
        r = p["recipient"]
        row = {"recipient_idx": r.idx, "recipient_answer": r.answer, "base": base[r.idx]}
        for level in LEVELS:
            d = p[level]
            z = z_cache[d.idx]
            preds = {name: answer_for(r.question, override_at_pass={s: z[s] for s in site_set})[0]
                     for name, site_set in conditions.items()}
            row[level] = {"donor_idx": d.idx, "donor_answer": d.answer, "targets": targets(r, d), "preds": preds}
        rows.append(row)
        if (pi + 1) % 25 == 0:
            print(f"patched {pi + 1}/{len(pairs)} recipients ({time.perf_counter() - t1:.0f}s)", flush=True)
    patch_elapsed = time.perf_counter() - t1

    rng = random.Random(la.pair_seed + 1)
    summary = {level: {name: summarize(rows, name, level, la.reps, rng) for name in conditions} for level in LEVELS}
    for level in LEVELS:
        for name in conditions:
            print(format_row(f"{level} {name}", summary[level][name]))

    if la.smoke_n:
        print("smoke run -- not logging")
        return

    record = RunRecord(
        run_id=new_run_id("coconut", la.slug),
        mechanism="coconut",
        stage=la.stage,
        model=ModelInfo(backbone=la.model_id, checkpoint=la.checkpoint_label, n_params=n_params),
        dataset=DatasetInfo(name="gsm8k-aug", split="test", n_examples=len(examples), seed=la.eval_seed),
        metrics={
            "final_answer_accuracy": accuracy,
            "compute_steps": la.num_latents,
            "sec_per_example": decode_elapsed / len(examples),
            "intervention_accuracy": summary["L2"]["all_slot"]["rates"]["donor_final"],
            "extra": {
                "design": "P1 transfer ladder L2 (same ops) / L3 (same length, other ops) / L4 (any problem), "
                          "same recipients at every level; buckets + permutation null (scripts/ladder_common.py)",
                "site_definition": "pass p = vector spliced into the p-th <|latent|> slot (pass 0 = hidden at <|start-latent|>)",
                "conditions": conditions, "n_recipients": len(rows), "summary": summary,
                "decode_elapsed_sec": decode_elapsed, "patch_elapsed_sec": patch_elapsed,
                "related_runs": ["20260927-003331_coconut_minimal-pair-patch-full"],
            },
        },
        hyperparams={"num_latents": la.num_latents, "carriers": carriers, "n_recipients_cap": la.n_recipients,
                     "pair_seed": la.pair_seed, "reps": la.reps, "max_new_tokens": la.max_new_tokens},
        seed=la.pair_seed,
        hardware=f"{la.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes="P1 transfer ladder, Coconut: donors at L2/L3/L4 transplanted into the same recipients "
              "(all-slot, carriers passes 1/4, each carrier alone).",
    )
    manifest = record.save(predictions=rows)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
