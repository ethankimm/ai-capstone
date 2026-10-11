#!/usr/bin/env python3
"""Zhang et al. 2025 (*Do Latent Tokens Think?*, arXiv 2512.21711, Table 3) swap
replication, next to our E3 twin swap on the SAME recipients, for Coconut and the
matching explicit-CoT checkpoint from the Dilgren & Wiegreffe grid (`PILOT_ZHANG_SWAP.md`).

Zhang's swap: a fixed random permutation over the eval set; each example's latent (or
CoT) tokens are replaced by its permutation partner's. They report accuracy before/after
and the inconsistency rate (IR = answer changed). On MMLU / Llama-3-8B: CoT 62.8 -> 43.4
(IR 52.8%), Coconut 60.9 -> 61.0 (IR 17.9%) -> "latents are placeholders".

Conditions, per example:
  random_swap   donor = permutation partner (a derangement, fixed by --perm_seed).
                Coconut: all 6 latent passes replaced by the donor's (aligned, pass p -> pass p).
                CoT: the recipient's question followed by the donor's generated reasoning text
                (everything before "###"), then "###" and greedy answer.
  twin_swap     (gsm8k only) donor = one-number-edited twin of the same question
                (`latentreasoning.data.minimal_pairs`, as E3), restricted to pairs where
                recipient AND twin are answered correctly. Same splice as random_swap.

Metrics per condition: accuracy after, IR, and steer = P(swapped answer == donor's own
answer | donor's answer != recipient's answer). Reported on all examples (Zhang's
framing) and on base-correct recipients (ours).

  .venv/bin/python scripts/swap_random_coconut.py --method coconut --dataset gsm8k \\
      --checkpoint_path /workspace/grid/gpt2-gsm8k-coconut/checkpoint_33 \\
      --data_dir /workspace/are-lrms-easily-interpretable/data \\
      --slug zhang-swap-gsm8k --stage pilot --hardware "RunPod RTX A5000 (secure)"
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coconut_common import (  # noqa: E402
    END_LATENT_TOKEN, LATENT_TOKEN, START_LATENT_TOKEN, encode_question, extract_answer_after_delimiter,
    finish_and_decode, mcnemar_exact_p, num_match, run_passes, wilson_ci,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from latentreasoning.data.gsm8k_aug import Example  # noqa: E402
from latentreasoning.data.minimal_pairs import generate_minimal_pair  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps, qualifying_steps  # noqa: E402
from latentreasoning.runlog.manifest import DatasetInfo, ModelInfo, RunRecord, new_run_id  # noqa: E402

DELTAS = (1, -1, 2, -2, 3, -3)
DATA_FILES = {"gsm8k": "gsm_valid-gold-reasoning-trace_test.json",
              "prontoqa": "prontoqa_test.json", "prosqa": "prosqa_test.json"}


@dataclass
class SwapArguments:
    checkpoint_path: str = field(metadata={"help": "grid checkpoint file (checkpoint_NN)"})
    data_dir: str = field(metadata={"help": "are-lrms-easily-interpretable/data"})
    method: str = field(default="coconut", metadata={"help": "coconut | cot"})
    dataset: str = field(default="gsm8k", metadata={"help": "gsm8k | prontoqa | prosqa"})
    model_id: str = field(default="openai-community/gpt2")
    checkpoint_label: Optional[str] = field(default=None)
    slug: str = field(default="zhang-swap")
    stage: str = field(default="pilot")
    hardware: str = field(default="RunPod GPU")
    num_latents: int = field(default=6)
    eval_n: int = field(default=0, metadata={"help": "0 = full test file"})
    eval_seed: int = field(default=0)
    perm_seed: int = field(default=0)
    n_twin: int = field(default=400, metadata={"help": "cap on responsive twin pairs (gsm8k)"})
    max_new_tokens: int = field(default=0, metadata={"help": "0 = 48 for coconut, 256 for cot"})
    smoke_n: int = field(default=0, metadata={"help": ">0: first n examples only, don't log"})
    device: str = field(default="cuda")


def load_model(model_id: str, checkpoint_path: str, device: str):
    """Grid checkpoints: Coconut saves `base_causallm.*` with the three latent tokens added
    to the vocab; CoT/no-CoT save the bare LM with the base vocab (latent ids unused)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    for t in (START_LATENT_TOKEN, END_LATENT_TOKEN, LATENT_TOKEN):
        tokenizer.add_tokens(t)
    special_ids = {"latent": tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
                   "start_latent": tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN),
                   "end_latent": tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN)}
    state = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("base_causallm.") for k in state):
        state = {k.removeprefix("base_causallm."): v for k, v in state.items() if k.startswith("base_causallm.")}
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(state["lm_head.weight"].shape[0])  # CoT/no-CoT grid ckpts keep the base vocab
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith("attn.bias") and not k.endswith("masked_bias")]
    print(f"loaded {checkpoint_path}: missing={len(real_missing)} unexpected={len(unexpected)}")
    assert not real_missing, f"checkpoint missing weights: {real_missing[:5]}"
    model = model.to(device).eval()
    return tokenizer, model, model.get_input_embeddings(), special_ids


def load_examples(data_dir: str, dataset: str, eval_n: int, eval_seed: int) -> list[Example]:
    rows = json.loads((Path(data_dir) / DATA_FILES[dataset]).read_text())
    examples = [Example(question=r["question"], rationale=" ".join(r["steps"]),
                        answer=r["answer"].strip(), idx=i) for i, r in enumerate(rows)]
    random.Random(eval_seed).shuffle(examples)
    return examples[:eval_n] if eval_n else examples


def same_answer(a: Optional[str], b: Optional[str], dataset: str) -> bool:
    if a is None or b is None:
        return False
    if dataset == "gsm8k":
        return num_match([a], b)
    return a.strip().rstrip(".").lower() == b.strip().rstrip(".").lower()


def derangement(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(i != p for i, p in enumerate(perm)):
            return perm


def main() -> None:
    (sa,) = transformers.HfArgumentParser((SwapArguments,)).parse_args_into_dataclasses()
    assert sa.method in ("coconut", "cot") and sa.dataset in DATA_FILES
    argv = " ".join(sys.argv)
    device = sa.device
    max_new = sa.max_new_tokens or (48 if sa.method == "coconut" else 256)
    tokenizer, model, embedding, special_ids = load_model(sa.model_id, sa.checkpoint_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    eos_id = tokenizer.eos_token_id

    examples = load_examples(sa.data_dir, sa.dataset, sa.eval_n, sa.eval_seed)
    if sa.smoke_n:
        examples = examples[:sa.smoke_n]
    print(f"method={sa.method} dataset={sa.dataset} n_examples={len(examples)}")

    # ---- method-specific "run" (returns answer + the donor payload) and "run with donor payload"
    @torch.no_grad()
    def greedy(prompt_ids: list[int]) -> str:
        ids = torch.tensor([prompt_ids], device=device)
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False,
                             eos_token_id=eos_id, pad_token_id=eos_id)
        return tokenizer.decode(out[0, len(prompt_ids):], skip_special_tokens=True)

    def q_ids(question: str) -> list[int]:
        return tokenizer.encode(question if question.endswith("\n") else question + "\n", add_special_tokens=True)

    if sa.method == "coconut":
        def run(question: str, payload=None):
            input_ids, attn = encode_question(tokenizer, special_ids, question, sa.num_latents, device)
            override = {p: payload[p] for p in range(sa.num_latents)} if payload is not None else None
            recs, emb, kv, ncr = run_passes(model, embedding, input_ids, attn, device, sa.num_latents,
                                            override_at_pass=override, latent_token_id=special_ids["latent"])
            raw = finish_and_decode(model, embedding, tokenizer, emb, kv, ncr, attn, device, max_new, eos_id)
            return extract_answer_after_delimiter(raw), [r["live_hidden"] for r in recs], raw
    else:
        def run(question: str, payload=None):
            if payload is None:
                raw = greedy(q_ids(question))
                reasoning = raw.split("###")[0] if "###" in raw else None
                return extract_answer_after_delimiter(raw), reasoning, raw
            tail = greedy(q_ids(question) + tokenizer.encode(payload + "###", add_special_tokens=False))
            return tail.strip().replace(",", "") or None, None, "###" + tail

    # ---- base pass ------------------------------------------------------------------------------
    t0 = time.perf_counter()
    base: dict[int, dict] = {}
    for i, ex in enumerate(examples):
        pred, payload, raw = run(ex.question)
        base[ex.idx] = {"pred": pred, "correct": same_answer(pred, ex.answer, sa.dataset), "payload": payload, "raw": raw}
        if (i + 1) % 200 == 0:
            print(f"decoded {i + 1}/{len(examples)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    decode_elapsed = time.perf_counter() - t0
    acc_before = sum(b["correct"] for b in base.values()) / len(base)
    print(f"base accuracy={acc_before:.3f} ({decode_elapsed:.0f}s)")

    def cond(ex: Example, donor_pred: Optional[str], payload) -> dict:
        if payload is None:  # CoT donor with no "###" -> nothing to splice
            return {"pred": None, "skipped": True}
        pred, _, raw = run(ex.question, payload)
        rb = base[ex.idx]["pred"]
        return {"pred": pred, "raw": raw[:200], "skipped": False,
                "correct": same_answer(pred, ex.answer, sa.dataset),
                "answer_changed": not same_answer(pred, rb, sa.dataset),
                "donor_differs": not same_answer(donor_pred, rb, sa.dataset),
                "matches_donor": same_answer(pred, donor_pred, sa.dataset) and not same_answer(pred, rb, sa.dataset)}

    # ---- random swap (Zhang) --------------------------------------------------------------------
    t1 = time.perf_counter()
    perm = derangement(len(examples), sa.perm_seed)
    records = []
    for i, ex in enumerate(examples):
        donor = examples[perm[i]]
        r = cond(ex, base[donor.idx]["pred"], base[donor.idx]["payload"])
        records.append({"recipient_idx": ex.idx, "donor_idx": donor.idx, "gold": ex.answer,
                        "base_pred": base[ex.idx]["pred"], "base_correct": base[ex.idx]["correct"],
                        "donor_pred": base[donor.idx]["pred"], "random_swap": r})
        if (i + 1) % 200 == 0:
            print(f"random-swapped {i + 1}/{len(examples)} ({time.perf_counter() - t1:.0f}s)", flush=True)

    # ---- twin swap (gsm8k only, E3 design) ------------------------------------------------------
    twin_by_recipient: dict[int, dict] = {}
    cfr = None
    if sa.dataset == "gsm8k":
        rng = random.Random(sa.perm_seed)
        cands = []
        for ex in examples:
            if not base[ex.idx]["correct"]:
                continue
            for k in qualifying_steps(parse_steps(ex.rationale)):
                mp = next((m for d in DELTAS if (m := generate_minimal_pair(ex, k, d, rng=rng)) is not None), None)
                if mp is not None:
                    cands.append(mp)
                    break  # one twin per recipient so random vs twin is paired 1:1
        n_decoded = 0
        for mp in cands:
            if len(twin_by_recipient) >= sa.n_twin:
                break
            tpred, tpayload, _ = run(mp.twin.question)
            n_decoded += 1
            if not same_answer(tpred, mp.twin.answer, sa.dataset):
                continue
            r = cond(mp.original, tpred, tpayload)
            r.update({"twin_answer": mp.twin.answer, "step": mp.step, "delta": mp.delta})
            twin_by_recipient[mp.original.idx] = r
        cfr = {"n_candidates": len(cands), "n_twins_decoded": n_decoded, "n_responsive": len(twin_by_recipient),
               "twin_correct_given_original_correct": len(twin_by_recipient) / n_decoded if n_decoded else None}
        for rec in records:
            if rec["recipient_idx"] in twin_by_recipient:
                rec["twin_swap"] = twin_by_recipient[rec["recipient_idx"]]
    swap_elapsed = time.perf_counter() - t1

    # ---- aggregate ------------------------------------------------------------------------------
    def agg(rows: list[dict], key: str) -> dict:
        rs = [r[key] for r in rows if key in r and not r[key]["skipped"]]
        n = len(rs)
        if not n:
            return {"n": 0}
        ch = sum(r["answer_changed"] for r in rs)
        cor = sum(r["correct"] for r in rs)
        dd = [r for r in rs if r["donor_differs"]]
        st = sum(r["matches_donor"] for r in dd)
        return {"n": n, "n_skipped": sum(1 for r in rows if key in r and r[key]["skipped"]),
                "accuracy_after": cor / n, "accuracy_after_ci": list(wilson_ci(cor, n)),
                "ir": ch / n, "ir_ci": list(wilson_ci(ch, n)),
                "n_donor_differs": len(dd), "steer": st / len(dd) if dd else None,
                "steer_ci": list(wilson_ci(st, len(dd)))}

    correct_rows = [r for r in records if r["base_correct"]]
    paired = [r for r in correct_rows if "twin_swap" in r and not r["random_swap"]["skipped"]]
    b = sum(r["twin_swap"]["answer_changed"] and not r["random_swap"]["answer_changed"] for r in paired)
    c = sum(r["random_swap"]["answer_changed"] and not r["twin_swap"]["answer_changed"] for r in paired)
    summary = {
        "accuracy_before": acc_before,
        "random_swap_all": agg(records, "random_swap"),
        "random_swap_base_correct": agg(correct_rows, "random_swap"),
        "twin_swap_base_correct": agg(correct_rows, "twin_swap"),
        "random_swap_on_twin_recipients": agg(paired, "random_swap"),
        "paired_ir_twin_vs_random": {"n": len(paired), "b_twin_only": b, "c_random_only": c,
                                     "mcnemar_p": mcnemar_exact_p(b, c)},
        "cfr": cfr,
    }
    for k, v in summary.items():
        print(f"{k}: {json.dumps(v)}")

    if sa.smoke_n:
        print("smoke run -- not logging")
        return

    mechanism = "coconut" if sa.method == "coconut" else "explicit_cot"
    label = sa.checkpoint_label or sa.checkpoint_path
    record = RunRecord(
        run_id=new_run_id(mechanism, sa.slug),
        mechanism=mechanism,
        stage=sa.stage,
        model=ModelInfo(backbone=sa.model_id, checkpoint=label, n_params=n_params),
        dataset=DatasetInfo(name=sa.dataset if sa.dataset != "gsm8k" else "gsm8k-aug", split="test",
                            n_examples=len(examples), seed=sa.eval_seed),
        metrics={
            "final_answer_accuracy": acc_before,
            "compute_steps": sa.num_latents if sa.method == "coconut" else None,
            "sec_per_example": decode_elapsed / len(examples),
            "intervention_accuracy": summary["random_swap_all"].get("accuracy_after"),
            "extra": {"design": "Zhang et al. 2025 random-permutation full-chain swap (+ E3 twin swap on gsm8k, "
                                "same recipients); IR = answer changed; steer = P(answer == donor's | donor differs)",
                      **summary, "decode_elapsed_sec": decode_elapsed, "swap_elapsed_sec": swap_elapsed},
        },
        hyperparams={"method": sa.method, "num_latents": sa.num_latents, "perm_seed": sa.perm_seed,
                     "n_twin_cap": sa.n_twin, "max_new_tokens": max_new, "deltas": list(DELTAS)},
        seed=sa.perm_seed,
        hardware=f"{sa.hardware} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        notes=f"Zhang-style random swap vs twin swap, {sa.method} on {sa.dataset}.",
    )
    manifest = record.save(predictions=records)
    (manifest.parent / "eval_command.txt").write_text(argv + "\n")
    print(f"run_id={record.run_id} -> fill in {manifest.parent / 'notes.md'}, then: uv run python scripts/rebuild_index.py")


if __name__ == "__main__":
    main()
