"""GSM8K-Aug loader (HF `zen-E/GSM8k-Aug`: 385,620 train / 1,319 test, GPT-4-augmented
GSM8K used by CODI and Coconut at GPT-2 scale). Fields: `question`, `cot` (calculator
rationale, e.g. "<<16-3-4=9>> <<9*2=18>>"), `answer` (bare number). `zen-E/GSM8k-Aug-NL`
is the natural-language-rationale variant (`variant="nl"`).

HF has no validation split, so `split="validation"` is a held-out set of VALIDATION_N
examples taken from train with a fixed seed (VALIDATION_SEED) -- the same examples for
everyone -- and `split="train"` returns train minus those. `test` stays untouched for
reported numbers.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

HF_DATASET_IDS = {"cot": "zen-E/GSM8k-Aug", "nl": "zen-E/GSM8k-Aug-NL"}
SAMPLE_PATH = Path(__file__).resolve().parent / "gsm8k_aug_sample.jsonl"

VALIDATION_N = 1000
VALIDATION_SEED = 1234  # fixed on purpose -- do not vary per run

_CALC_RE = re.compile(r"<<[^<>=]*=([^<>]*)>>")


@dataclass
class Example:
    question: str
    rationale: str
    answer: str
    idx: int  # position in the original HF split (stable across shuffles)

    @property
    def intermediate_values(self) -> list[str]:
        """Results of each `<<...=x>>` step in the rationale, in order -- the natural
        decoding targets for intermediate-state probing. Empty for the NL variant."""
        return parse_intermediate_values(self.rationale)


def parse_intermediate_values(rationale: str) -> list[str]:
    return [v.strip() for v in _CALC_RE.findall(rationale)]


def load_gsm8k_aug(
    split: Literal["train", "validation", "test"] = "train",
    variant: Literal["cot", "nl"] = "cot",
    n: int | None = None,
    seed: int | None = 0,
) -> list[Example]:
    """Load from Hugging Face (needs `datasets` + network on first call). Shuffles with
    `seed` (None = keep HF order) then takes the first `n`."""
    from datasets import load_dataset  # keep this module importable without `datasets`

    hf_split = "train" if split == "validation" else split
    ds = load_dataset(HF_DATASET_IDS[variant], split=hf_split)
    examples = [
        Example(question=row["question"], rationale=row["cot"], answer=row["answer"].strip(), idx=i)
        for i, row in enumerate(ds)
    ]
    if split in ("train", "validation"):
        val_idx = set(random.Random(VALIDATION_SEED).sample(range(len(examples)), VALIDATION_N))
        examples = [ex for ex in examples if (ex.idx in val_idx) == (split == "validation")]
    if seed is not None:
        random.Random(seed).shuffle(examples)
    return examples[:n] if n is not None else examples


def load_local_sample(n: int | None = None) -> list[Example]:
    """30 real *test* examples bundled in the repo -- for smoke-testing the pipeline
    with no deps/network. Never train on these."""
    rows = [json.loads(line) for line in SAMPLE_PATH.read_text().splitlines() if line.strip()]
    examples = [Example(r["question"], r["rationale"], r["answer"], idx=i) for i, r in enumerate(rows)]
    return examples[:n] if n is not None else examples
