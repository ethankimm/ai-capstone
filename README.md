# ai-capstone

**Do Different Latent Scratchpads Think Alike?** — CIS5980 capstone (Ethan Kim, Henning
Lindig, Kevin Nguyen) comparing four mechanisms for hidden computation before answering:

| mechanism | axis | idea |
|---|---|---|
| `explicit_cot` | baseline | visible chain-of-thought |
| `filler_tokens` | horizontal | contentless tokens (`...`) before the answer ([Pfau et al. 2024](https://arxiv.org/abs/2404.15758)) |
| `codi` | horizontal | continuous-thought self-distillation ([arXiv:2502.21074](https://arxiv.org/abs/2502.21074), [repo](https://github.com/zhenyi4/codi)) |
| `recurrent_depth` | vertical | looped transformer block, r iterations/token ([arXiv:2502.05171](https://arxiv.org/abs/2502.05171), [repo](https://github.com/seal-rg/recurrent-pretraining)) |

Backbone: HF `gpt2` (124M). Benchmark: GSM8K-Aug (`zen-E/GSM8k-Aug`, 385,620 train /
1,319 test). **Plan: [`PROPOSAL.md`](PROPOSAL.md).**

## Plan

1. **Sep 25** — reproduce all four on GPT-2 / GSM8K-Aug, verified against each paper
   (recurrent depth first — highest risk).
2. **Oct 9** — decode intermediate states (probes, J-lens, logit lens).
3. **Oct 16** — causal tests (patching, ablations) + cross-mechanism transplant.
4. **Oct 30** — sweeps: more mechanisms, benchmarks, interp methods.
5. **Nov 16** draft → early December final.

We plan for null findings; every run gets logged, failed ones included.

## What this repo is

The shared layer that keeps independent work comparable — and nothing more, on
purpose. For the Sep 25 reproductions everyone builds their mechanism their own way
(own training loop, prompt format, generation); we compare methodologies afterwards.
Three things are shared:

1. **Data** — `latentreasoning.data.gsm8k_aug.load_gsm8k_aug(split, n, seed)`; a fixed
   held-out validation set of 1000 examples taken from train (same for everyone); eval settings in `configs/mechanisms/*.yaml`
   (`eval_n=200`, `eval_seed=0`, `split=test`).
2. **Scoring** — `latentreasoning.eval.harness.score_outputs(examples, raw_outputs)`: run
   your own eval loop, hand over one raw output string per example.
3. **Run records** — `RunRecord.save(predictions=result.records)`, then fill in
   `results/<run_id>/notes.md`, then `scripts/rebuild_index.py`. Convention:
   [`.claude/skills/record-run/SKILL.md`](.claude/skills/record-run/SKILL.md).

`latentreasoning/mechanisms/<name>.py` is each owner's notes (citation, reference repo, what
`compute_steps` means for it, gotchas) — not an implementation.

## Setup

```bash
uv sync                  # base deps + `latentreasoning` as editable package
uv sync --extra train    # + torch/transformers/accelerate (RunPod; scripts/runpod_setup.sh)
uv sync --extra dev      # + pytest
```

## Quickstart (no GPU, no network)

```bash
uv run pytest
uv run python scripts/run_baseline.py     # zero-model floor over 30 bundled examples; logs a run
```

## Reproducing a mechanism

```python
from latentreasoning.data.gsm8k_aug import load_gsm8k_aug
from latentreasoning.eval.harness import score_outputs
from latentreasoning.runlog.manifest import RunRecord, ModelInfo, DatasetInfo, new_run_id

examples = load_gsm8k_aug(split="test", n=200, seed=0)
raw_outputs = my_eval_loop([ex.question for ex in examples])   # your code, your way
result = score_outputs(examples, raw_outputs)
record = RunRecord(run_id=new_run_id("codi", "repro"), mechanism="codi", stage="pilot",
                   model=ModelInfo(backbone="gpt2", checkpoint="..."),
                   dataset=DatasetInfo(split="test", n_examples=200, seed=0),
                   metrics={"final_answer_accuracy": result.final_answer_accuracy,
                            "unparseable_rate": result.unparseable_rate, "compute_steps": 6})
record.save(predictions=result.records)
```

`results/README.md` is the comparison table; `EXPERIMENTS.md` the narrative log (both
generated — never hand-edit).

## Layout

```
PROPOSAL.md             # the plan + references
latentreasoning/
  data/gsm8k_aug.py     # loader (HF or bundled sample), fixed held-out validation set
  eval/                 # metrics.py (scoring), harness.py (score_outputs / run_eval)
  mechanisms/           # one owner's-notes stub per mechanism; base.py = names + convention
  baselines/            # zero-model floor
  runlog/               # manifest.py (RunRecord), index.py (generated files)
  utils.py              # set_seed
configs/                # per-mechanism YAML: shared eval settings, your hyperparams (not loaded)
scripts/                # run_baseline.py, rebuild_index.py, runpod_setup.sh
results/<run_id>/       # manifest.json, notes.md, predictions.jsonl
tests/
```
