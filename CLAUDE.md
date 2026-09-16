# CLAUDE.md

## Project

CIS5980 capstone (research track), "Do Different Latent Scratchpads Think Alike?" —
Ethan Kim, Henning Lindig, Kevin Nguyen. Interpretability of latent reasoning: do
different mechanisms for hidden computation before answering encode comparable,
causally faithful, portable intermediate states? Four mechanisms on a shared backbone
(HF `gpt2`, 124M) and benchmark (GSM8K-Aug): **explicit CoT** (baseline),
**filler tokens** and **CODI** (horizontal / token-wise), **recurrent depth**
(vertical / layer-wise).

**The plan is `PROPOSAL.md`** (the submitted milestone 1 doc) — read it before proposing
experiments. Compressed:

1. Reproduce each mechanism on GPT-2 / GSM8K-Aug and verify against its paper
   — **Sep 25**. Recurrent depth first (highest risk).
2. Decode intermediate states per mechanism (linear/non-linear probes, J-lens, logit
   lens); individual vs shared decoders — **Oct 9**.
3. Causality tests on decoded states (activation patching, ablations, EAP) and a
   lightweight cross-architecture mapping + transplant — **Oct 16**.
4. Sweeps: more mechanisms (Coconut, Soft Thinking, System-1.5, PCCoT), more benchmarks
   (GSM8K-Aug-NL, HotpotQA, graph reasoning), more interp methods — **Oct 30**.
5. Full draft **Nov 16**; polished final early December.

**We plan for null findings.** Extracting and transplanting latent reasoning is a
long-shot goal; a negative result at any step is a result. That is why the logging rule
below covers failed runs too.

## How we work

Three people, roughly one mechanism each. **For the Sep 25 reproductions, everyone
builds their mechanism their own way** — own training loop (reference repo or from
scratch), own prompt format, own generation code. We compare methodologies and gaps
afterwards and consolidate then, not before. So this repo is deliberately thin: it is
the shared data / scoring / run-record layer that keeps independent work comparable,
not a training framework, and the mechanism modules are owner's-notes stubs, not
implementations. Don't add shared abstractions for things only one person has needed.

## The one hard rule

**Log every real run with the `record-run` skill** (`.claude/skills/record-run/SKILL.md`):
`RunRecord.save(predictions=...)` → fill in `notes.md` → `scripts/rebuild_index.py` →
commit. Failed and null runs included. Throwaway smoke tests (n < 10) are the only
exception.

## Layout

- `PROPOSAL.md` — the plan; also a references list.
- `latentreasoning/data/gsm8k_aug.py` — loader. `load_gsm8k_aug(split=...)` (HF, network);
  `split="validation"` is a fixed held-out set of 1000 examples taken from train
  (same for everyone; `split="train"` excludes them).
  `load_local_sample()` = 30 bundled test examples, no deps. `Example.intermediate_values`
  parses the `<<a+b=c>>` rationale into per-step results — the step-2 decoding targets.
- `latentreasoning/eval/` — `metrics.py` (answer extraction + scoring); `harness.py` —
  **`score_outputs(examples, raw_outputs)` is the contract**: run your own eval loop,
  hand over one raw output string per example, get a scored `EvalResult` with
  per-example `records`. `run_eval(generate_fn, examples)` is a convenience wrapper.
- `latentreasoning/mechanisms/` — one stub module per mechanism (citation, reference repo,
  `compute_steps` definition, gotchas); `base.py` has `MECHANISM_NAMES` and the
  convention. No implementations on purpose — see "How we work".
- `latentreasoning/baselines/heuristic.py` — zero-model floor.
- `latentreasoning/runlog/` — `manifest.py` (`RunRecord` schema, validated on construction;
  `CANONICAL_METRICS` incl. names reserved for the October metrics), `index.py`
  (renders the generated files).
- `latentreasoning/utils.py` — `set_seed()`.
- `configs/` — per-mechanism YAML. Nothing loads them; they document the shared eval
  settings (`eval_n=200`, `eval_seed=0`, `split=test`) and are a place to write down
  the hyperparameters you actually used.
- `results/<run_id>/` — `manifest.json`, `notes.md`, `predictions.jsonl`
  (`train_log.json` if trained). Everything else in there is gitignored.
- `results/README.md`, `EXPERIMENTS.md` — **generated** by `scripts/rebuild_index.py`.
  Don't hand-edit; on merge conflict, take either side and rerun.
- `scripts/` — `run_baseline.py`, `rebuild_index.py`, `runpod_setup.sh`.
- `tests/` — `uv run pytest`; no GPU/model deps. Includes a freshness check on the
  generated files.

## Environment

- `uv sync` installs base deps and `latentreasoning` as an editable package (so
  `import latentreasoning` works from any cwd / notebook). `--extra train` adds
  torch/transformers/accelerate — only where you're loading a model.
- Local machines are Apple Silicon, no CUDA. Validate locally with the no-model paths
  (`pytest`, `run_baseline.py`), then run on RunPod (`scripts/runpod_setup.sh`).
- `RunRecord.author` defaults to `git config user.name` — whoever ran the experiment.
