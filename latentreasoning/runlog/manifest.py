"""Shared run-record schema (the machine-readable half of the run-logging convention --
see `.claude/skills/record-run/SKILL.md` for the procedure).

`RunRecord.save()` writes `results/<run_id>/manifest.json`, `predictions.jsonl` (if
given), and a `notes.md` template to fill in. `scripts/rebuild_index.py` regenerates
`results/README.md` + `EXPERIMENTS.md` from those.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from latentreasoning.mechanisms.base import MECHANISM_NAMES

MECHANISMS = MECHANISM_NAMES
STAGES = ("smoke_test", "pilot", "full_run")
SPLITS = ("train", "validation", "test")

# Use these keys when they apply; anything mechanism-specific goes under
# `metrics["extra"]` so jq/pandas scans across manifests stay simple.
CANONICAL_METRICS = (
    "final_answer_accuracy",  # numeric match on the final answer (latentreasoning.eval.metrics)
    "unparseable_rate",       # fraction of outputs with no extractable number
    "compute_steps",          # filler length / loop iterations r / # continuous-thought tokens
    "train_loss",
    "eval_loss",
    "sec_per_example",
    # Reserved for the Oct milestones (PROPOSAL.md metrics) so three people don't invent
    # three spellings. Nothing computes these yet; define them in notes.md when you do.
    "decoding_accuracy",            # intermediate-state decoding accuracy (probes / lenses)
    "early_termination_necessity",  # accuracy drop when the scratchpad is truncated at
                                    # inference on a full-budget model -- NOT the same as
                                    # a compute_steps sweep (train+eval at each budget)
    "intervention_accuracy",        # counterfactual intervention accuracy
    "transplant_success",           # cross-mechanism transplant success
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_info() -> dict[str, Any]:
    try:
        return {"commit": _git("rev-parse", "--short", "HEAD"), "dirty": _git("status", "--porcelain") != ""}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": None}


def default_author() -> str:
    try:
        return _git("config", "user.name")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def new_run_id(mechanism: str, slug: str) -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_{mechanism}_{slug}"


@dataclass
class ModelInfo:
    backbone: str  # HF id ("gpt2") or "gpt2-edited"; "none" for model-free baselines
    checkpoint: str  # HF id, or path under results/<run_id>/ (not committed)
    n_params: int | None = None


@dataclass
class DatasetInfo:
    name: str = "gsm8k-aug"
    split: str = "test"
    n_examples: int = 0
    seed: int | None = 0  # shuffle/subset seed passed to the loader


@dataclass
class RunRecord:
    run_id: str
    mechanism: str  # one of MECHANISMS, or "baseline_<something>"
    stage: str  # one of STAGES
    model: ModelInfo
    dataset: DatasetInfo
    metrics: dict[str, Any]
    hyperparams: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None  # training/generation RNG seed (latentreasoning.utils.set_seed); None if no RNG
    hardware: str = "local-cpu"
    author: str = field(default_factory=default_author)
    notes: str = ""
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git: dict[str, Any] = field(default_factory=git_info)

    def __post_init__(self) -> None:
        if self.mechanism not in MECHANISMS and not self.mechanism.startswith("baseline_"):
            raise ValueError(f"mechanism must be one of {MECHANISMS} or start with 'baseline_', got {self.mechanism!r}")
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {self.stage!r}")
        if self.dataset.split not in SPLITS:
            raise ValueError(f"dataset.split must be one of {SPLITS}, got {self.dataset.split!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, out_dir: Path | str | None = None, predictions: list[dict] | None = None) -> Path:
        """Write results/<run_id>/{manifest.json, predictions.jsonl, notes.md}. Returns
        the manifest path. `predictions` is `EvalResult.records` -- always pass it for
        real runs so per-example outputs aren't lost. notes.md is only created if absent.
        """
        out_dir = Path(out_dir) if out_dir else RESULTS_DIR / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")
        if predictions is not None:
            with (out_dir / "predictions.jsonl").open("w") as f:
                for r in predictions:
                    f.write(json.dumps(r, default=str) + "\n")
        notes = out_dir / "notes.md"
        if not notes.exists():
            notes.write_text(notes_template(self))
        return path


def notes_template(record: RunRecord) -> str:
    m = record.metrics
    date = record.timestamp_utc[:10]
    headline = ", ".join(
        f"`{k}={m[k]:.3f}`" if isinstance(m.get(k), float) else f"`{k}={m[k]}`"
        for k in ("final_answer_accuracy", "unparseable_rate", "compute_steps") if k in m
    )
    return f"""## {date} — TODO short title ({record.mechanism}, run_id: {record.run_id})

**Goal:** TODO what question this run was trying to answer.
**Mechanism / model:** `{record.mechanism}`, {record.model.backbone} / {record.model.checkpoint}, compute_steps={m.get('compute_steps', '-')}.
**Data:** {record.dataset.name} {record.dataset.split} n={record.dataset.n_examples} seed={record.dataset.seed}; run seed={record.seed}.
**Hyperparams:** {record.hyperparams or 'none'}
**Command:** TODO
**Headline results:** {headline}
**Interpretation:** TODO
**Gotchas hit:** TODO / none
**Caveats:** TODO
**Next:** TODO
"""


def load_all_runs(results_dir: Path | str = RESULTS_DIR) -> list[dict[str, Any]]:
    """Every results/*/manifest.json, oldest first."""
    runs = [json.loads(p.read_text()) for p in Path(results_dir).glob("*/manifest.json")]
    return sorted(runs, key=lambda r: (r.get("timestamp_utc", ""), r["run_id"]))


def index_row(record: RunRecord | dict[str, Any]) -> str:
    """One markdown row for results/README.md."""
    r = record.to_dict() if isinstance(record, RunRecord) else record
    m = r["metrics"]
    acc = m.get("final_answer_accuracy")
    acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else "-"
    return (
        f"| [{r['run_id']}]({r['run_id']}/) | {r['mechanism']} | {r['model']['backbone']} | "
        f"{r['dataset']['split']} (n={r['dataset']['n_examples']}) | {m.get('compute_steps', '-')} | "
        f"{acc_str} | {r.get('seed', '-')} | {r['stage']} | {r['author']} |"
    )
