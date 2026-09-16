from latentreasoning.runlog.index import render_experiments_log, render_results_readme
from latentreasoning.runlog.manifest import REPO_ROOT, RESULTS_DIR

STALE = "stale -- run `uv run python scripts/rebuild_index.py` and commit the result"


def test_results_readme_up_to_date():
    assert (RESULTS_DIR / "README.md").read_text() == render_results_readme(), f"results/README.md {STALE}"


def test_experiments_log_up_to_date():
    assert (REPO_ROOT / "EXPERIMENTS.md").read_text() == render_experiments_log(), f"EXPERIMENTS.md {STALE}"
