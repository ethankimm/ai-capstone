# t-runs: route and necessity analyses (2026-11-07 onward)

Copies of the analysis scripts, printed logs and tables from `~/Documents/Penn/CIS5980/codi_audit_artifacts/`, where they were written
and run. Run records in `results/` cite them as `codi_audit_artifacts/<name>`; the file here is the same script.

**`t_predictions.md` is the pre-registration log.** Each entry is timestamped and was written before any output of the runs it names.
From 2026-11-08 on, each new entry is committed here before its run starts, so the commit time backs the written timestamp.

- `t1`-`t15*.py`: the analyses (model jobs; run through `run_guarded.sh` with `caffeinate -dims`, one at a time on the Mac). They take
  `scripts/codi_null_audit` (for `harness.py`, `rc.py`), a device and an output folder; see each docstring.
- `record9`-`record*.py`: turn a run's JSON output into a `RunRecord` (`uv run python recordN.py <audit dir>`).
- `t*.log`, `t*_table*.md`: the printed outputs. Raw JSON outputs stay in `codi_audit_artifacts` (the records hold per-item predictions).
- `a26`-`a29`, `b23`-`b26`: earlier one-offs from the same folder that later records cite.

Paths inside the scripts point at the audit folder layout (`BASE` = the folder holding `ai-capstone/`, `codi/` and the run folders).
