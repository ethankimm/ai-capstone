# Codex project instructions

Read [CLAUDE.md](CLAUDE.md) before working in this repository. It is the shared
source for project context, research scope, data/scoring contracts, and run-recording
rules for both Claude Code and Codex. Read `PROPOSAL.md` before proposing experiments.

The `record-run` skill is available to Codex at
`.agents/skills/record-run/SKILL.md`. That directory links to the existing
`.claude/skills/record-run` directory; maintain the workflow there so both tools use
the same instructions. Follow it when recording or comparing experiments.

## Environment and shared-node rules

These rules take precedence over the Apple Silicon / RunPod environment assumptions
in `CLAUDE.md`; this checkout is on a shared Linux GPU node.

- Before every vLLM server launch or restart, run
  `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`.
  Choose GPUs with near-zero used memory. If the requested GPUs are busy, report
  that and wait or choose free GPUs; never launch on busy GPUs anyway. Recheck on
  restarts because another user may have claimed the GPUs in the meantime.
- Report timestamps and ETAs in California time (`America/Los_Angeles`, PST/PDT).
  Use `TZ=America/Los_Angeles date` for the current local time. Convert UTC log
  timestamps and run-ID prefixes before reporting them, and label the result PT.
  Keep machine-readable run IDs and `timestamp_utc` fields in UTC; include UTC
  alongside PT only when useful for matching a log entry.
