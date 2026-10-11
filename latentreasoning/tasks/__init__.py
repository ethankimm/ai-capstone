"""Synthetic depth/width arithmetic tasks (see ``synthetic`` for the task definitions and
``generate`` for splits and the CLI)."""
from latentreasoning.tasks.synthetic import (
    DepthProblem,
    WidthProblem,
    build_ladder,
    depth_decoys,
    parse_depth_question,
    parse_question,
    parse_step,
    parse_width_question,
    render,
    sample_problem,
    width_decoys,
)

__all__ = [
    "DepthProblem",
    "WidthProblem",
    "build_ladder",
    "depth_decoys",
    "parse_depth_question",
    "parse_question",
    "parse_step",
    "parse_width_question",
    "render",
    "sample_problem",
    "width_decoys",
]
