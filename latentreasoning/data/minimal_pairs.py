"""Same-problem minimal-pair donor generation (`steered_to_donor_audit.md` #4.3): perturb
one question number the recipient's own chain uses at or before step `step`, re-execute
the chain, and use the result as the donor for a causal-patching pair. The twin's final
answer then equals the recipient's own chain re-evaluated with the injected value --
Metric A (`matches_donor_final`) and Metric B (`matches_cf`,
`latentreasoning.eval.counterfactual`) coincide, and the cross-problem context confound
(the donor's remaining program living in question text the recipient never saw) is gone.
Standard causal-abstraction interchange-pair design (Geiger et al.).

Note: the perturbed question number is substituted everywhere it appears as a literal
operand, not just downstream of `step` in a linear chain -- GSM8K-Aug rationales often
reuse the same raw question number as an independent operand in several steps (see
Josh's house-flip example in the test suite), which `counterfactual.reeval_chain`'s
single-value propagation model doesn't capture. So a `MinimalPair`'s twin answer is
ground truth by direct re-execution, not by re-deriving it through
`counterfactual_answer`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from latentreasoning.data.gsm8k_aug import Example
from latentreasoning.eval.counterfactual import NUM_RE, fmt_num, num_equal, parse_steps, safe_eval, substitute

QUESTION_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class MinimalPair:
    original: Example
    twin: Example
    step: int  # 0-indexed chain step whose result differs between original and twin
    perturbed_number: str  # the question-text number that was changed
    delta: int


def _reexecute(steps: list[dict], seed_mapping: dict[str, str]) -> list[str] | None:
    """Re-evaluate every step's expression, substituting `seed_mapping` (the perturbed
    question number) and, as steps resolve, any of their own results that changed too.
    Returns the new per-step result list, or None if any step can't be evaluated."""
    mapping = dict(seed_mapping)
    results = []
    for st in steps:
        v = safe_eval(substitute(st["expr"], mapping))
        if v is None:
            return None
        v_str = fmt_num(v)
        results.append(v_str)
        if not num_equal(v_str, st["val"]) and st["val"] not in mapping:
            mapping[st["val"]] = v_str
    return results


def _eligible_operands(question: str, steps: list[dict], step: int) -> list[tuple[str, tuple[int, int]]]:
    """Question-text numbers that (a) occur exactly once in the question and (b) are an
    operand of some step's expression at or before `step` -- perturbing one of these is
    guaranteed to reach step `step`."""
    q_matches = list(QUESTION_NUM_RE.finditer(question))
    counts: dict[str, int] = {}
    for m in q_matches:
        counts[m.group(0)] = counts.get(m.group(0), 0) + 1
    operands: set[str] = set()
    for st in steps[: step + 1]:
        operands.update(NUM_RE.findall(st["expr"]))
    out = []
    for m in q_matches:
        tok = m.group(0)
        if counts[tok] == 1 and any(num_equal(tok, o) for o in operands):
            out.append((tok, m.span()))
    return out


def generate_minimal_pair(
    example: Example, step: int, delta: int, rng=None, require_nonneg_int: bool = True,
) -> MinimalPair | None:
    """Perturb one eligible question number by `delta`, re-execute the chain, and build a
    twin `Example`. Returns None if there's no eligible operand, the perturbed chain
    can't be evaluated, any step's result goes negative/non-integer (when
    `require_nonneg_int`), or step `step`'s result doesn't actually change (delta was
    absorbed, e.g. by a floor/rounding op)."""
    steps = parse_steps(example.rationale)
    if step >= len(steps):
        return None
    candidates = _eligible_operands(example.question, steps, step)
    if not candidates:
        return None
    tok, span = rng.choice(candidates) if rng is not None else candidates[0]
    try:
        orig_num = float(tok.replace(",", ""))
    except ValueError:
        return None
    new_num = orig_num + delta
    if require_nonneg_int and (new_num < 0 or not new_num.is_integer()):
        return None
    new_tok = fmt_num(new_num)

    new_results = _reexecute(steps, {tok: new_tok})
    if new_results is None:
        return None
    if require_nonneg_int and any(
        (safe_eval(r) is None or safe_eval(r) < 0 or not safe_eval(r).is_integer()) for r in new_results
    ):
        return None
    if num_equal(new_results[step], steps[step]["val"]):
        return None  # delta didn't propagate to the target step

    twin_question = example.question[: span[0]] + new_tok + example.question[span[1]:]

    mapping = {tok: new_tok}
    twin_exprs = []
    for st, new_val in zip(steps, new_results):
        twin_exprs.append(substitute(st["expr"], mapping))
        if not num_equal(new_val, st["val"]) and st["val"] not in mapping:
            mapping[st["val"]] = new_val
    twin_rationale = " ".join(f"<<{e}={v}>>" for e, v in zip(twin_exprs, new_results))

    # final answer: whichever step's original value equals the example's gold answer
    # (mirrors `counterfactual_answer`'s convention), falling back to the last step.
    ans_idx = len(steps) - 1
    for j in range(len(steps) - 1, -1, -1):
        if num_equal(steps[j]["val"], example.answer):
            ans_idx = j
            break
    twin_answer = new_results[ans_idx]

    twin = Example(question=twin_question, rationale=twin_rationale, answer=twin_answer, idx=example.idx)
    return MinimalPair(original=example, twin=twin, step=step, perturbed_number=tok, delta=delta)


def generate_minimal_pairs(
    examples: list[Example], deltas: tuple[int, ...] = (1, -1, 2, -2, 3, -3), rng=None,
) -> list[MinimalPair]:
    """One minimal pair per (example, step) that succeeds, trying `deltas` in order until
    one produces a valid twin -- the E3 donor pool (`steered_to_donor_audit.md` #4.3)."""
    out = []
    for ex in examples:
        n_steps = len(parse_steps(ex.rationale))
        for step in range(n_steps):
            for delta in deltas:
                pair = generate_minimal_pair(ex, step, delta, rng=rng)
                if pair is not None:
                    out.append(pair)
                    break
    return out
