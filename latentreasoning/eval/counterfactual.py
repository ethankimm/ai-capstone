"""Counterfactual-propagation scoring for activation-patching runs, shared across every
mechanism (and the explicit-CoT positive control) so "did the patch work" means the same
thing everywhere.

Origin: `steered_to_donor` (`answer_patched == donor.answer`, "Metric A") is what every
CODI/Coconut patching script logged first. The positive control
(`scripts/patch_explicit_cot.py`) was scored on a different, weaker, better-motivated
criterion instead: does the answer equal the counterfactual value obtained by
substituting the injected value into the recipient's OWN remaining chain and
re-evaluating it ("Metric B", `matches_cf`) -- because for a cross-problem donor, Metric
A also requires the recipient to import the donor's entire remaining program, which the
recipient never saw. See `steered_to_donor_audit.md` for the full writeup; this module
is the promoted-to-shared version of that audit's re-scorer (E0/E1).

`parse_steps` / `reeval_chain` / `counterfactual_answer` / `num_equal` were lifted
unchanged from `patch_explicit_cot.py` (which now imports them from here). `score_patch`
is new: one call gives every metric (`matches_cf`, `matches_donor_final`,
`answer_changed`, `outcome`) for a single patched record, aligned (a known step) or
unaligned (try every step and take the closest thing to a hit).
"""
from __future__ import annotations

import re

STEP_RE = re.compile(r"<<([^<>=]*)=([^<>]*)>>")
# Unsigned on purpose: inside a calculator expression "16-4" the minus is the operator, and
# a signed pattern would swallow it into "-4" and miss the operand. (Signed results such as
# "<<5-8=-3>>" are rare in GSM8K-Aug and are simply not matched.)
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
SAFE_EXPR_RE = re.compile(r"^[\d\s.+\-*/()]+$")

OUTCOME_ORDER = (
    "unparseable", "unchanged", "counterfactual", "donor_final",
    "recipient_gold", "recipient_intermediate", "donor_intermediate", "other_number",
)


def num_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    a, b = a.strip().replace(",", ""), b.strip().replace(",", "")
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def safe_eval(expr: str) -> float | None:
    expr = expr.replace(",", "").strip()
    if not expr or not SAFE_EXPR_RE.match(expr):
        return None
    try:
        v = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 -- whitelisted charset above
    except Exception:  # noqa: BLE001 -- ZeroDivisionError, SyntaxError, ...
        return None
    return float(v) if isinstance(v, (int, float)) else None


def operand_in(val: str, expr: str) -> bool:
    return any(num_equal(m, val) for m in NUM_RE.findall(expr))


def substitute(expr: str, mapping: dict[str, str]) -> str:
    pieces = re.split(r"(\d[\d,]*(?:\.\d+)?)", expr)
    out = []
    for p in pieces:
        rep = p
        if p and NUM_RE.fullmatch(p):
            for old, new in mapping.items():
                if num_equal(p, old):
                    rep = new
                    break
        out.append(rep)
    return "".join(out)


def parse_steps(text: str) -> list[dict]:
    """`<<expr=val>>` steps in order, with the value's character span in `text`."""
    steps = []
    for m in STEP_RE.finditer(text):
        raw_val = m.group(2)
        lead = len(raw_val) - len(raw_val.lstrip())
        val = raw_val.strip()
        vs = m.start(2) + lead
        steps.append({"expr": m.group(1).strip(), "val": val, "val_span": (vs, vs + len(val))})
    return steps


def reeval_chain(steps: list[dict], k: int, new_val: str) -> dict[int, str] | None:
    """Substitute `new_val` for step k's result in the downstream expressions of a chain and
    re-evaluate it forward, propagating any changed results. Returns {j: new result} for
    j > k, or None if any downstream step can't be evaluated."""
    mapping = {steps[k]["val"]: new_val}
    results: dict[int, str] = {}
    for j in range(k + 1, len(steps)):
        r = safe_eval(substitute(steps[j]["expr"], mapping))
        if r is None:
            return None
        r_str = fmt_num(r)
        results[j] = r_str
        if not num_equal(r_str, steps[j]["val"]) and steps[j]["val"] not in mapping:
            mapping[steps[j]["val"]] = r_str
    return results


def counterfactual_answer(steps: list[dict], k: int, new_val: str, base_answer: str | None) -> str | None:
    """The answer a faithful continuation would give if step k's result were `new_val`:
    re-evaluate the chain and return the re-evaluated version of whichever downstream step
    the baseline answer came from. None if undefined (answer isn't a downstream result, or
    the chain can't be evaluated)."""
    if base_answer is None:
        return None
    results = reeval_chain(steps, k, new_val)
    if results is None:
        return None
    for j in range(len(steps) - 1, k, -1):
        if num_equal(steps[j]["val"], base_answer):
            return results[j]
    return None


def counterfactual_candidates(steps: list[dict], donor_value: str, base_answer: str | None) -> set[str]:
    """Every counterfactual answer obtainable by injecting `donor_value` at ANY step
    k < len(steps)-1 of `steps` and re-evaluating -- used when the patched step isn't
    known (unaligned pairs). Inflates the "hit" rate vs. a single known step; pair with
    a permutation null."""
    out: set[str] = set()
    for k in range(len(steps) - 1):
        cf = counterfactual_answer(steps, k, donor_value, base_answer)
        if cf is not None:
            out.add(cf)
    return out


def _assemble(
    answer_base: str | None, answer_patched: str | None, recipient_gold: str | None,
    donor_final: str | None, matches_cf: bool | None, recipient_values: list[str],
    donor_values: list[str],
) -> dict:
    answer_changed = (not num_equal(answer_patched, answer_base)) if (answer_patched or answer_base) else False
    matches_donor_final = num_equal(answer_patched, donor_final) if donor_final is not None else None

    if answer_patched is None:
        outcome = "unparseable"
    elif num_equal(answer_patched, answer_base):
        outcome = "unchanged"
    elif matches_cf:
        outcome = "counterfactual"
    elif donor_final is not None and num_equal(answer_patched, donor_final):
        outcome = "donor_final"
    elif recipient_gold is not None and num_equal(answer_patched, recipient_gold):
        outcome = "recipient_gold"
    elif any(num_equal(answer_patched, v) for v in recipient_values):
        outcome = "recipient_intermediate"
    elif any(num_equal(answer_patched, v) for v in donor_values):
        outcome = "donor_intermediate"
    else:
        outcome = "other_number"

    return {
        "answer_changed": answer_changed,
        "matches_donor_final": matches_donor_final,
        "matches_cf": matches_cf,
        "outcome": outcome,
    }


def score_patch(
    *,
    answer_base: str | None,
    answer_patched: str | None,
    recipient_gold: str | None = None,
    donor_final: str | None = None,
    recipient_chain: list[dict] | None = None,
    donor_value: str | None = None,
    step: int | None = None,
    recipient_values: list[str] | None = None,
    donor_values: list[str] | None = None,
) -> dict:
    """Score one patched record under Metric B (`matches_cf`) with Metric A
    (`matches_donor_final`) as a secondary column, plus an outcome taxonomy.

    `recipient_chain`: recipient's gold `<<expr=val>>` chain (`parse_steps(rationale)`),
    used to compute the counterfactual. `donor_value`: the value injected at the patch
    site. `step`: 0-indexed chain position the patch targeted, if known (step-aligned
    pairs); `None` means try every step and take the union for this one donor value
    (unaligned single-value case -- for "any donor value at any step", see
    `score_patch_unaligned`). `recipient_values`/`donor_values`: intermediate values for
    the taxonomy's `recipient_intermediate`/`donor_intermediate` buckets.
    """
    recipient_values = recipient_values or []
    donor_values = donor_values or []

    matches_cf = None
    if recipient_chain and donor_value is not None:
        if step is not None:
            cf = counterfactual_answer(recipient_chain, step, donor_value, answer_base)
            if cf is not None:
                matches_cf = num_equal(answer_patched, cf) and not num_equal(answer_base, cf)
        else:
            cands = counterfactual_candidates(recipient_chain, donor_value, answer_base)
            if cands:
                matches_cf = any(
                    num_equal(answer_patched, c) and not num_equal(answer_base, c) for c in cands
                )

    return _assemble(answer_base, answer_patched, recipient_gold, donor_final, matches_cf,
                      recipient_values, donor_values)


def qualifying_steps(steps: list[dict]) -> list[int]:
    """Step indices k (0-indexed, k < len(steps) - 1) whose result is an operand of the
    NEXT step's expression -- so a perturbation injected at k CAN propagate to k+1 and
    beyond. `steered_to_donor_audit.md` #4.2 condition (i); used to build qualified
    patch pools for E2/E3 (a step with no downstream use makes `matches_cf` undefined
    by construction, not just null)."""
    return [k for k in range(len(steps) - 1) if operand_in(steps[k]["val"], steps[k + 1]["expr"])]


def site_to_step(site_idx: int, n_sites: int, max_step: int) -> int:
    """Deterministic positional map from a latent 'site' index (0-indexed, 0..n_sites-1
    -- a CODI iteration or a Coconut pass) to the step index (0-indexed) it is tested
    against, splitting `n_sites` into `max_step` contiguous groups. Used instead of a
    decoding-accuracy-fit mapping (`decode_patch_codi.py`'s `best_iter_for_step`) when
    EVERY site must be tested, including ones a decoding-accuracy fit would never pick
    as "best" for any step (steered_to_donor_audit.md E2 -- e.g. CODI's non-decodable
    but load-bearing z0/z3)."""
    if max_step <= 0:
        return 0
    return min(site_idx * max_step // n_sites, max_step - 1)


def score_patch_unaligned(
    *,
    answer_base: str | None,
    answer_patched: str | None,
    recipient_gold: str | None = None,
    donor_final: str | None = None,
    recipient_chain: list[dict] | None = None,
    donor_values: list[str] | None = None,
    recipient_values: list[str] | None = None,
    donor_intermediate_values: list[str] | None = None,
) -> dict:
    """`score_patch`'s unaligned variant, for pairs with no recorded patch step (donor
    chosen by different final answer only, e.g. the DAS / raw-interchange sweeps):
    `matches_cf` at ANY step, injecting ANY of the donor's own intermediate values --
    `steered_to_donor_audit.md` section 3.3's "Metric B at any step" convention. This
    inflates the chance rate; pair with a permutation-null baseline (same patched
    answers, donors shuffled) when reporting it."""
    donor_values = donor_values or []
    cands: set[str] = set()
    if recipient_chain:
        for dv in donor_values:
            cands |= counterfactual_candidates(recipient_chain, dv, answer_base)
    matches_cf = None
    if cands:
        matches_cf = any(
            num_equal(answer_patched, c) and not num_equal(answer_base, c) for c in cands
        )
    return _assemble(answer_base, answer_patched, recipient_gold, donor_final, matches_cf,
                      recipient_values or [], donor_intermediate_values or [])
