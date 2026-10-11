"""Synthetic arithmetic tasks with two knobs: ``n_chains`` (width) and ``n_steps`` (depth).

Only the two edges of the grid are implemented:

* **depth** -- ``n_chains=1, n_steps=1..12``: one serial chain,
  ``a = 7*3; b = a-5; c = b*2; d = c+9. What is d?``. Ops ``+ - *``; new operands 1..9;
  variable names a..l in order. Step 1 has two literal operands; step k>1 is
  ``<prev var><op><operand>``. Every intermediate value (and the answer) lies in 0..99
  (single GPT-2 token, no mod 100). Multiplication by 1 is excluded everywhere (it
  would make a step skippable; at step 1 neither operand of ``*`` may be 1), and so is
  ``*`` when the previous value is 0 (``0*c`` = 0 regardless of ``c``; v1.2). A value
  of 0 itself is allowed and may be followed by ``+``. With these
  ops every step is injective in the previous value, so an operand change that changes
  one step's value always changes the final answer.
  Sampling is per step: step 1 picks ``(op, x, y)`` uniformly among valid triples, each
  later step picks ``(op, operand)`` uniformly among choices that keep the value in
  0..99. (Not uniform over chains: deep chains are biased toward whatever ops stay
  feasible -- see the op mix in ``stats.json``.)

* **width** -- ``n_chains=2..16, n_steps=1``: independent single-digit products,
  ``7*3, 4*6, 9*2. Which product is largest?``. Operands are drawn uniformly from
  **2..9** (1 excluded so every product is a genuine multiplication, matching the
  depth task's no-``*1`` rule). The largest product must be unique (strictly greater
  than all others). Sampling is answer-balanced (see :func:`sample_width`): the max
  value is uniform over achievable products, the others are uniform pairs strictly
  below it.

``n_chains > 1 and n_steps > 1`` (the two-knob interior) is not implemented.

Twins (counterfactual single-operand edits) and decoys (edits that leave the answer
unchanged) are built here too; split construction and file writing live in
``latentreasoning.tasks.generate``.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, replace

VAR_NAMES = "abcdefghijkl"
MAX_DEPTH = len(VAR_NAMES)  # 12
WIDTH_LEVELS = range(2, 17)
DEPTH_LEVELS = range(1, MAX_DEPTH + 1)
VALUE_MAX = 99
OPS = ("+", "-", "*")
DEPTH_OPERANDS = range(1, 10)
WIDTH_OPERANDS = range(2, 10)
LADDER_TARGETS = (0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00)
LADDER_TOLERANCE = 1.5  # candidate is "within tolerance" if t/1.5 <= rel <= t*1.5


def apply_op(op: str, x: int, y: int) -> int:
    if op == "+":
        return x + y
    if op == "-":
        return x - y
    if op == "*":
        return x * y
    raise ValueError(op)


def in_range(v: int) -> bool:
    return 0 <= v <= VALUE_MAX


# ---------------------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthProblem:
    """``x0 ops[0] operands[0]`` then ``prev ops[k] operands[k]`` for k >= 1.

    ``swapped[k]`` (k >= 1) renders step k as ``operand op prev`` (only produced by
    commute decoys; always False for base problems). ``swapped[0]`` is always False --
    swapping step 1's literals is just a different ``(x0, operands[0])``.
    """

    x0: int
    ops: tuple[str, ...]
    operands: tuple[int, ...]
    swapped: tuple[bool, ...]

    @property
    def n_steps(self) -> int:
        return len(self.ops)

    def values(self) -> list[int]:
        vals = []
        prev = self.x0
        for k, (op, c) in enumerate(zip(self.ops, self.operands)):
            prev = apply_op(op, prev, c)
            vals.append(prev)
        return vals

    def step_operands(self, k: int) -> tuple[int, int]:
        """Numeric (left, right) of step k as rendered."""
        if k == 0:
            return self.x0, self.operands[0]
        prev = self.values()[k - 1]
        c = self.operands[k]
        return (c, prev) if self.swapped[k] else (prev, c)

    def key(self) -> tuple:
        """Canonical content key used for split disjointness."""
        return ("depth", self.x0, self.ops, self.operands, self.swapped)


def depth_step_ok(op: str, prev: int, c: int) -> bool:
    # no `*1` (skippable step) and no `0*c` (v1.2: would pin every later * step at 0)
    if op == "*" and (c == 1 or prev == 0):
        return False
    return in_range(apply_op(op, prev, c))


def depth_first_step_choices() -> list[tuple[str, int, int]]:
    out = []
    for op in OPS:
        for x in DEPTH_OPERANDS:
            for y in DEPTH_OPERANDS:
                if op == "*" and (x == 1 or y == 1):
                    continue
                if in_range(apply_op(op, x, y)):
                    out.append((op, x, y))
    return out


_FIRST_CHOICES = depth_first_step_choices()


def depth_next_choices(prev: int) -> list[tuple[str, int]]:
    return [(op, c) for op in OPS for c in DEPTH_OPERANDS if depth_step_ok(op, prev, c)]


_NEXT_CHOICES = {v: depth_next_choices(v) for v in range(VALUE_MAX + 1)}


def depth_valid(p: DepthProblem) -> bool:
    if p.ops[0] == "*" and (p.x0 == 1 or p.operands[0] == 1):
        return False
    if not all(1 <= c <= 9 for c in (p.x0, *p.operands)):
        return False
    prev = p.x0
    for op, c in zip(p.ops, p.operands):
        if not depth_step_ok(op, prev, c):
            return False
        prev = apply_op(op, prev, c)
    return True


def sample_depth(rng: random.Random, n_steps: int) -> DepthProblem:
    if not 1 <= n_steps <= MAX_DEPTH:
        raise ValueError(f"n_steps must be in 1..{MAX_DEPTH}")
    op, x, y = rng.choice(_FIRST_CHOICES)
    ops, operands = [op], [y]
    prev = apply_op(op, x, y)
    for _ in range(n_steps - 1):
        op, c = rng.choice(_NEXT_CHOICES[prev])  # never empty: '+1' or '-1' always fits
        ops.append(op)
        operands.append(c)
        prev = apply_op(op, prev, c)
    return DepthProblem(x, tuple(ops), tuple(operands), (False,) * n_steps)


def enumerate_depth(n_steps: int) -> list[DepthProblem]:
    """All base problems with ``n_steps`` steps (only call for small levels)."""
    out: list[DepthProblem] = []

    def rec(x0, ops, operands, prev):
        if len(ops) == n_steps:
            out.append(DepthProblem(x0, tuple(ops), tuple(operands), (False,) * n_steps))
            return
        for op, c in _NEXT_CHOICES[prev]:
            rec(x0, ops + [op], operands + [c], apply_op(op, prev, c))

    for op, x, y in _FIRST_CHOICES:
        rec(x, [op], [y], apply_op(op, x, y))
    return out


def depth_space_size(n_steps: int) -> int:
    counts = [0] * (VALUE_MAX + 1)
    for op, x, y in _FIRST_CHOICES:
        counts[apply_op(op, x, y)] += 1
    for _ in range(n_steps - 1):
        nxt = [0] * (VALUE_MAX + 1)
        for v, k in enumerate(counts):
            if k:
                for op, c in _NEXT_CHOICES[v]:
                    nxt[apply_op(op, v, c)] += k
        counts = nxt
    return sum(counts)


def render_depth(p: DepthProblem) -> dict:
    vals = p.values()
    parts, steps = [], []
    for k, op in enumerate(p.ops):
        var = VAR_NAMES[k]
        if k == 0:
            lhs = f"{p.x0}{op}{p.operands[0]}"
        else:
            prev_var = VAR_NAMES[k - 1]
            c = p.operands[k]
            lhs = f"{c}{op}{prev_var}" if p.swapped[k] else f"{prev_var}{op}{c}"
        parts.append(f"{var} = {lhs}")
        left, right = p.step_operands(k)
        steps.append(f"<<{left}{op}{right}={vals[k]}>>")
    question = "; ".join(parts) + f". What is {VAR_NAMES[p.n_steps - 1]}?"
    return {"question": question, "steps": steps, "answer": str(vals[-1])}


_DEPTH_ASSIGN = re.compile(r"^([a-l]) = (\d+|[a-l])([+\-*])(\d+|[a-l])$")


def parse_depth_question(question: str) -> DepthProblem:
    """Inverse of :func:`render_depth` (handles commuted ``3*a`` forms)."""
    m = re.fullmatch(r"(.*)\. What is ([a-l])\?", question)
    if not m:
        raise ValueError(f"unparseable depth question: {question!r}")
    body, asked = m.groups()
    ops, operands, swapped = [], [], []
    x0 = None
    for k, part in enumerate(body.split("; ")):
        am = _DEPTH_ASSIGN.match(part)
        if not am:
            raise ValueError(f"bad assignment {part!r}")
        var, left, op, right = am.groups()
        if var != VAR_NAMES[k]:
            raise ValueError(f"expected variable {VAR_NAMES[k]}, got {var}")
        if k == 0:
            x0, c, sw = int(left), int(right), False
        else:
            prev_var = VAR_NAMES[k - 1]
            if left == prev_var and right.isdigit():
                c, sw = int(right), False
            elif right == prev_var and left.isdigit():
                c, sw = int(left), True
            else:
                raise ValueError(f"step {k} must reference {prev_var}: {part!r}")
        ops.append(op)
        operands.append(c)
        swapped.append(sw)
    if asked != VAR_NAMES[len(ops) - 1]:
        raise ValueError("question must ask for the last variable")
    return DepthProblem(x0, tuple(ops), tuple(operands), tuple(swapped))


# ---------------------------------------------------------------------------------------
# Width
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WidthProblem:
    pairs: tuple[tuple[int, int], ...]

    @property
    def n_chains(self) -> int:
        return len(self.pairs)

    def products(self) -> list[int]:
        return [a * b for a, b in self.pairs]

    def argmax(self) -> int | None:
        """Index of the unique largest product, or None on a tie."""
        prods = self.products()
        m = max(prods)
        idx = [i for i, p in enumerate(prods) if p == m]
        return idx[0] if len(idx) == 1 else None

    def key(self) -> tuple:
        return ("width", self.pairs)


def width_valid(p: WidthProblem) -> bool:
    return all(a in WIDTH_OPERANDS and b in WIDTH_OPERANDS for a, b in p.pairs) and p.argmax() is not None


_WIDTH_PAIRS = [(a, b) for a in WIDTH_OPERANDS for b in WIDTH_OPERANDS]


_PAIRS_BY_PRODUCT: dict[int, list[tuple[int, int]]] = {}
for _a, _b in _WIDTH_PAIRS:
    _PAIRS_BY_PRODUCT.setdefault(_a * _b, []).append((_a, _b))
# Max values: every achievable product that leaves at least one strictly smaller pair.
WIDTH_MAX_VALUES = sorted(v for v in _PAIRS_BY_PRODUCT if v > min(_PAIRS_BY_PRODUCT))
_PAIRS_BELOW = {m: [p for p in _WIDTH_PAIRS if p[0] * p[1] < m] for m in WIDTH_MAX_VALUES}


def width_problem_prob(p: WidthProblem) -> float:
    """Probability of ``p`` under :func:`sample_width` (0 if the max is not unique)."""
    i = p.argmax()
    if i is None:
        return 0.0
    m = max(p.products())
    n = p.n_chains
    return (1 / len(WIDTH_MAX_VALUES)) * (1 / len(_PAIRS_BY_PRODUCT[m])) * (1 / n) * (
        1 / len(_PAIRS_BELOW[m])
    ) ** (n - 1)


def sample_width(rng: random.Random, n_chains: int) -> WidthProblem:
    """Answer-balanced sampler (v1.1): the max product value is uniform over
    ``WIDTH_MAX_VALUES``; its operand pair is uniform among pairs with that product; every
    other product is a uniform pair with a strictly smaller product; the max's position
    is uniform. (Uniform-iid sampling made 72/81 dominate at high width.)"""
    if n_chains < 2:
        raise ValueError("width task needs n_chains >= 2")
    m = rng.choice(WIDTH_MAX_VALUES)
    top = rng.choice(_PAIRS_BY_PRODUCT[m])
    others = [rng.choice(_PAIRS_BELOW[m]) for _ in range(n_chains - 1)]
    pos = rng.randrange(n_chains)
    return WidthProblem(tuple(others[:pos] + [top] + others[pos:]))


def enumerate_width(n_chains: int) -> list[WidthProblem]:
    import itertools

    out = []
    for combo in itertools.product(_WIDTH_PAIRS, repeat=n_chains):
        p = WidthProblem(tuple(combo))
        if p.argmax() is not None:
            out.append(p)
    return out


def width_space_size(n_chains: int) -> int:
    """Number of ordered product lists with a unique maximum (exact)."""
    from collections import Counter

    cnt = Counter(a * b for a, b in _WIDTH_PAIRS)
    total = 0
    for m, k in cnt.items():
        below = sum(v for p, v in cnt.items() if p < m)
        total += n_chains * k * below ** (n_chains - 1)
    return total


def render_width(p: WidthProblem) -> dict:
    prods = p.products()
    question = ", ".join(f"{a}*{b}" for a, b in p.pairs) + ". Which product is largest?"
    steps = [f"<<{a}*{b}={a * b}>>" for a, b in p.pairs]
    return {"question": question, "steps": steps, "answer": str(max(prods))}


_WIDTH_Q = re.compile(r"^(.*)\. Which product is largest\?$")


def parse_width_question(question: str) -> WidthProblem:
    m = _WIDTH_Q.match(question)
    if not m:
        raise ValueError(f"unparseable width question: {question!r}")
    pairs = []
    for part in m.group(1).split(", "):
        a, b = part.split("*")
        pairs.append((int(a), int(b)))
    return WidthProblem(tuple(pairs))


# ---------------------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------------------


def sample_problem(rng: random.Random, n_chains: int, n_steps: int):
    """The single generator. Only the grid edges are implemented."""
    if n_chains > 1 and n_steps > 1:
        raise NotImplementedError("Phase B two-knob grid")
    if n_chains == 1:
        return sample_depth(rng, n_steps)
    return sample_width(rng, n_chains)


def render(p) -> dict:
    return render_depth(p) if isinstance(p, DepthProblem) else render_width(p)


def parse_question(task: str, question: str):
    return parse_depth_question(question) if task == "depth" else parse_width_question(question)


_STEP_RE = re.compile(r"^<<(\d+)([+\-*])(\d+)=(\d+)>>$")


def parse_step(step: str) -> tuple[int, str, int, int]:
    m = _STEP_RE.match(step)
    if not m:
        raise ValueError(f"bad step {step!r}")
    l, op, r, v = m.groups()
    return int(l), op, int(r), int(v)


# ---------------------------------------------------------------------------------------
# Twins and decoys
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    problem: object
    step_changed: int
    operand_position: int
    old_operand: int
    new_operand: int
    old_step_value: int
    new_step_value: int

    @property
    def rel_change_step(self) -> float:
        return abs(self.new_step_value - self.old_step_value) / self.old_step_value


def depth_twin_candidates(p: DepthProblem) -> list[_Candidate]:
    """Every in-distribution single-operand edit that changes the edited step's value.

    For step 0 ``operand_position`` is 0 (x0) or 1 (second literal); for later steps it
    is always 1 (the literal; position 0 is the previous variable). Edits obey the same
    constraints as generation (operands 1..9, no ``*1``, all values in 0..99). Edits whose
    step value is unchanged (``0*c``) are not twins and are dropped, as are edits at a
    step whose base value is 0 (relative change undefined).
    """
    base_vals = p.values()
    out = []
    for k in range(p.n_steps):
        positions = (0, 1) if k == 0 else (1,)
        for pos in positions:
            old = p.x0 if (k == 0 and pos == 0) else p.operands[k]
            for new in DEPTH_OPERANDS:
                if new == old:
                    continue
                if k == 0 and pos == 0:
                    q = replace(p, x0=new)
                else:
                    ops_ = list(p.operands)
                    ops_[k] = new
                    q = replace(p, operands=tuple(ops_))
                if not depth_valid(q):
                    continue
                vals = q.values()
                if vals[k] == base_vals[k] or base_vals[k] == 0:
                    continue
                out.append(_Candidate(q, k, pos, old, new, base_vals[k], vals[k]))
    return out


def width_twin_candidates(p: WidthProblem) -> list[_Candidate]:
    """Edits of one operand of the largest product that change the (unique) answer."""
    i = p.argmax()
    old_ans = max(p.products())
    out = []
    for pos in (0, 1):
        old = p.pairs[i][pos]
        for new in WIDTH_OPERANDS:
            if new == old:
                continue
            pair = (new, p.pairs[i][1]) if pos == 0 else (p.pairs[i][0], new)
            pairs = list(p.pairs)
            pairs[i] = pair
            q = WidthProblem(tuple(pairs))
            if q.argmax() is None or max(q.products()) == old_ans:
                continue
            out.append(_Candidate(q, i, pos, old, new, old_ans, pair[0] * pair[1]))
    return out


def _answer(p) -> int:
    return p.values()[-1] if isinstance(p, DepthProblem) else max(p.products())


def _ladder_rel(task: str, base, cand: _Candidate) -> float:
    if task == "depth":
        return cand.rel_change_step
    old = _answer(base)
    return abs(_answer(cand.problem) - old) / old


def build_ladder(task: str, base, rng: random.Random, targets=LADDER_TARGETS) -> list[tuple[tuple, dict]]:
    """One twin per target relative change.

    Depth: relative change of the *changed step's* value. Width: relative change of the
    answer. Candidates within a factor ``LADDER_TOLERANCE`` of the target are chosen
    uniformly (seeded), preferring candidates not already used for an earlier target;
    if none are within tolerance, the closest in ``|log(rel/t)|`` is taken
    (ties broken uniformly). Duplicates across targets are *kept* (the ladder always has
    one entry per target) and flagged via ``duplicate_of_target``.
    """
    cands = depth_twin_candidates(base) if task == "depth" else width_twin_candidates(base)
    if not cands:
        return []
    rels = [_ladder_rel(task, base, c) for c in cands]
    base_ans = _answer(base)
    used: dict[tuple, float] = {}
    twins = []
    for t in targets:
        pool = [i for i, r in enumerate(rels) if t / LADDER_TOLERANCE <= r <= t * LADDER_TOLERANCE]
        within = bool(pool)
        if pool:
            fresh = [i for i in pool if cands[i].problem.key() not in used]
            idx = rng.choice(fresh or pool)
        else:
            dist = [abs(math.log(r / t)) for r in rels]
            best = min(dist)
            idx = rng.choice([i for i, d in enumerate(dist) if d - best < 1e-12])
        c = cands[idx]
        new_ans = _answer(c.problem)
        key = c.problem.key()
        rec = render(c.problem)
        rec.update(
            target_rel_change=t,
            within_tolerance=within,
            step_changed=c.step_changed,
            operand_position=c.operand_position,
            old_operand=c.old_operand,
            new_operand=c.new_operand,
            rel_change_step=round(c.rel_change_step, 6),
            signed_change_step=c.new_step_value - c.old_step_value,
            abs_change_answer=abs(new_ans - base_ans),
            rel_change_answer=None if base_ans == 0 else round(abs(new_ans - base_ans) / base_ans, 6),
            duplicate_of_target=used.get(key),
        )
        used.setdefault(key, t)
        twins.append((key, rec))
    return twins


def depth_decoys(p: DepthProblem, rng: random.Random) -> list[tuple[tuple, dict]]:
    """Up to one ``commute`` and one ``same_value`` decoy (answer unchanged)."""
    out = []
    # commute: swap operands of a commutative (+ or *) step
    commute_steps = [
        k for k, op in enumerate(p.ops)
        if op in "+*" and (k > 0 or p.x0 != p.operands[0])
    ]
    if commute_steps:
        k = rng.choice(commute_steps)
        if k == 0:
            q = replace(p, x0=p.operands[0], operands=(p.x0,) + p.operands[1:])
        else:
            sw = list(p.swapped)
            sw[k] = not sw[k]
            q = replace(p, swapped=tuple(sw))
        rec = render(q)
        rec.update(decoy_type="commute", step_changed=k)
        out.append((q.key(), rec))
    # same_value: step 0 only, same op, different operand multiset, same result
    op, x, y = p.ops[0], p.x0, p.operands[0]
    v = apply_op(op, x, y)
    alts = [
        (x2, y2) for (op2, x2, y2) in _FIRST_CHOICES
        if op2 == op and apply_op(op, x2, y2) == v and sorted((x2, y2)) != sorted((x, y))
    ]
    if alts:
        x2, y2 = rng.choice(alts)
        q = replace(p, x0=x2, operands=(y2,) + p.operands[1:])
        rec = render(q)
        rec.update(decoy_type="same_value", step_changed=0)
        out.append((q.key(), rec))
    return out


def width_decoys(p: WidthProblem, rng: random.Random, n: int = 2) -> list[tuple[tuple, dict]]:
    """Up to ``n`` decoys, each editing one operand of a different non-max product so it
    stays strictly below the max (answer unchanged)."""
    i = p.argmax()
    m = max(p.products())
    by_product: dict[int, list] = {}
    for j, (a, b) in enumerate(p.pairs):
        if j == i:
            continue
        for pos in (0, 1):
            old = (a, b)[pos]
            for new in WIDTH_OPERANDS:
                if new == old:
                    continue
                pair = (new, b) if pos == 0 else (a, new)
                if pair[0] * pair[1] < m:
                    by_product.setdefault(j, []).append((pos, old, new, pair))
    js = sorted(by_product)
    rng.shuffle(js)
    out = []
    for j in js[:n]:
        pos, old, new, pair = rng.choice(by_product[j])
        pairs = list(p.pairs)
        pairs[j] = pair
        q = WidthProblem(tuple(pairs))
        rec = render(q)
        rec.update(
            decoy_type="smaller_product", step_changed=j,
            operand_position=pos, old_operand=old, new_operand=new,
        )
        out.append((q.key(), rec))
    return out
