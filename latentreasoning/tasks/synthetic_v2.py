"""Synthetic tasks v2 (generator version 2.0.0). See ``V2_TASKS_SPEC.md`` (draft 2).

v1 depth was affine (``+ - *``, no wrap) so nominal depth overstated serial depth ~3x, and v1
width was solved at 0.997 by everything. v2:

* **depth2** (called **RuleChain** in the paper; ``depth2`` stays the task id in data paths, hash keys,
  run names and CODI's ``--data_name``) -- a *rule chain* (modelled on Nanda's nocot-bench ``chain`` bank): a start value
  and ``n`` data-dependent rules, each applied once to the running value ``s`` in 0..99::

      Start with 57. Rule 1: if it is even, halve it; otherwise add 5.
      Rule 2: if it is at least 40, subtract 12; otherwise add 7.
      Rule 3: if it is below 30, double it; otherwise subtract 9. What is the final number?

  Each CoT step shows the arithmetic actually applied (``<<57+5=62>>``), so the branch decision is
  the hidden part. Rules P (parity), T (threshold), D (double) have item-specific constants.
  Steps are ``<<L op R=V>>`` with op in ``+ - * /`` (``/`` only for exact halving).
* **paircount** -- N numbers mod 50 and a target T; how many pairs ``i<j`` sum to T mod 50?
  CoT = one running-count step per element, ``<<c+m=c'>>`` (m = earlier partners of element i).
* **distinct** -- N numbers in 0..99; how many distinct values? CoT = running distinct count.
  (Pilot: solved to 1.000 by no-CoT GPT-2 within 2 epochs -- pure exact-match, no depth.)
* **modal** -- N small expressions (``a+b``, ``a-b``, ``a*b``, every value in 0..99); which value do
  most of them evaluate to? Two parallel stages (evaluate each expression, then tally across
  positions), modelled on nocot-bench's ``modes`` bank. CoT = N evaluation steps ``<<a*b=v>>`` then a
  final ``<<m+0=m>>`` answer step (so the last step still equals the answer, as CODI requires).

Output records match v1: ``{id, task, level, question, steps, answer(str)}`` (+ ``nudge_depth``
for depth2). ``level`` is the nominal rule count (depth2) or N (width families). Twins and decoys
are deliberately not built (no consumer yet; v1's injectivity guarantee is gone).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from latentreasoning.tasks.generate import _rng, _split_quota, dumps_records

GENERATOR_VERSION = "2.0.0"
DEFAULT_SEED = 5980
VALUE_MAX = 99
DEPTH_LEVELS = list(range(1, 13))
PAIRCOUNT_LEVELS = list(range(6, 17, 2))
PAIRCOUNT_MOD = 50
DISTINCT_LEVELS = list(range(8, 25, 4))
NUDGE_DELTAS = (-2, -1, 1, 2)
ORDER_BINDING_PERMS = 8
ATTRACTOR_FRAC = 0.4
ATTRACTOR_MIN_STEPS = 5

MODAL_LEVELS = list(range(6, 17, 2))

TASK_LEVELS = {"depth2": DEPTH_LEVELS, "paircount": PAIRCOUNT_LEVELS, "distinct": DISTINCT_LEVELS,
               "modal": MODAL_LEVELS}


# ---------------------------------------------------------------------------------------
# depth2: rule chain
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    kind: str  # "P" parity, "T" threshold, "D" double
    t: int = 0
    a: int = 0
    b: int = 0

    def text(self) -> str:
        if self.kind == "P":
            return f"if it is even, halve it; otherwise add {self.a}"
        if self.kind == "T":
            return f"if it is at least {self.t}, subtract {self.b}; otherwise add {self.a}"
        return f"if it is below {self.t}, double it; otherwise subtract {self.b}"

    def step(self, s: int) -> tuple[int, str, int, int] | None:
        """Return (left, op, right, value), or None if the result leaves 0..99."""
        if self.kind == "P":
            out = (s, "/", 2, s // 2) if s % 2 == 0 else (s, "+", self.a, s + self.a)
        elif self.kind == "T":
            out = (s, "-", self.b, s - self.b) if s >= self.t else (s, "+", self.a, s + self.a)
        else:
            out = (s, "*", 2, 2 * s) if s < self.t else (s, "-", self.b, s - self.b)
        return out if 0 <= out[3] <= VALUE_MAX else None


_RULE_RES = {
    "P": re.compile(r"^if it is even, halve it; otherwise add (\d+)$"),
    "T": re.compile(r"^if it is at least (\d+), subtract (\d+); otherwise add (\d+)$"),
    "D": re.compile(r"^if it is below (\d+), double it; otherwise subtract (\d+)$"),
}


def sample_rule(rng: random.Random) -> Rule:
    kind = rng.choice("PTD")
    if kind == "P":
        return Rule("P", a=rng.randint(1, 25))
    if kind == "T":
        t = rng.randint(20, 80)
        return Rule("T", t=t, a=rng.randint(1, min(20, 100 - t)), b=rng.randint(1, min(20, t)))
    t = rng.randint(10, 50)
    return Rule("D", t=t, b=rng.randint(1, t))


@dataclass(frozen=True)
class RuleChain:
    x0: int
    rules: tuple[Rule, ...]

    @property
    def n_steps(self) -> int:
        return len(self.rules)

    def key(self) -> tuple:
        return ("depth2", self.x0, self.rules)


def run_chain(x0: int, rules) -> list[tuple[int, str, int, int]] | None:
    """Step records for the walk, or None if any value leaves 0..99."""
    s, out = x0, []
    for r in rules:
        st = r.step(s)
        if st is None:
            return None
        out.append(st)
        s = st[3]
    return out


def _final(x0: int, rules) -> int | None:
    walk = run_chain(x0, rules)
    return walk[-1][3] if walk else None


def chain_accepted(c: RuleChain, rng: random.Random) -> bool:
    walk = run_chain(c.x0, c.rules)
    if walk is None:
        return False
    vals = [w[3] for w in walk]
    n = len(vals)
    if vals[-1] == c.x0:
        return False
    if n >= ATTRACTOR_MIN_STEPS and max(Counter(vals).values()) / n > ATTRACTOR_FRAC:
        return False
    if n >= 2:  # order-binding: reordering the rules must change the answer
        for _ in range(ORDER_BINDING_PERMS):
            perm = list(c.rules)
            rng.shuffle(perm)
            if tuple(perm) != c.rules and _final(c.x0, perm) == vals[-1]:
                return False
    return True


def sample_chain(rng: random.Random, n_steps: int) -> RuleChain:
    while True:
        c = RuleChain(rng.randint(1, VALUE_MAX), tuple(sample_rule(rng) for _ in range(n_steps)))
        if chain_accepted(c, rng):
            return c


def nudge_depth(c: RuleChain) -> int:
    """Integrity check, not a difficulty knob (the measure saturates; see the spec).

    Perturb the state entering rule k+1 (s_0 is the start) by each delta in NUDGE_DELTAS that
    keeps the walk valid; position k "counts" if a majority of valid deltas change the final
    answer. Returns the length of the longest run of counted positions ending at the last rule
    (so the maximum is n).
    """
    walk = run_chain(c.x0, c.rules)
    states = [c.x0] + [w[3] for w in walk]
    final, n = states[-1], len(c.rules)
    depth = 0
    for k in range(n - 1, -1, -1):
        changed = valid = 0
        for d in NUDGE_DELTAS:
            s = states[k] + d
            if not 0 <= s <= VALUE_MAX:
                continue
            w = run_chain(s, c.rules[k:])
            if w is None:
                continue
            valid += 1
            changed += w[-1][3] != final
        if valid and changed * 2 > valid:
            depth += 1
        else:
            break
    return depth


def _fmt_step(left, op, right, v) -> str:
    return f"<<{left}{op}{right}={v}>>"


def render_chain(c: RuleChain) -> dict:
    walk = run_chain(c.x0, c.rules)
    rules = " ".join(f"Rule {i + 1}: {r.text()}." for i, r in enumerate(c.rules))
    return {
        "question": f"Start with {c.x0}. {rules} What is the final number?",
        "steps": [_fmt_step(*w) for w in walk],
        "answer": str(walk[-1][3]),
        "nudge_depth": nudge_depth(c),
    }


_Q_RE = re.compile(r"^Start with (\d+)\. (.*) What is the final number\?$")


def parse_chain_question(q: str) -> RuleChain:
    m = _Q_RE.match(q)
    if not m:
        raise ValueError(f"not a depth2 question: {q!r}")
    parts = re.split(r" (?=Rule \d+: )", m.group(2))
    rules = []
    for i, part in enumerate(parts):
        pm = re.match(rf"^Rule {i + 1}: (.*)\.$", part)
        if not pm:
            raise ValueError(f"bad rule {i + 1}: {part!r}")
        body = pm.group(1)
        for kind, rx in _RULE_RES.items():
            mm = rx.match(body)
            if mm:
                g = [int(x) for x in mm.groups()]
                rules.append(
                    Rule("P", a=g[0]) if kind == "P"
                    else Rule("T", t=g[0], b=g[1], a=g[2]) if kind == "T"
                    else Rule("D", t=g[0], b=g[1])
                )
                break
        else:
            raise ValueError(f"unknown rule: {body!r}")
    return RuleChain(int(m.group(1)), tuple(rules))


# ---------------------------------------------------------------------------------------
# width families
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairCount:
    nums: tuple[int, ...]
    target: int

    def key(self) -> tuple:
        return ("paircount", self.nums, self.target)

    def partners(self) -> list[int]:
        """m_i = number of earlier elements j < i with nums[j] + nums[i] = target (mod 50)."""
        return [sum((self.nums[j] + x) % PAIRCOUNT_MOD == self.target for j in range(i))
                for i, x in enumerate(self.nums)]

    def answer(self) -> int:
        return sum(self.partners())


def sample_paircount(rng: random.Random, n: int) -> PairCount:
    k = rng.randint(0, min(4, n // 2))
    while True:
        target = rng.randrange(PAIRCOUNT_MOD)
        nums = [rng.randrange(PAIRCOUNT_MOD) for _ in range(n)]
        p = PairCount(tuple(nums), target)
        if p.answer() == k:
            return p


def render_paircount(p: PairCount) -> dict:
    ms, c, steps = p.partners(), 0, []
    for m in ms:
        steps.append(_fmt_step(c, "+", m, c + m))
        c += m
    nums = ", ".join(map(str, p.nums))
    return {
        "question": (f"Numbers: {nums}. Target: {p.target} (mod {PAIRCOUNT_MOD}). "
                     "How many pairs of numbers add up to the target?"),
        "steps": steps,
        "answer": str(c),
    }


_PC_RE = re.compile(r"^Numbers: ([\d, ]+)\. Target: (\d+) \(mod 50\)\. How many pairs of numbers add up to the target\?$")


def parse_paircount_question(q: str) -> PairCount:
    m = _PC_RE.match(q)
    if not m:
        raise ValueError(f"not a paircount question: {q!r}")
    return PairCount(tuple(int(x) for x in m.group(1).split(", ")), int(m.group(2)))


@dataclass(frozen=True)
class Distinct:
    nums: tuple[int, ...]

    def key(self) -> tuple:
        return ("distinct", self.nums)

    def answer(self) -> int:
        return len(set(self.nums))


def sample_distinct(rng: random.Random, n: int) -> Distinct:
    d = rng.randint(1, n)
    vals = rng.sample(range(VALUE_MAX + 1), d)
    nums = vals + [rng.choice(vals) for _ in range(n - d)]
    rng.shuffle(nums)
    return Distinct(tuple(nums))


def render_distinct(p: Distinct) -> dict:
    seen, c, steps = set(), 0, []
    for x in p.nums:
        new = x not in seen
        seen.add(x)
        steps.append(_fmt_step(c, "+", int(new), c + int(new)))
        c += int(new)
    return {
        "question": f"Numbers: {', '.join(map(str, p.nums))}. How many different numbers are there?",
        "steps": steps,
        "answer": str(c),
    }


_D_RE = re.compile(r"^Numbers: ([\d, ]+)\. How many different numbers are there\?$")


def parse_distinct_question(q: str) -> Distinct:
    m = _D_RE.match(q)
    if not m:
        raise ValueError(f"not a distinct question: {q!r}")
    return Distinct(tuple(int(x) for x in m.group(1).split(", ")))


@dataclass(frozen=True)
class Modal:
    exprs: tuple[tuple[int, str, int], ...]  # (left, op, right), op in + - *

    def key(self) -> tuple:
        return ("modal", self.exprs)

    def values(self) -> list[int]:
        return [apply_op(op, a, b) for a, op, b in self.exprs]

    def counts(self) -> Counter:
        return Counter(self.values())

    def answer(self) -> int:
        ranked = self.counts().most_common(2)
        assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "mode is not unique"
        return ranked[0][0]


def _expr_for(rng: random.Random, t: int) -> tuple[int, str, int]:
    """A random expression ``a op b`` (operands 0..99) that evaluates to exactly t."""
    ops = ["+", "-"]
    factors = [(a, t // a) for a in range(2, 10) if t >= 4 and t % a == 0 and t // a >= 2]
    if factors:
        ops.append("*")
    op = rng.choice(ops)
    if op == "+":
        a = rng.randint(0, t)
        return (a, "+", t - a)
    if op == "-":
        b = rng.randint(0, VALUE_MAX - t)
        return (t + b, "-", b)
    a, b = rng.choice(factors)
    return (a, "*", b) if rng.random() < 0.5 else (b, "*", a)


def sample_modal(rng: random.Random, n: int) -> Modal:
    """Unique mode with count m in 2..2+n//6; every other value appears at most m-1 times."""
    m = rng.randint(2, 2 + n // 6)
    values = rng.sample(range(VALUE_MAX + 1), n)  # more than enough distinct values
    counts = [m]
    remaining, i = n - m, 1
    while remaining > 0:
        c = min(rng.randint(1, m - 1), remaining)
        counts.append(c)
        remaining -= c
        i += 1
    exprs: list[tuple[int, str, int]] = []
    for v, c in zip(values, counts):
        seen: set = set()
        while len(seen) < c:
            seen.add(_expr_for(rng, v))
        exprs += sorted(seen)
    rng.shuffle(exprs)
    return Modal(tuple(exprs))


def render_modal(p: Modal) -> dict:
    steps = [_fmt_step(a, op, b, apply_op(op, a, b)) for a, op, b in p.exprs]
    m = p.answer()
    steps.append(_fmt_step(m, "+", 0, m))
    return {
        "question": (f"Expressions: {', '.join(f'{a}{op}{b}' for a, op, b in p.exprs)}. "
                     "Which value do the most expressions evaluate to?"),
        "steps": steps,
        "answer": str(m),
    }


_M_RE = re.compile(r"^Expressions: ([\d+\-*, ]+)\. Which value do the most expressions evaluate to\?$")
_EXPR_RE = re.compile(r"^(\d+)([+\-*])(\d+)$")


def parse_modal_question(q: str) -> Modal:
    m = _M_RE.match(q)
    if not m:
        raise ValueError(f"not a modal question: {q!r}")
    exprs = []
    for part in m.group(1).split(", "):
        em = _EXPR_RE.match(part)
        if not em:
            raise ValueError(f"bad expression: {part!r}")
        exprs.append((int(em.group(1)), em.group(2), int(em.group(3))))
    return Modal(tuple(exprs))


_STEP_RE = re.compile(r"^<<(\d+)([+\-*/])(\d+)=(\d+)>>$")


def parse_step(step: str) -> tuple[int, str, int, int]:
    m = _STEP_RE.match(step)
    if not m:
        raise ValueError(f"bad step: {step!r}")
    return int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))


def apply_op(op: str, x: int, y: int) -> int:
    if op == "/":
        if y == 0 or x % y:
            raise ValueError("inexact division")
        return x // y
    return {"+": x + y, "-": x - y, "*": x * y}[op]


_SAMPLERS = {"depth2": sample_chain, "paircount": sample_paircount, "distinct": sample_distinct,
             "modal": sample_modal}
_RENDERERS = {"depth2": render_chain, "paircount": render_paircount, "distinct": render_distinct,
              "modal": render_modal}
_PARSERS = {"depth2": parse_chain_question, "paircount": parse_paircount_question,
            "distinct": parse_distinct_question, "modal": parse_modal_question}


def render(task: str, problem) -> dict:
    return _RENDERERS[task](problem)


def parse_question(task: str, q: str):
    return _PARSERS[task](q)


# ---------------------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------------------


def build_task(
    task: str,
    seed: int = DEFAULT_SEED,
    train_total: int = 100_000,
    valid_total: int = 2_000,
    test_per_level: int = 500,
    monitor_per_level: int = 30,
    levels: list[int] | None = None,
    train_quota: dict[int, int] | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Return ({split: records}, stats). Pure function of its arguments.

    `train_quota` ({level: n}) overrides the even split of `train_total` (for up-weighting levels). Test and valid are
    drawn before train from each level's own RNG stream, so they do not depend on the train quota.

    Every level is sampled (the content spaces are far larger than the quotas); the splits
    are made disjoint by rejecting canonical keys already drawn (test, then valid, then train).
    """
    levels = levels or TASK_LEVELS[task]
    train_q = dict(train_quota) if train_quota else _split_quota(train_total, levels)
    valid_q = _split_quota(valid_total, levels)
    splits: dict[str, list[dict]] = {"train": [], "valid": [], "valid_monitor": [], "test": []}
    level_stats: dict[str, dict] = {}

    for level in levels:
        rng = _rng(seed, task, level, "sample")
        seen: set = set()

        def draw(n):
            got = []
            while len(got) < n:
                p = _SAMPLERS[task](rng, level)
                if p.key() not in seen:
                    seen.add(p.key())
                    got.append(p)
            return got

        parts = {"test": draw(test_per_level), "valid": draw(valid_q[level]), "train": draw(train_q[level])}
        recs = {}
        for split, probs in parts.items():
            recs[split] = []
            for i, p in enumerate(probs):
                rec = render(task, p)
                rec.update(id=f"{task}-{split}-L{level:02d}-{i:05d}", task=task, level=level)
                recs[split].append(rec)
        splits["train"] += recs["train"]
        splits["valid"] += recs["valid"]
        splits["valid_monitor"] += recs["valid"][:monitor_per_level]
        splits["test"] += recs["test"]
        level_stats[str(level)] = _level_stats(task, parts["test"], recs["test"])

    _rng(seed, task, "train-order").shuffle(splits["train"])
    stats = {
        "task": task,
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "counts": {k: len(v) for k, v in splits.items()},
        "levels": level_stats,
    }
    return splits, stats


def _level_stats(task: str, problems, recs) -> dict:
    answers = Counter(r["answer"] for r in recs)
    out = {"n_test": len(recs), "best_constant_guess": round(max(answers.values()) / len(recs), 4)}
    if task == "depth2":
        nd = [r["nudge_depth"] for r in recs]
        n = problems[0].n_steps
        out["rule_mix"] = dict(Counter(r.kind for p in problems for r in p.rules))
        out["nudge_depth_mean"] = round(sum(nd) / len(nd), 3)
        out["nudge_depth_below_n_minus_1"] = round(sum(d < n - 1 for d in nd) / len(nd), 4)
    return out


def write_task(out_dir: Path, task: str, **kwargs) -> dict:
    splits, stats = build_task(task, **kwargs)
    d = Path(out_dir) / task
    d.mkdir(parents=True, exist_ok=True)
    sha = {}
    for name, recs in splits.items():
        text = dumps_records(recs)
        (d / f"{name}.json").write_text(text)
        sha[name] = hashlib.sha256(text.encode()).hexdigest()
    stats["sha256"] = sha
    (d / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True))
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=[*TASK_LEVELS, "all"], default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--train-total", type=int, default=100_000)
    ap.add_argument("--valid-total", type=int, default=2_000)
    ap.add_argument("--test-per-level", type=int, default=500)
    ap.add_argument("--monitor-per-level", type=int, default=30)
    a = ap.parse_args(argv)
    for task in (TASK_LEVELS if a.task == "all" else [a.task]):
        stats = write_task(Path(a.out).expanduser(), task, seed=a.seed, train_total=a.train_total,
                           valid_total=a.valid_total, test_per_level=a.test_per_level,
                           monitor_per_level=a.monitor_per_level)
        print(task, stats["counts"])


if __name__ == "__main__":
    main()
