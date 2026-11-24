"""Donor pairs for the depth2 rule chain: the H4 patching test and the portability (donor-distance) tiers of
COCONUT_CODI_PLAN.md (section 5, H4).

A *pair* = (recipient row, donor row, tier, k). The latents of rule-chain step k (Coconut: latent pair k, i.e. latents
2k-2 and 2k-1 for c_thought 2; k <= 3) are spliced from the donor's run into the recipient's run. Predictions:

* ``twin`` donor: the donor chain equals the recipient chain except rule k (one constant of its rule is resampled), so
  s_1..s_{k-1} agree and s_k differs, and the final answers differ. Rules k+1..n are shared, so the state-carrying
  prediction ``run_chain(donor s_k, recipient rules k+1..n)`` IS the donor's own final answer: a twin pass shows the slot is
  *used* but cannot tell "holds s_k" from "holds the final answer". Content claims therefore rest on the tiers below.
* ``same_level`` / ``other_level`` donors: an unrelated chain at the same / a different level. ``predicted_answer`` =
  ``run_chain(donor s_k, recipient rules k+1..n)`` (None -> the pair is dropped when the carried state would leave 0..99).
  These are the "different-sequence" donors; keep only pairs where ``predicted_answer`` differs from BOTH the recipient's
  answer and the donor's own answer (otherwise follows-donor / keeps-recipient / carries-state are not separable).

Pairs are deterministic given (rows, seed). Nothing here trains or evaluates a model.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from latentreasoning.tasks import synthetic_v2 as V

TIERS = ("twin", "same_level", "other_level")
MAX_SLOT_STEP = 3  # Coconut A: 6 latents = 3 pairs of 2, replacing steps 1-3 in the curriculum


def _resample_rule(rule: V.Rule, rng: random.Random) -> V.Rule:
    """Keep the rule kind; resample ONE constant within the generator's bounds."""
    if rule.kind == "P":
        return replace(rule, a=rng.randint(1, 25))
    if rule.kind == "T":
        which = rng.choice("tab")
        if which == "t":
            t = rng.randint(20, 80)
            return V.Rule("T", t=t, a=min(rule.a, 100 - t) or 1, b=min(rule.b, t) or 1)
        if which == "a":
            return replace(rule, a=rng.randint(1, min(20, 100 - rule.t)))
        return replace(rule, b=rng.randint(1, min(20, rule.t)))
    which = rng.choice("tb")
    if which == "t":
        t = rng.randint(10, 50)
        return V.Rule("D", t=t, b=min(rule.b, t) or 1)
    return replace(rule, b=rng.randint(1, rule.t))


def _states(chain: V.RuleChain) -> list[int] | None:
    walk = V.run_chain(chain.x0, chain.rules)
    return None if walk is None else [chain.x0] + [w[3] for w in walk]


def make_twin(chain: V.RuleChain, k: int, rng: random.Random, tries: int = 200) -> V.RuleChain | None:
    """Twin that differs from `chain` only in rule k (1-based), with s_k and the final answer changed."""
    base = _states(chain)
    if base is None or not 1 <= k <= len(chain.rules):
        return None
    for _ in range(tries):
        rules = list(chain.rules)
        rules[k - 1] = _resample_rule(rules[k - 1], rng)
        twin = V.RuleChain(chain.x0, tuple(rules))
        st = _states(twin)
        if st is None or st[k] == base[k] or st[-1] == base[-1]:
            continue
        if V.chain_accepted(twin, rng):
            return twin
    return None


def carried_answer(donor: V.RuleChain, recipient: V.RuleChain, k: int) -> int | None:
    """Final answer if the recipient continued from the donor's state s_k (rules k+1..n of the recipient)."""
    st = _states(donor)
    if st is None or k > len(donor.rules):
        return None
    walk = V.run_chain(st[k], recipient.rules[k:])
    return None if walk is None and recipient.rules[k:] else (walk[-1][3] if walk else st[k])


@dataclass
class Pair:
    recipient_id: str
    donor_id: str
    tier: str
    k: int
    donor: dict          # the donor row (question/steps/answer/level/id); twins are new items not in the split
    predicted_answer: str  # state-carrying prediction (see module docstring)
    donor_answer: str
    recipient_answer: str
    separable: bool       # predicted differs from both donor_answer and recipient_answer

    def to_json(self) -> dict:
        return self.__dict__.copy()


def _row_from_chain(chain: V.RuleChain, level: int, rid: str) -> dict:
    r = V.render_chain(chain)
    return {"question": r["question"], "steps": r["steps"], "answer": r["answer"], "level": level,
            "task": "depth2", "id": rid}


def build_pairs(rows: list[dict], seed: int = 0, ks=(1, 2, 3), per_item: int = 1, max_level: int | None = None) -> list[dict]:
    """For each recipient row, one pair per (tier, k) for k <= min(max step, level). Donors for same_level/other_level are
    drawn from `rows` (a different item), resampled up to 50 times until the carried state is valid."""
    rng = random.Random(seed)
    by_level: dict[int, list[dict]] = {}
    for r in rows:
        by_level.setdefault(int(r["level"]), []).append(r)
    chains = {r["id"]: V.parse_chain_question(r["question"]) for r in rows}
    out: list[dict] = []
    for r in rows:
        lvl = int(r["level"])
        if max_level is not None and lvl > max_level:
            continue
        rc = chains[r["id"]]
        for k in ks:
            if k > min(lvl, MAX_SLOT_STEP):
                continue
            for _ in range(per_item):
                tw = make_twin(rc, k, rng)
                if tw is not None:
                    drow = _row_from_chain(tw, lvl, f"{r['id']}-twin{k}")
                    out.append(Pair(r["id"], drow["id"], "twin", k, drow, drow["answer"], drow["answer"],
                                    r["answer"], drow["answer"] != r["answer"]).to_json())
                for tier in ("same_level", "other_level"):
                    for _try in range(50):
                        if tier == "same_level":
                            cand = rng.choice(by_level[lvl])
                        else:
                            other = [l for l in by_level if l != lvl and l >= k]
                            if not other:
                                break
                            cand = rng.choice(by_level[rng.choice(other)])
                        if cand["id"] == r["id"]:
                            continue
                        pred = carried_answer(chains[cand["id"]], rc, k)
                        if pred is None:
                            continue
                        sep = str(pred) != cand["answer"] and str(pred) != r["answer"]
                        out.append(Pair(r["id"], cand["id"], tier, k, cand, str(pred), cand["answer"], r["answer"], sep).to_json())
                        break
    return out
