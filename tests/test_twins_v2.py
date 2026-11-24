"""Depth2 donor pairs (H4 / portability tiers): twin invariants and state-carrying predictions."""
import json
import random
from pathlib import Path

import pytest

from latentreasoning.tasks import synthetic_v2 as V
from latentreasoning.tasks import twins_v2 as T

DATA = Path.home() / "Documents/Penn/CIS5980/synthetic_data_v2/depth2/valid.json"


def _rows(n=300, seed=3):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        lvl = rng.randint(1, 8)
        r = V.render_chain(V.sample_chain(rng, lvl))
        rows.append({**r, "level": lvl, "task": "depth2", "id": f"t{i}"})
    return rows


@pytest.fixture(scope="module")
def pairs_and_rows():
    rows = _rows()
    return T.build_pairs(rows, seed=1), {r["id"]: r for r in rows}


def test_twin_differs_only_in_rule_k_and_changes_state_and_answer(pairs_and_rows):
    pairs, rows = pairs_and_rows
    twins = [p for p in pairs if p["tier"] == "twin"]
    assert len(twins) > 100
    for p in twins:
        rc = V.parse_chain_question(rows[p["recipient_id"]]["question"])
        dc = V.parse_chain_question(p["donor"]["question"])
        k = p["k"]
        assert rc.x0 == dc.x0 and len(rc.rules) == len(dc.rules)
        diff = [i for i, (a, b) in enumerate(zip(rc.rules, dc.rules)) if a != b]
        assert diff == [k - 1]
        s_r, s_d = T._states(rc), T._states(dc)
        assert s_r[:k] == s_d[:k] and s_r[k] != s_d[k] and s_r[-1] != s_d[-1]
        assert p["predicted_answer"] == p["donor_answer"] == str(s_d[-1])  # shared tail -> carrying s_k == donor's answer


def test_cross_sequence_prediction_is_state_carrying(pairs_and_rows):
    pairs, rows = pairs_and_rows
    cross = [p for p in pairs if p["tier"] in ("same_level", "other_level")]
    assert len(cross) > 200
    for p in cross:
        rc = V.parse_chain_question(rows[p["recipient_id"]]["question"])
        dc = V.parse_chain_question(p["donor"]["question"])
        k = p["k"]
        s_d = T._states(dc)
        walk = V.run_chain(s_d[k], rc.rules[k:])
        want = walk[-1][3] if walk else s_d[k]
        assert p["predicted_answer"] == str(want)
        assert p["separable"] == (str(want) != p["donor_answer"] and str(want) != rows[p["recipient_id"]]["answer"])
        if p["tier"] == "other_level":
            assert int(p["donor"]["level"]) != int(rows[p["recipient_id"]]["level"])


def test_k_bounded_by_level_and_slot_limit(pairs_and_rows):
    pairs, rows = pairs_and_rows
    for p in pairs:
        assert p["k"] <= min(int(rows[p["recipient_id"]]["level"]), T.MAX_SLOT_STEP)


def test_deterministic():
    rows = _rows(80)
    assert T.build_pairs(rows, seed=5) == T.build_pairs(rows, seed=5)


@pytest.mark.skipif(not DATA.exists(), reason="v2 data not on this machine")
def test_real_valid_split_builds():
    rows = json.loads(DATA.read_text())[:200]
    pairs = T.build_pairs(rows, seed=0)
    assert {p["tier"] for p in pairs} == set(T.TIERS)
