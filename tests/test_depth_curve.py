import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from depth_curve import (  # noqa: E402
    curve_metrics, fit_sigmoid4, halfway_level, mcnemar_exact_p, newcombe_paired_ci, normalized_auc,
    paired_mcnemar, per_level, sigmoid4, wilson_ci,
)


def _records(accs: dict[int, float], n: int = 200) -> list[dict]:
    out = []
    for lvl, acc in accs.items():
        k = round(acc * n)
        out += [{"id": f"{lvl}-{i}", "level": lvl, "correct": i < k} for i in range(n)]
    return out


def test_fit_recovers_known_sigmoid():
    levels = np.arange(1, 13)
    y = sigmoid4(levels, floor=0.1, ceiling=0.95, midpoint=6.5, slope=1.2)
    fit = fit_sigmoid4(levels, y)
    assert fit["floor"] == pytest.approx(0.1, abs=0.01)
    assert fit["ceiling"] == pytest.approx(0.95, abs=0.01)
    assert fit["midpoint_param"] == pytest.approx(6.5, abs=0.05)
    assert fit["slope"] == pytest.approx(1.2, rel=0.05)
    assert fit["rmse"] < 1e-3


def test_fit_canonicalizes_increasing_curve():
    levels = np.arange(1, 13)
    y = sigmoid4(levels, floor=0.9, ceiling=0.2, midpoint=5.0, slope=1.0)  # rising with level
    fit = fit_sigmoid4(levels, y)
    assert fit["slope"] >= 0
    assert fit["floor"] == pytest.approx(0.9, abs=0.01)  # floor = high-level asymptote
    assert fit["ceiling"] == pytest.approx(0.2, abs=0.01)


def test_midpoint_halfway_between_floor_and_best_observed():
    levels = np.arange(1, 13)
    y = sigmoid4(levels, 0.0, 1.0, 6.0, 1.5)
    fit = fit_sigmoid4(levels, y)
    best = float(y.max())
    mid, status, _ = halfway_level(fit, best, 1, 12)
    assert status == "in_range"
    # analytic: level where curve == best/2
    expected = 6.0 + math.log(1 / (best / 2) - 1) / 1.5
    assert mid == pytest.approx(expected, abs=0.05)


def test_midpoint_beyond_range_when_curve_never_drops():
    levels = np.arange(1, 13)
    y = sigmoid4(levels, 0.0, 1.0, 20.0, 1.0)  # collapse happens past level 12
    m = curve_metrics(_records(dict(zip(levels.tolist(), y.tolist())), n=1000))
    assert m["midpoint"] is None
    assert m["midpoint_status"] == "beyond_range"


def test_flat_curve_is_beyond_range():
    m = curve_metrics(_records({l: 0.5 for l in range(1, 9)}))
    assert m["midpoint"] is None and m["midpoint_status"] == "beyond_range"


def test_curve_metrics_end_to_end_noisy():
    rng = np.random.default_rng(0)
    recs = []
    for lvl in range(1, 13):
        p = float(sigmoid4(lvl, 0.05, 0.98, 7.0, 1.0))
        recs += [{"id": f"{lvl}-{i}", "level": lvl, "correct": bool(rng.random() < p)} for i in range(500)]
    m = curve_metrics(recs)
    assert m["midpoint_status"] == "in_range"
    assert 6.0 < m["midpoint"] < 8.0
    assert set(m["per_level_accuracy"]) == {str(l) for l in range(1, 13)}
    assert all(0 <= lo <= hi <= 1 for lo, hi in m["per_level_ci95"].values())


def test_auc_normalization():
    assert normalized_auc([1, 2, 3], [1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert normalized_auc([1, 2, 3], [0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert normalized_auc([1, 3], [1.0, 0.0]) == pytest.approx(0.5)
    assert normalized_auc([3, 1, 2], [0.0, 1.0, 0.5]) == pytest.approx(0.5)  # unsorted input
    assert normalized_auc([4], [0.3]) == pytest.approx(0.3)


def test_per_level_and_wilson():
    rows = per_level(_records({1: 0.5, 2: 1.0}, n=10))
    assert [r["level"] for r in rows] == [1, 2]
    assert rows[0]["accuracy"] == 0.5 and rows[1]["accuracy"] == 1.0
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0)


def test_mcnemar_exact_p():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 5) == 1.0
    # b=0, c=6: 2 * 0.5**6
    assert mcnemar_exact_p(0, 6) == pytest.approx(2 / 64)
    # b=2, c=10: 2 * sum_{i<=2} C(12,i) / 2^12
    assert mcnemar_exact_p(2, 10) == pytest.approx(2 * (1 + 12 + 66) / 4096)


def test_newcombe_paired_ci_properties():
    lo, hi = newcombe_paired_ci(8, 5, 2, 5)
    assert lo < 0.15 < hi and -1 <= lo and hi <= 1
    # swapping A and B mirrors the interval
    lo2, hi2 = newcombe_paired_ci(8, 2, 5, 5)
    assert lo2 == pytest.approx(-hi) and hi2 == pytest.approx(-lo)
    # no discordant pairs: point estimate 0, interval stays non-degenerate (Newcombe is conservative here)
    lo3, hi3 = newcombe_paired_ci(50, 0, 0, 0)
    assert lo3 < 0 < hi3 and lo3 == pytest.approx(-hi3)


def test_newcombe_matches_wald_for_large_n():
    a, b, c, d = 1200, 300, 200, 300
    n = a + b + c + d
    diff = (b - c) / n
    se = math.sqrt(((b + c) / n - diff ** 2) / n)
    lo, hi = newcombe_paired_ci(a, b, c, d)
    assert lo == pytest.approx(diff - 1.96 * se, abs=2e-3)
    assert hi == pytest.approx(diff + 1.96 * se, abs=2e-3)


def test_paired_mcnemar_pairs_by_id_and_filters_level():
    a = [{"id": f"x{i}", "level": 1 + i % 10, "correct": True} for i in range(100)]
    b = [{"id": f"x{i}", "level": 1 + i % 10, "correct": (i // 10) % 2 == 0} for i in range(100)]
    b.append({"id": "extra", "level": 9, "correct": True})
    res = paired_mcnemar(a, b, min_level=6)
    assert res["n_pairs"] == 50  # levels 6..10
    assert res["n_only_in_b"] == 1
    assert res["acc_a"] == 1.0 and res["acc_b"] == 0.5
    assert res["diff"] == pytest.approx(0.5)
    assert res["a_only"] == 25 and res["b_only"] == 0
    assert res["diff_ci95"][0] < 0.5 < res["diff_ci95"][1]
    assert res["mcnemar_exact_p"] == pytest.approx(2 * 0.5 ** 25)
