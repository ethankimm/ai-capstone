import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from continuous_metrics import compute_metrics, mae, normalized_mae, pearson_r, r_squared, tolerance_hit_rate


def test_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0)
    assert pearson_r(y, y) == pytest.approx(1.0)
    assert mae(y, y) == pytest.approx(0.0)
    assert normalized_mae(y, y) == pytest.approx(0.0)
    assert tolerance_hit_rate(y, y, 0.01) == pytest.approx(1.0)


def test_constant_predictions_are_worse_than_mean_baseline():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.full_like(y, y.mean())
    assert r_squared(y, yhat) == pytest.approx(0.0)
    assert not np.isnan(pearson_r(y, y))


def test_pearson_r_nan_on_constant_input():
    y = np.array([5.0, 5.0, 5.0])
    yhat = np.array([1.0, 2.0, 3.0])
    assert np.isnan(pearson_r(y, yhat))


def test_tolerance_sweep_is_monotonic_in_tau():
    y = np.array([100.0, 200.0, 300.0, 400.0])
    yhat = y * 1.08  # 8% relative error everywhere
    hits = {tau: tolerance_hit_rate(y, yhat, tau) for tau in (0.01, 0.05, 0.10, 0.20)}
    assert hits[0.01] == hits[0.05] == 0.0
    assert hits[0.10] == hits[0.20] == 1.0


def test_compute_metrics_shape_and_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yhat = y + np.array([0.1, -0.1, 0.2, -0.2, 0.0])
    m = compute_metrics(y, yhat)
    assert m["n"] == 5
    assert set(m["tolerance_hit_rate"]) == {"0.01", "0.05", "0.1", "0.2"}
    assert 0.0 <= m["r2"] <= 1.0


def test_compute_metrics_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0, 2.0]), np.array([1.0]))
