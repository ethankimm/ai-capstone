"""Continuous-target metrics engine for probe evaluation: R^2, Pearson r, MAE
(+ normalized MAE), and a relaxed relative-error tolerance sweep.

Follow-up to the binary tol=0.01 hit-rate in `probe_codi.py`
(`20260920-032053_codi_probe-pilot`, ridge/MLP both at floor). A hard 1% threshold
collapses "no recoverable signal" and "signal exists but not to 1% precision" into the
same zero -- these continuous metrics don't collapse that way, per that run's own
"Next" note.

No torch/model dependency (pure numpy) so it runs locally on any cached (y_gold, y_hat)
pairs, independent of the GPU-side extraction that produces them.
"""
from __future__ import annotations

import numpy as np

DEFAULT_TOLERANCES = (0.01, 0.05, 0.10, 0.20)


def r_squared(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, dtype=np.float64), np.asarray(yhat, dtype=np.float64)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def pearson_r(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, dtype=np.float64), np.asarray(yhat, dtype=np.float64)
    if y.std() == 0.0 or yhat.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(y, yhat)[0, 1])


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, dtype=np.float64), np.asarray(yhat, dtype=np.float64)
    return float(np.mean(np.abs(y - yhat)))


def normalized_mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """MAE / std(y_gold); scale-free so it's comparable across (iteration, step) cells
    whose gold values span very different magnitudes."""
    y = np.asarray(y, dtype=np.float64)
    sigma = y.std()
    if sigma == 0.0:
        return float("nan")
    return mae(y, yhat) / sigma


def tolerance_hit_rate(y: np.ndarray, yhat: np.ndarray, tau: float) -> float:
    """Fraction of predictions within relative error `tau` of gold:
    |yhat - y| / (|y| + 1e-5) <= tau."""
    y, yhat = np.asarray(y, dtype=np.float64), np.asarray(yhat, dtype=np.float64)
    rel_err = np.abs(yhat - y) / (np.abs(y) + 1e-5)
    return float(np.mean(rel_err <= tau))


def compute_metrics(y: np.ndarray, yhat: np.ndarray, tolerances=DEFAULT_TOLERANCES) -> dict:
    """All metrics for one (iteration, step, predictor) cell. `y`/`yhat` must already be
    in real scale (un-transformed, e.g. signed-log1p inverted)."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    if y.shape != yhat.shape or y.size == 0:
        raise ValueError(f"y/yhat shape mismatch or empty: {y.shape} vs {yhat.shape}")
    return {
        "n": int(y.size),
        "r2": r_squared(y, yhat),
        "pearson_r": pearson_r(y, yhat),
        "mae": mae(y, yhat),
        "normalized_mae": normalized_mae(y, yhat),
        "tolerance_hit_rate": {f"{tau:g}": tolerance_hit_rate(y, yhat, tau) for tau in tolerances},
    }
