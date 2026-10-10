"""Probability calibration and calibration diagnostics.

The cost rule in decision.py is only optimal if scores are probabilities, so
calibrators are fitted on the validation weeks and judged on the test weeks.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

Calibrator = Callable[[np.ndarray], np.ndarray]


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt_params(score, y) -> tuple[float, float]:
    """Platt scaling parameters (a, b): p = sigmoid(a * logit(score) + b)."""
    lr = LogisticRegression(C=1e6).fit(_logit(score).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt(score, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * _logit(score) + b)))


def fit_platt(score, y) -> Calibrator:
    """Platt scaling: a logistic regression on the score's log-odds (two parameters)."""
    a, b = fit_platt_params(score, y)
    return lambda s: platt(s, a, b)


def fit_isotonic(score, y) -> Calibrator:
    """Isotonic regression: the best monotone step function; flexible, needs more data."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(score, dtype=float), np.asarray(y))
    return lambda s: iso.predict(np.asarray(s, dtype=float))


def _bins(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:  # quantile: equal-count bins, needed when most scores sit near 0
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    return np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, len(edges) - 2)


def reliability_table(y, p, n_bins: int = 15, strategy: str = "quantile") -> pd.DataFrame:
    """Per bin: count, mean predicted probability, observed fraud rate."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    b = _bins(p, n_bins, strategy)
    return (
        pd.DataFrame({"bin": b, "p": p, "y": y})
        .groupby("bin")
        .agg(n=("y", "size"), mean_p=("p", "mean"), observed=("y", "mean"))
    )


def expected_calibration_error(y, p, n_bins: int = 15, strategy: str = "quantile") -> float:
    """Average |mean predicted - observed| over bins, weighted by bin size."""
    t = reliability_table(y, p, n_bins, strategy)
    return float(np.sum(t["n"] * (t["mean_p"] - t["observed"]).abs()) / t["n"].sum())


def brier_decomposition(y, p, n_bins: int = 15, strategy: str = "quantile") -> dict:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty.

    Reliability measures miscalibration (lower is better), resolution how much
    the forecasts separate cases from the base rate (higher is better),
    uncertainty is the base rate's own variance. The identity is exact when p
    is constant inside each bin; `residual` reports the binning error.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    t = reliability_table(y, p, n_bins, strategy)
    w = t["n"] / t["n"].sum()
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    reliability = float(np.sum(w * (t["mean_p"] - t["observed"]) ** 2))
    resolution = float(np.sum(w * (t["observed"] - base) ** 2))
    uncertainty = float(base * (1 - base))
    return {
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "residual": brier - (reliability - resolution + uncertainty),
    }
