"""Metrics and bootstrap confidence intervals.

Transactions are not independent: the same card holder appears many times and
fraud comes in bursts. Resampling single rows then understates uncertainty, so
every bootstrap here can resample whole groups (days or entities) instead.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


class CI(NamedTuple):
    estimate: float
    low: float
    high: float


def _sort(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending order of scores, and the last position of each run of tied scores."""
    order = np.argsort(-score, kind="mergesort")
    s = score[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    return order, ends


def _ap(y_sorted: np.ndarray, w_sorted: np.ndarray, ends: np.ndarray) -> float:
    """AP = sum_n (R_n - R_{n-1}) * P_n over distinct score thresholds."""
    tp = np.cumsum(w_sorted * y_sorted)[ends]
    fp = np.cumsum(w_sorted * (1.0 - y_sorted))[ends]
    if tp[-1] == 0:
        return np.nan
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def average_precision(y, score, weight=None) -> float:
    """Weighted PR-AUC (average precision); matches sklearn's average_precision_score."""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    w = np.ones_like(y) if weight is None else np.asarray(weight, dtype=float)
    order, ends = _sort(score)
    return _ap(y[order], w[order], ends)


def _bootstrap_weights(groups, n: int, n_boot: int, rng: np.random.Generator):
    """Yield per-row weights: how many times each row's group was drawn.

    groups=None resamples single rows. Otherwise whole groups are drawn with
    replacement, which keeps the correlation inside a group intact.
    """
    # Missing group labels form one group of their own rather than code -1.
    codes = np.arange(n) if groups is None else pd.factorize(groups, use_na_sentinel=False)[0]
    n_groups = codes.max() + 1
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        yield counts[codes].astype(float)


def bootstrap_ci(
    y, score, groups=None, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0, weight=None
) -> CI:
    """Percentile bootstrap CI for PR-AUC; `weight` (e.g. amount) gives the dollar-weighted version."""
    y = np.asarray(y, dtype=float)
    base = np.ones_like(y) if weight is None else np.asarray(weight, dtype=float)
    order, ends = _sort(np.asarray(score, dtype=float))
    y_sorted, base_sorted = y[order], base[order]
    estimate = _ap(y_sorted, base_sorted, ends)
    rng = np.random.default_rng(seed)
    stats = [_ap(y_sorted, base_sorted * w[order], ends) for w in _bootstrap_weights(groups, len(y), n_boot, rng)]
    low, high = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return CI(estimate, float(low), float(high))


def paired_bootstrap_diff(
    y, score_a, score_b, groups=None, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0, weight=None
) -> CI:
    """CI for PR-AUC(b) - PR-AUC(a), evaluating both models on the same resamples.

    Pairing cancels the noise both models share (which transactions happen to
    be in the sample), so the CI of the difference is much narrower than two
    separate CIs would suggest. `weight` (e.g. amount) compares dollar-weighted PR-AUC.
    """
    y = np.asarray(y, dtype=float)
    base = np.ones_like(y) if weight is None else np.asarray(weight, dtype=float)
    order_a, ends_a = _sort(np.asarray(score_a, dtype=float))
    order_b, ends_b = _sort(np.asarray(score_b, dtype=float))
    ya, yb = y[order_a], y[order_b]

    def diff(w):
        w = base * w
        return _ap(yb, w[order_b], ends_b) - _ap(ya, w[order_a], ends_a)

    estimate = diff(np.ones_like(y))
    rng = np.random.default_rng(seed)
    stats = [diff(w) for w in _bootstrap_weights(groups, len(y), n_boot, rng)]
    low, high = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return CI(estimate, float(low), float(high))


def value_capture_at_rate(y, score, amount, rate: float) -> float:
    """Share of fraud dollars among the `rate` fraction of transactions with the highest scores."""
    y, score, amount = (np.asarray(a, dtype=float) for a in (y, score, amount))
    top = np.argsort(-score, kind="mergesort")[: int(np.ceil(rate * len(score)))]
    fraud_usd = amount * y
    return float(fraud_usd[top].sum() / fraud_usd.sum())


def summarize(y, score) -> dict:
    """Point metrics for one model on one dataset."""
    y = np.asarray(y)
    p = np.clip(np.asarray(score, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "n": len(y),
        "fraud_rate": float(y.mean()),
        "pr_auc": average_precision(y, p),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
    }
