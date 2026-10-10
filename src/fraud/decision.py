"""Decision policies and their cost.

Actions per transaction: APPROVE, REVIEW (sent to a human), DECLINE.
Costs follow configs/costs.yaml; every number there is an assumption.

  approved fraud     -> amount_share * amount + chargeback_fee
  declined legit     -> fixed + margin_share * amount
  reviewed           -> cost_per_case, plus the fraud cost if the reviewer misses it
                        (reviewers stop a `catch_rate` share of frauds and never decline good customers)
  correct decisions  -> 0
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraud.config import load_config
from fraud.evaluate import CI, _bootstrap_weights

APPROVE, REVIEW, DECLINE = 0, 1, 2


def load_costs(**overrides) -> dict:
    """Read costs.yaml; overrides like fp_fixed=10 or margin_share=0.05 replace single values."""
    costs = copy.deepcopy(load_config("costs"))
    keys = {
        "fp_fixed": ("false_positive", "fixed"),
        "margin_share": ("false_positive", "margin_share"),
        "chargeback_fee": ("missed_fraud", "chargeback_fee"),
    }
    for name, value in overrides.items():
        section, key = keys[name]
        costs[section][key] = value
    return costs


def cost_fn(amount, costs: dict) -> np.ndarray:
    """Cost of approving a fraud."""
    c = costs["missed_fraud"]
    return c["amount_share"] * np.asarray(amount, dtype=float) + c["chargeback_fee"]


def cost_fp(amount, costs: dict) -> np.ndarray:
    """Cost of declining a legitimate customer."""
    c = costs["false_positive"]
    return c["fixed"] + c["margin_share"] * np.asarray(amount, dtype=float)


def threshold_policy(score, threshold: float) -> np.ndarray:
    return np.where(np.asarray(score) >= threshold, DECLINE, APPROVE)


def top_rate_threshold(score, rate: float) -> float:
    """Score threshold that declines a `rate` share of the given transactions."""
    return float(np.quantile(np.asarray(score, dtype=float), 1 - rate))


def best_f1_threshold(y, score) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def bayes_policy(p, amount, costs: dict, fp_penalty: float = 0.0) -> np.ndarray:
    """Decline when the expected cost of approving exceeds that of declining: p*C_FN > (1-p)*(C_FP + penalty).

    Equivalent to a per-transaction threshold p* = C_FP / (C_FP + C_FN), which
    falls as the amount grows. `fp_penalty` is a Lagrange multiplier on
    declined good customers, used to enforce a cap on their share.
    """
    p = np.asarray(p, dtype=float)
    return np.where(p * cost_fn(amount, costs) > (1 - p) * (cost_fp(amount, costs) + fp_penalty), DECLINE, APPROVE)


def three_action_policy(p, amount, day, costs: dict, fp_penalty: float = 0.0) -> np.ndarray:
    """Approve, review or decline by lowest expected cost, with a daily review capacity.

    When a day has more review candidates than capacity, the ones with the
    largest expected saving from review are kept; the others fall back to the
    better of approve and decline. `fp_penalty` works as in bayes_policy.
    """
    p = np.asarray(p, dtype=float)
    review = costs["manual_review"]
    expected = {
        APPROVE: p * cost_fn(amount, costs),
        DECLINE: (1 - p) * (cost_fp(amount, costs) + fp_penalty),
    }
    fallback = np.where(expected[DECLINE] < expected[APPROVE], DECLINE, APPROVE)
    best_other = np.minimum(expected[APPROVE], expected[DECLINE])
    review_cost = review["cost_per_case"] + (1 - review["catch_rate"]) * p * cost_fn(amount, costs)
    saving = best_other - review_cost

    action = fallback.copy()
    candidates = pd.DataFrame({"day": np.asarray(day), "saving": saving})
    candidates = candidates[candidates["saving"] > 0]
    chosen = (
        candidates.sort_values("saving", ascending=False)
        .groupby("day")
        .head(review["daily_capacity"])
        .index
    )
    action[chosen] = REVIEW
    return action


def realized_cost(action, y, amount, costs: dict) -> np.ndarray:
    """Cost of each decision once the true label is known (review outcomes in expectation)."""
    action, y = np.asarray(action), np.asarray(y, dtype=float)
    fn, fp = cost_fn(amount, costs), cost_fp(amount, costs)
    review = costs["manual_review"]
    return np.select(
        [action == APPROVE, action == DECLINE, action == REVIEW],
        [y * fn, (1 - y) * fp, review["cost_per_case"] + y * (1 - review["catch_rate"]) * fn],
    )


def policy_summary(action, y, amount, costs: dict) -> dict:
    action, y = np.asarray(action), np.asarray(y, dtype=float)
    amount = np.asarray(amount, dtype=float)
    cost = realized_cost(action, y, amount, costs)
    fraud_usd = amount * y
    caught = np.where(action == DECLINE, 1.0, np.where(action == REVIEW, costs["manual_review"]["catch_rate"], 0.0))
    return {
        "total_cost": float(cost.sum()),
        "fraud_loss": float(np.sum(y * (1 - caught) * cost_fn(amount, costs))),
        "declined_share": float((action == DECLINE).mean()),
        "reviewed_share": float((action == REVIEW).mean()),
        "good_customers_declined": float(((action == DECLINE) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "fraud_usd_stopped": float((caught * fraud_usd).sum() / fraud_usd.sum()),
    }


def good_decline_rate(action, y) -> float:
    """Share of legitimate transactions that were declined."""
    action, y = np.asarray(action), np.asarray(y)
    return float(((action == DECLINE) & (y == 0)).sum() / max((y == 0).sum(), 1))


def penalty_for_cap(policy, y, cap: float, tol: float = 1e-3, max_penalty: float = 1e7) -> float:
    """Smallest Lagrange penalty λ >= 0 with good-customer decline rate <= cap.

    `policy(λ)` returns actions; the decline rate falls as λ grows, so λ is
    found by bisection (on a log scale, since λ can range over decades).
    """
    if good_decline_rate(policy(0.0), y) <= cap:
        return 0.0
    low, high = 1e-3, 1.0
    while good_decline_rate(policy(high), y) > cap:
        low, high = high, high * 4
        if high > max_penalty:
            raise ValueError("cap not reachable")
    while high / low > 1 + tol:
        mid = np.sqrt(low * high)
        low, high = (mid, high) if good_decline_rate(policy(mid), y) > cap else (low, mid)
    return float(high)


def threshold_for_cap(score, y, cap: float) -> float:
    """Lowest score threshold whose good-customer decline rate is <= cap."""
    legit = np.sort(np.asarray(score, dtype=float)[np.asarray(y) == 0])
    k = int(np.floor(cap * len(legit)))  # legitimate rows allowed above the threshold
    return float(np.nextafter(legit[len(legit) - k - 1], np.inf)) if k < len(legit) else float(legit[0])


def paired_cost_diff(cost_a, cost_b, groups=None, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> CI:
    """CI for total cost(b) - total cost(a) on the same resampled card holders."""
    diff = np.asarray(cost_b, dtype=float) - np.asarray(cost_a, dtype=float)
    rng = np.random.default_rng(seed)
    stats = [float(np.sum(w * diff)) for w in _bootstrap_weights(groups, len(diff), n_boot, rng)]
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return CI(float(diff.sum()), float(low), float(high))
