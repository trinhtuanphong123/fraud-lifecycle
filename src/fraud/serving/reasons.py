"""Reason codes from TreeSHAP contributions.

LightGBM computes exact TreeSHAP values natively (`pred_contrib=True`): one
additive contribution per feature in log-odds, plus a bias, summing to the raw
score's log-odds. The features pushing the risk up the most become the reasons.
Platt calibration is monotone, so it does not change their order.

Most IEEE-CIS columns are masked by Vesta, so many reasons can only name the
column family. That limit is stated in docs/problem_framing.md.
"""

from __future__ import annotations

import math

import numpy as np

from fraud.profiling import column_group

WINDOW_TEXT = {"1h": "the past hour", "24h": "the past 24 hours", "7d": "the past 7 days"}
WHO = {"card1": "the card", "entity_key": "the card holder"}

NAMED = {
    "TransactionAmt": "transaction amount is ${v:,.2f}",
    "ProductCD": "product code is {v}",
    "card4": "card network is {v}",
    "card6": "card type is {v}",
    "P_emaildomain": "purchaser email domain is {v}",
    "R_emaildomain": "recipient email domain is {v}",
    "DeviceType": "device type is {v}",
    "DeviceInfo": "device is {v}",
    "addr1": "billing region code is {v}",
    "addr2": "billing country code is {v}",
    "log_amt": "transaction amount (log scale) is {v:.2f}",
    "amt_cents": "amount has {v:.0f} cents",
    "hour": "hour of day {v:.0f} (on the dataset's own clock)",
    "weekday": "day of week {v:.0f}",
}

FAMILY = {
    "C1-C14": "count {f} = {v} (masked; e.g. addresses linked to the card)",
    "D1-D15": "time delta {f} = {v} days (masked; e.g. days since the card was first used)",
    "M1-M9": "match check {f} = {v} (masked; e.g. name vs address match)",
    "V1-V339": "Vesta-engineered feature {f} = {v} (meaning masked)",
    "id_01-38": "device or network attribute {f} = {v} (masked)",
    "card1-6": "card attribute {f} = {v} (masked code)",
    "dist1-2": "distance {f} = {v} (masked)",
}


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt(v):
    if _is_missing(v):
        return "missing"
    if isinstance(v, (float, np.floating)):
        return f"{v:g}"
    return str(v)


def describe(feature: str, value) -> str:
    """A short human-readable description of one feature value."""
    for who_key, who in WHO.items():
        prefix = f"{who_key}_"
        if feature.startswith(prefix):
            name = feature[len(prefix):]
            if _is_missing(value):
                return f"no earlier transaction of {who} on record ({name})"
            if name.startswith("n_"):
                return f"{who} made {value:.0f} earlier transactions in {WINDOW_TEXT[name[2:]]}"
            if name == "secs_since_prev":
                return f"{value / 60:,.0f} minutes since {who}'s previous transaction"
            if name == "past_mean_amt":
                return f"{who}'s average past amount is ${value:,.2f}"
            if name == "amt_to_past_mean":
                return f"amount is {value:.1f}x {who}'s past average"
    if feature in NAMED:
        if _is_missing(value):
            return f"{feature} is missing"
        try:
            return NAMED[feature].format(v=value)
        except (ValueError, TypeError):  # e.g. a numeric format applied to a text value
            return f"{feature} = {_fmt(value)}"
    template = FAMILY.get(column_group(feature))
    if template:
        return template.format(f=feature, v=_fmt(value))
    return f"{feature} = {_fmt(value)}"


def top_reasons(features: list[str], contributions: np.ndarray, values: dict, k: int = 3) -> list[dict]:
    """The k features with the largest positive contribution (log-odds) to the fraud score."""
    contrib = np.asarray(contributions[: len(features)], dtype=float)
    order = [i for i in np.argsort(-contrib) if contrib[i] > 0][:k]
    return [
        {"feature": features[i], "value": None if _is_missing(values[features[i]]) else values[features[i]],
         "contribution": round(float(contrib[i]), 4), "description": describe(features[i], values[features[i]])}
        for i in order
    ]
