"""A transparent baseline built directly from the data analysis.

The scorecard adds, for each analysed attribute, the log-likelihood ratio of
its band (weight of evidence) to the prior log-odds. This is naive Bayes:
correct if the attributes were independent given the label, which they are
not, so its probabilities are over-confident while its ranking can still be
useful.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from fraud.analysis import amount_decimals, value_band
from fraud.data import TIME_COL

INF = np.inf


def scorecard_bands(df: pd.DataFrame) -> pd.DataFrame:
    """The eight attributes from reports/data_analysis.md, cut into the bands used there."""

    def category(s):
        return s.astype(object).where(s.notna(), "NaN").astype(str)

    hour = (df[TIME_COL] // 3600) % 24
    return pd.DataFrame({
        "ProductCD": category(df["ProductCD"]),
        "card_age_D1": value_band(df["D1"], [-INF, 0, 30, INF], ["0", "1-30", ">30"]),
        "C1": value_band(df["C1"], [-1, 1, 3, 10, 100, INF], ["0-1", "2-3", "4-10", "11-100", ">100"]),
        "V258": value_band(df["V258"], [-INF, 0.5, 1.5, 2.5, INF], ["0", "1", "2", ">=3"]),
        "hour": value_band(hour, [-1, 4, 11, 17, 23], ["0-4", "5-11", "12-17", "18-23"]),
        "card6": category(df["card6"]),
        "DeviceType": category(df["DeviceType"]),
        "amount_decimals": amount_decimals(df["TransactionAmt"]).astype(str),
    }, index=df.index)


class Scorecard(NamedTuple):
    prior_logit: float
    woe: dict[str, pd.Series]


def fit_scorecard(bands: pd.DataFrame, y, smoothing: float = 1.0) -> Scorecard:
    """Weight of evidence per band: log(P(band | fraud) / P(band | legit)).

    `smoothing` pseudo-observations are added at the overall fraud rate, so a
    rare band is pulled towards zero evidence and an unseen band gets none.
    """
    y = np.asarray(y)
    n, n_fraud = len(y), int(y.sum())
    n_legit = n - n_fraud
    prior_logit = float(np.log(n_fraud / n_legit))
    woe = {}
    for col in bands.columns:
        counts = pd.DataFrame({"band": bands[col].to_numpy(), "y": y}).groupby("band")["y"].agg(["sum", "size"])
        fraud = counts["sum"] + smoothing * n_fraud / n
        legit = counts["size"] - counts["sum"] + smoothing * n_legit / n
        woe[col] = np.log(fraud / legit) - prior_logit
    return Scorecard(prior_logit, woe)


def score_scorecard(model: Scorecard, bands: pd.DataFrame) -> np.ndarray:
    logit = model.prior_logit + sum(
        bands[col].map(w).astype(float).fillna(0.0).to_numpy() for col, w in model.woe.items()
    )
    return 1.0 / (1.0 + np.exp(-logit))
