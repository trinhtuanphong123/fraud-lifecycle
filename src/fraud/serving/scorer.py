"""Score one transaction with exactly the training-time feature pipeline."""

from __future__ import annotations

import math
import threading

import numpy as np
import pandas as pd

from fraud.calibrate import platt
from fraud.data import ENTITY_COL, TIME_COL, add_entity_key, add_time_columns
from fraud.decision import DECLINE, bayes_policy, cost_fn, cost_fp
from fraud.features import PAST_KEYS, add_row_features
from fraud.serving.bundle import Bundle
from fraud.serving.history import HistoryStore
from fraud.serving.reasons import top_reasons


class Scorer:
    def __init__(self, bundle: Bundle, store: HistoryStore):
        self.bundle, self.store = bundle, store
        self._lock = threading.Lock()  # the history store is shared state

    def frame(self, tx: dict) -> pd.DataFrame:
        """One-row frame with every model feature, built with the same functions as in training."""
        data = {}
        for col, dtype in self.bundle.raw_dtypes.items():
            v = tx.get(col)
            missing = v is None or (isinstance(v, float) and math.isnan(v))
            if dtype == "category":
                data[col] = pd.Categorical([None if missing else str(v)])
            elif missing:
                data[col] = np.array([np.nan], dtype=np.float32 if dtype.startswith("int") else dtype)
            else:
                data[col] = np.array([v], dtype=dtype)
        df = add_row_features(add_entity_key(add_time_columns(pd.DataFrame(data))))
        past = self.store.features(int(df[TIME_COL].iat[0]), float(df["TransactionAmt"].iat[0]), self._keys(df))
        return df.assign(**{k: np.float32(v) for k, v in past.items()})

    @staticmethod
    def _keys(df: pd.DataFrame) -> dict:
        return {k: df[k].iat[0] for k in PAST_KEYS}

    def score(self, tx: dict, record: bool = True, explain: str = "auto") -> dict:
        """Score one transaction.

        explain: "auto" computes reasons only for declines, the decision the
        customer sees at checkout (TreeSHAP over the full model takes ~0.65 s
        per row, scoring alone ~35 ms). Review cases get their reasons when a
        reviewer opens them (explain="always"), off the payment path.
        """
        b = self.bundle
        with self._lock:  # the history store is read, then updated
            df = self.frame(tx)
            if record:
                self.store.record(int(df[TIME_COL].iat[0]), float(df["TransactionAmt"].iat[0]), self._keys(df))
        x = df[b.features]
        raw = float(b.booster.predict(x, num_threads=1)[0])  # one row: threads cost more than they save

        p = float(platt(np.array([raw]), b.platt_a, b.platt_b)[0])
        amount = float(df["TransactionAmt"].iat[0])
        c_fn, c_fp = float(cost_fn(amount, b.costs)), float(cost_fp(amount, b.costs))
        decline = bayes_policy([p], [amount], b.costs, fp_penalty=b.fp_penalty)[0] == DECLINE
        review = b.costs["manual_review"]
        review_cost = review["cost_per_case"] + (1 - review["catch_rate"]) * p * c_fn
        cost_approve, cost_decline = p * c_fn, (1 - p) * (c_fp + b.fp_penalty)
        review_recommended = bool(review_cost < min(cost_approve, cost_decline))
        explained = explain == "always" or (explain == "auto" and decline)
        reasons = []
        if explained:
            contributions = b.booster.predict(x, pred_contrib=True, num_threads=1)[0]
            reasons = top_reasons(b.features, contributions, {f: _plain(x[f].iat[0]) for f in b.features})
        return {
            "transaction_id": int(tx["TransactionID"]),
            "fraud_probability": round(p, 6),
            "raw_score": round(raw, 6),
            "decision": "decline" if decline else "approve",
            "decline_threshold": round((c_fp + b.fp_penalty) / (c_fp + b.fp_penalty + c_fn), 6),
            "expected_cost": {"approve": round(cost_approve, 2), "decline": round((1 - p) * c_fp, 2), "review": round(review_cost, 2)},
            "review_recommended": review_recommended,
            "explained": explained,
            "reasons": reasons,
            "model_version": b.version,
        }


def _plain(v):
    """numpy / pandas scalars to JSON-friendly Python values."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, np.generic):
        v = v.item()
        return None if isinstance(v, float) and math.isnan(v) else v
    return v
