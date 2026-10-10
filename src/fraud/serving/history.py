"""Online history for the past-only behaviour features.

Offline, `features.add_past_features` computes the features for a whole table
at once. Online, transactions arrive one at a time, so the same quantities are
kept incrementally per key (card1 and entity_key):

- the times and amounts of the last 7 days (enough for the 1 h / 24 h / 7 d counts);
- for older transactions only their latest time, amount sum and count.

Semantics match the offline code exactly: only transactions with a strictly
earlier TransactionDT count, the window is [t - W, t), and the past mean uses
all earlier transactions. tests/test_history.py replays real data through both
and checks they agree. Transactions are assumed to arrive in time order (ties
allowed).
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraud.data import TIME_COL
from fraud.features import PAST_KEYS, WINDOWS

HORIZON = max(WINDOWS.values())


@dataclass
class KeyState:
    times: list = field(default_factory=list)      # sorted, within HORIZON of the latest time
    amounts: list = field(default_factory=list)    # aligned with times
    older_last: float = np.nan                     # latest time among pruned transactions
    older_sum: float = 0.0
    older_count: int = 0


class HistoryStore:
    def __init__(self):
        self.state: dict[tuple[str, str], KeyState] = {}

    # ---------------------------------------------------------- queries

    def features(self, t: int, amount: float, keys: dict[str, object]) -> dict[str, float]:
        """The 6 past features for each key, from transactions strictly before t."""
        amount = float(np.float32(amount))
        out = {}
        for key_type in PAST_KEYS:
            s = self.state.get((key_type, str(keys[key_type])))
            times = s.times if s else []
            before = bisect_left(times, t)  # entries with time < t
            for name, seconds in WINDOWS.items():
                out[f"{key_type}_n_{name}"] = float(before - bisect_left(times, t - seconds))
            if before > 0:
                prev = times[before - 1]
            elif s is not None and not np.isnan(s.older_last) and s.older_last < t:
                prev = s.older_last
            else:
                prev = np.nan
            out[f"{key_type}_secs_since_prev"] = t - prev
            n_past = (s.older_count if s else 0) + before
            if n_past:
                past_sum = s.older_sum + float(np.sum(s.amounts[:before], dtype=np.float64))
                mean = past_sum / n_past
            else:
                mean = np.nan
            out[f"{key_type}_past_mean_amt"] = mean
            out[f"{key_type}_amt_to_past_mean"] = amount / mean
        return {k: float(np.float32(v)) for k, v in out.items()}

    # ---------------------------------------------------------- updates

    def record(self, t: int, amount: float, keys: dict[str, object]) -> None:
        amount = float(np.float32(amount))
        for key_type in PAST_KEYS:
            s = self.state.setdefault((key_type, str(keys[key_type])), KeyState())
            i = bisect_left(s.times, t + 1)  # after any equal times, keeps insertion order for ties
            s.times.insert(i, t)
            s.amounts.insert(i, amount)
            self._prune(s, s.times[-1] - HORIZON)

    @staticmethod
    def _prune(s: KeyState, cutoff: float) -> None:
        k = bisect_left(s.times, cutoff)
        if k:
            s.older_last = s.times[k - 1] if np.isnan(s.older_last) else max(s.older_last, s.times[k - 1])
            s.older_sum += float(np.sum(s.amounts[:k], dtype=np.float64))
            s.older_count += k
            del s.times[:k], s.amounts[:k]

    # ------------------------------------------------- bulk build and I/O

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> HistoryStore:
        """Build the state after all transactions in df (columns TransactionDT, TransactionAmt and the keys)."""
        return cls._from_tables(*cls._tables(df))

    @classmethod
    def _from_tables(cls, recent: pd.DataFrame, older: pd.DataFrame) -> HistoryStore:
        store = cls()
        for r in older.itertuples(index=False):
            store.state[(r.key_type, r.key)] = KeyState(
                older_last=float(r.older_last), older_sum=float(r.older_sum), older_count=int(r.older_count))
        for (key_type, key), g in recent.groupby(["key_type", "key"], sort=False):
            s = store.state.setdefault((key_type, key), KeyState())
            s.times, s.amounts = g["t"].tolist(), g["amount"].tolist()
        return store

    @staticmethod
    def _tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        parts_recent, parts_older = [], []
        for key_type in PAST_KEYS:
            d = pd.DataFrame({
                "key_type": key_type,
                "key": df[key_type].astype(str).to_numpy(),
                "t": df[TIME_COL].to_numpy(dtype=np.int64),
                "amount": df["TransactionAmt"].to_numpy(dtype=np.float32).astype(np.float64),
            }).sort_values(["key", "t"], kind="stable")
            latest = d.groupby("key")["t"].transform("max")
            is_recent = d["t"] >= latest - HORIZON
            parts_recent.append(d[is_recent])
            parts_older.append(
                d[~is_recent].groupby("key", as_index=False)
                .agg(older_last=("t", "max"), older_sum=("amount", "sum"), older_count=("amount", "size"))
                .assign(key_type=key_type)
            )
        return pd.concat(parts_recent, ignore_index=True), pd.concat(parts_older, ignore_index=True)

    @classmethod
    def save_frame(cls, df: pd.DataFrame, directory) -> None:
        recent, older = cls._tables(df)
        recent.to_parquet(f"{directory}/history_recent.parquet", index=False)
        older.to_parquet(f"{directory}/history_older.parquet", index=False)

    @classmethod
    def load(cls, directory) -> HistoryStore:
        return cls._from_tables(pd.read_parquet(f"{directory}/history_recent.parquet"),
                                pd.read_parquet(f"{directory}/history_older.parquet"))
