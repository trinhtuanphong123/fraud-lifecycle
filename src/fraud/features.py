"""Feature engineering and screening.

Every time-based feature looks strictly into the past: a transaction only sees
rows of the same key with an earlier TransactionDT. Whole-dataset tricks such as
frequency encoding are avoided because they leak the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.data import ENTITY_COL, ID_COL, SECONDS_PER_DAY, TARGET, TIME_COL

# Raw time is excluded because a time split would ask the model to extrapolate
# it; the entity key exists only for leakage analysis and as a grouping key.
NON_FEATURES = {ID_COL, TARGET, TIME_COL, "timestamp", "day", "week", "month", ENTITY_COL}

PAST_KEYS = ("card1", ENTITY_COL)
WINDOWS = {"1h": 3_600, "24h": SECONDS_PER_DAY, "7d": 7 * SECONDS_PER_DAY}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def is_categorical(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype)


# ---------------------------------------------------------------- engineering


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computed from a single row."""
    amt = df["TransactionAmt"].astype(np.float64)
    return df.assign(
        log_amt=np.log1p(amt).astype(np.float32),
        # Cents part: tiny or odd amounts are typical of card testing.
        amt_cents=np.round((amt - np.floor(amt)) * 100).astype(np.int16),
        hour=((df[TIME_COL] // 3_600) % 24).astype(np.int8),
        weekday=(df["day"] % 7).astype(np.int8),
    )


def _past_stats(codes: np.ndarray, t: np.ndarray, amt: np.ndarray, windows: dict[str, int]) -> dict:
    """History statistics per row, using only rows of the same group with a strictly earlier time.

    Rows are placed on one sorted axis, key = group * span + time, so every
    group occupies its own interval and a binary search finds window bounds
    for all rows at once.
    """
    span = int(t.max()) + 1
    group_base = codes.astype(np.int64) * span
    key = group_base + t
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]

    start = np.searchsorted(sorted_key, group_base, side="left")  # first row of the group
    before = np.searchsorted(sorted_key, key, side="left")        # rows strictly earlier
    n_past = before - start

    stats = {}
    for name, seconds in windows.items():
        lower = group_base + np.maximum(t - seconds, 0)
        stats[f"n_{name}"] = before - np.searchsorted(sorted_key, lower, side="left")

    has_past = n_past > 0
    prev_time = sorted_key[np.maximum(before - 1, 0)] - group_base
    stats["secs_since_prev"] = np.where(has_past, t - prev_time, np.nan)

    cum_amt = np.r_[0.0, np.cumsum(amt[order])]
    past_sum = cum_amt[before] - cum_amt[start]
    past_mean = np.divide(past_sum, n_past, out=np.full(len(t), np.nan), where=has_past)
    stats["past_mean_amt"] = past_mean
    stats["amt_to_past_mean"] = amt / past_mean
    return stats


def add_past_features(
    df: pd.DataFrame, keys: tuple[str, ...] = PAST_KEYS, windows: dict[str, int] = WINDOWS
) -> pd.DataFrame:
    """Velocity and amount-history features per key, e.g. card1_n_24h, entity_key_amt_to_past_mean."""
    t = df[TIME_COL].to_numpy(dtype=np.int64)
    amt = df["TransactionAmt"].to_numpy(dtype=np.float64)
    new = {}
    for k in keys:
        codes, _ = pd.factorize(df[k], use_na_sentinel=False)
        for name, values in _past_stats(codes, t, amt, windows).items():
            new[f"{k}_{name}"] = values.astype(np.float32)
    return df.assign(**new)


# ------------------------------------------------------------------ screening


def binned_codes(s: pd.Series, n_bins: int) -> np.ndarray:
    """Discretise a column for MI: quantile bins for numbers, codes for categories. NaN is code 0."""
    if is_categorical(s):
        return s.cat.codes.to_numpy(dtype=np.int64) + 1
    x = s.to_numpy(dtype=np.float64)
    missing = np.isnan(x)
    if missing.all():
        return np.zeros(len(x), dtype=np.int64)
    edges = np.unique(np.quantile(x[~missing], np.linspace(0, 1, n_bins + 1)[1:-1]))
    codes = np.searchsorted(edges, x, side="right") + 1
    codes[missing] = 0
    return codes


def _mutual_information(codes: np.ndarray, y: np.ndarray, n_codes: int) -> float:
    """Plug-in MI (nats) between a discrete feature and a binary label."""
    joint = np.bincount(codes * 2 + y, minlength=2 * n_codes).reshape(n_codes, 2) / len(y)
    expected = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / expected[nz])))


def mi_table(
    df: pd.DataFrame,
    features: list[str],
    n_bins: int = 20,
    n_permutations: int = 20,
    null_quantile: float = 0.99,
    seed: int = 0,
) -> pd.DataFrame:
    """MI of each feature with the label, against a permutation null.

    The plug-in estimator is biased upwards, roughly by (k - 1) / (2n) nats for
    a feature with k occupied bins (Miller-Madow). High-cardinality features
    therefore look informative even when they are noise. Shuffling the label
    gives each feature its own null distribution, so the bias is controlled
    feature by feature.
    """
    y = df[TARGET].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    permuted = [rng.permutation(y) for _ in range(n_permutations)]
    rows = []
    for f in features:
        codes = binned_codes(df[f], n_bins)
        n_codes = int(codes.max()) + 1
        null = [_mutual_information(codes, yp, n_codes) for yp in permuted]
        rows.append({
            "feature": f,
            "mi": _mutual_information(codes, y, n_codes),
            "null_q": float(np.quantile(null, null_quantile)),
            "miller_madow_bias": (len(np.unique(codes)) - 1) / (2 * len(y)),
            "n_bins": len(np.unique(codes)),
        })
    out = pd.DataFrame(rows).set_index("feature").sort_values("mi", ascending=False)
    out["keep"] = out["mi"] > out["null_q"]
    return out


def drop_correlated(df: pd.DataFrame, features: list[str], max_corr: float = 0.98) -> list[str]:
    """Walk features in the given priority order; drop a numeric one if it nearly duplicates a kept one.

    NaN is filled with the column mean before computing correlations, which is
    an approximation; it is good enough to find the many near-copies among the
    V columns.
    """
    numeric = [f for f in features if not is_categorical(df[f])]
    x = df[numeric].astype(np.float64)
    x = x.fillna(x.mean()).fillna(0.0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.abs(np.corrcoef(x, rowvar=False)))
    position = {f: i for i, f in enumerate(numeric)}

    kept, kept_numeric = [], []
    for f in features:
        if f in position:
            if kept_numeric and corr[position[f], kept_numeric].max() > max_corr:
                continue
            kept_numeric.append(position[f])
        kept.append(f)
    return kept


def screen_features(df: pd.DataFrame, features: list[str], cfg: dict) -> tuple[list[str], pd.DataFrame]:
    """MI screening against the permutation null, then removal of near-duplicates (highest MI wins)."""
    table = mi_table(
        df, features, n_bins=cfg["n_bins"], n_permutations=cfg["n_permutations"],
        null_quantile=cfg["null_quantile"], seed=cfg.get("seed", 0),
    )
    informative = table.index[table["keep"]].tolist()
    return drop_correlated(df, informative, cfg["max_corr"]), table
