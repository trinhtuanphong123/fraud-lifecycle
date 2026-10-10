"""Problem-oriented data analysis: where the money is, where the signal is, what changes over time, how labels are structured.

Descriptive only; method choices live in docs/methodology.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.data import ENTITY_COL, SECONDS_PER_DAY, TARGET, TIME_COL
from fraud.profiling import column_group

AMOUNT = "TransactionAmt"


def value_band(s: pd.Series, edges, labels) -> pd.Series:
    """Cut a numeric column into labelled bands; missing values become their own band "NaN"."""
    b = pd.cut(s.astype(float), edges, labels=labels).astype(object)
    return b.where(s.notna(), "NaN")


def wilson_ci(k, n, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a proportion k / n; stays inside [0, 1] and works for small n."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def rate_and_loss(df: pd.DataFrame, by) -> pd.DataFrame:
    """Per group: volume, fraud rate with a Wilson CI, and fraud dollars.

    Separates two questions that are easy to mix up: where fraud is *frequent*
    (rate) and where fraud *costs money* (share of fraud dollars).
    """
    key = df[by] if isinstance(by, str) else pd.Series(by, index=df.index)
    has_missing = key.isna().any()
    if has_missing:
        key = key.astype(object).where(key.notna(), "NaN")
    fraud_usd = df[AMOUNT].astype(float) * df[TARGET]
    out = (
        pd.DataFrame({"fraud": df[TARGET], "fraud_usd": fraud_usd, "usd": df[AMOUNT].astype(float)})
        .groupby(key, observed=True, sort=not has_missing)
        .agg(n=("fraud", "size"), fraud=("fraud", "sum"), usd=("usd", "sum"), fraud_usd=("fraud_usd", "sum"))
    )
    out["share"] = out["n"] / out["n"].sum()
    out["fraud_rate"] = out["fraud"] / out["n"]
    out["ci_low"], out["ci_high"] = wilson_ci(out["fraud"], out["n"])
    out["share_fraud_usd"] = out["fraud_usd"] / out["fraud_usd"].sum()
    out["fraud_usd_rate"] = out["fraud_usd"] / out["usd"]
    return out[["n", "share", "fraud", "fraud_rate", "ci_low", "ci_high", "fraud_usd", "share_fraud_usd", "fraud_usd_rate"]]


def loss_concentration(amount, y) -> pd.DataFrame:
    """Lorenz-style curve: share of fraud dollars covered by the largest x% of fraud transactions."""
    fraud_amounts = np.sort(np.asarray(amount, dtype=float)[np.asarray(y) == 1])[::-1]
    covered = np.r_[0.0, np.cumsum(fraud_amounts) / fraud_amounts.sum()]
    share_txns = np.arange(len(covered)) / len(fraud_amounts)
    return pd.DataFrame({"share_of_fraud_txns": share_txns, "share_of_fraud_usd": covered})


def amount_decimals(amount) -> pd.Series:
    """Number of decimals (0-3) in each amount. Amounts carry at most 3 decimals in this data."""
    millis = np.round(np.asarray(amount, dtype=np.float64) * 1000).astype(np.int64)
    decimals = np.select([millis % 10 != 0, millis % 100 != 0, millis % 1000 != 0], [3, 2, 1], default=0)
    return pd.Series(decimals, index=getattr(amount, "index", None), name="decimals")


def weekly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Volume, fraud rate with CI and fraud dollars per week."""
    return rate_and_loss(df, "week")


def block_null_rate_by_week(df: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """Weekly missing rate of each missing-together block (rows: weeks, columns: blocks)."""
    rates = {
        f"{b['first']}-{b['last']} ({b['n_cols']})": df[b["first"]].isna().groupby(df["week"]).mean()
        for _, b in blocks.iterrows()
    }
    return pd.DataFrame(rates)


def new_value_rate_by_week(df: pd.DataFrame, col: str) -> pd.Series:
    """Share of each week's non-missing rows whose value never appeared in any earlier week."""
    seen: set = set()
    rates = {}
    for week, values in df[col].dropna().astype(str).groupby(df["week"], sort=True):
        rates[week] = (~values.isin(seen)).mean()
        seen.update(values.unique())
    return pd.Series(rates, name=f"new_{col}_rate").rename_axis("week")


def entity_fraud_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per card holder with at least one fraud: how their fraud and legitimate transactions are arranged in time."""
    d = df[[ENTITY_COL, TIME_COL, TARGET]]
    fraud = d[d[TARGET] == 1].groupby(ENTITY_COL, observed=True).agg(
        n_fraud=(TARGET, "size"), first_fraud=(TIME_COL, "min"), last_fraud=(TIME_COL, "max")
    )
    legit = d[(d[TARGET] == 0) & d[ENTITY_COL].isin(fraud.index)].join(fraud[["first_fraud", "last_fraud"]], on=ENTITY_COL)

    def count(mask):
        return legit[mask].groupby(ENTITY_COL, observed=True).size().reindex(fraud.index, fill_value=0)

    return fraud.assign(
        n_legit_before=count(legit[TIME_COL] < legit["first_fraud"]),
        n_legit_during=count((legit[TIME_COL] > legit["first_fraud"]) & (legit[TIME_COL] < legit["last_fraud"])),
        n_legit_after=count(legit[TIME_COL] > legit["last_fraud"]),
        streak_days=(fraud["last_fraud"] - fraud["first_fraud"]) / SECONDS_PER_DAY,
    )


def fraud_kind_by_week(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly rate of first-time fraud (a card holder's first fraud) vs repeat fraud (their later ones)."""
    first = df[df[TARGET] == 1].groupby(ENTITY_COL, observed=True)[TIME_COL].transform("min")
    is_fraud = df[TARGET] == 1
    is_first = pd.Series(False, index=df.index)
    is_first[first.index] = df.loc[first.index, TIME_COL] == first
    kinds = pd.DataFrame({"first_fraud": is_first, "repeat_fraud": is_fraud & ~is_first})
    return kinds.groupby(df["week"]).mean()


def signal_by_group(mi: pd.DataFrame) -> pd.DataFrame:
    """Summarise a per-feature MI table (see features.mi_table) by column family."""
    t = mi.assign(group=[column_group(f) for f in mi.index])
    top = t.sort_values("mi", ascending=False).groupby("group").head(1)
    out = t.groupby("group").agg(
        n_cols=("mi", "size"), n_above_null=("keep", "sum"), mi_median=("mi", "median"), mi_max=("mi", "max")
    )
    out["top_feature"] = top.reset_index().set_index("group")["feature"]
    return out.sort_values("mi_max", ascending=False)


def block_redundancy(df: pd.DataFrame, blocks: pd.DataFrame, variance: float = 0.95) -> pd.DataFrame:
    """Per block: median |corr| between its columns, and how many principal components hold `variance` of it."""
    rows = []
    for _, b in blocks.iterrows():
        x = df[b["columns"]].dropna().astype(np.float64)
        x = x.loc[:, x.std() > 0]
        if x.shape[1] < 2 or len(x) < 10:
            rows.append({"block": f"{b['first']}-{b['last']}", "n_cols": b["n_cols"], "median_abs_corr": np.nan, "n_components": x.shape[1]})
            continue
        corr = np.corrcoef(x.to_numpy(), rowvar=False)
        off_diag = np.abs(corr[~np.eye(len(corr), dtype=bool)])
        eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
        n_comp = int(np.searchsorted(np.cumsum(eig) / eig.sum(), variance) + 1)
        rows.append({"block": f"{b['first']}-{b['last']}", "n_cols": b["n_cols"], "median_abs_corr": float(np.median(off_diag)), "n_components": n_comp})
    return pd.DataFrame(rows).set_index("block")


def near_constant(df: pd.DataFrame, columns: list[str], threshold: float = 0.99) -> pd.Series:
    """Columns whose most frequent value (NaN included) covers at least `threshold` of rows."""
    top_share = {c: df[c].value_counts(normalize=True, dropna=False).iloc[0] for c in columns}
    s = pd.Series(top_share, name="top_value_share")
    return s[s >= threshold].sort_values(ascending=False)
