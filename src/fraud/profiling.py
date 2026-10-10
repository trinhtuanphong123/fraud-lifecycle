"""Data profiling: column groups, missingness structure, segment fraud rates, train/test comparison."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from fraud.data import TARGET
from fraud.features import is_categorical

# Vesta masks most column names; columns come in numbered families.
GROUP_PATTERNS = [
    (r"V\d+", "V1-V339"),
    (r"C\d+", "C1-C14"),
    (r"D\d+", "D1-D15"),
    (r"M\d", "M1-M9"),
    (r"card\d", "card1-6"),
    (r"addr\d", "addr1-2"),
    (r"dist\d", "dist1-2"),
    (r"id_\d+", "id_01-38"),
]


def column_group(col: str) -> str:
    for pattern, group in GROUP_PATTERNS:
        if re.fullmatch(pattern, col):
            return group
    return col


def group_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Per column family: size, how many are stored as categories, missing rates, cardinality."""
    columns = list(df.columns) if columns is None else columns
    per_col = pd.DataFrame({
        "group": [column_group(c) for c in columns],
        "categorical": [is_categorical(df[c]) for c in columns],
        "missing": df[columns].isna().mean().to_numpy(),
        "nunique": df[columns].nunique().to_numpy(),
    })
    return per_col.groupby("group", sort=False).agg(
        n_cols=("missing", "size"),
        n_categorical=("categorical", "sum"),
        miss_min=("missing", "min"),
        miss_median=("missing", "median"),
        miss_max=("missing", "max"),
        nunique_median=("nunique", "median"),
        nunique_max=("nunique", "max"),
    )


def missing_blocks(df: pd.DataFrame, columns: list[str], sample: int = 100_000, seed: int = 0) -> pd.DataFrame:
    """Group columns that are missing on exactly the same rows.

    Columns are compared by a hash of their NaN mask on a row sample, so two
    columns with the same missing rate but different missing rows land in
    different blocks.
    """
    mask = df[columns].isna()
    if len(mask) > sample:
        mask = mask.sample(sample, random_state=seed)
    signature = {c: hashlib.sha1(np.packbits(mask[c].to_numpy()).tobytes()).hexdigest() for c in columns}
    blocks: dict[str, list[str]] = {}
    for c in columns:
        blocks.setdefault(signature[c], []).append(c)
    rows = [
        {"n_cols": len(cols), "missing_rate": df[cols[0]].isna().mean(), "first": cols[0], "last": cols[-1], "columns": cols}
        for cols in blocks.values()
    ]
    return pd.DataFrame(rows).sort_values("missing_rate").reset_index(drop=True)


def missing_vs_target(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fraud rate on rows where each column is missing vs present."""
    y = df[TARGET]
    rows = []
    for c in columns:
        missing = df[c].isna()
        rows.append({
            "column": c,
            "missing_rate": missing.mean(),
            "fraud_if_missing": y[missing].mean(),
            "fraud_if_present": y[~missing].mean(),
        })
    return pd.DataFrame(rows).set_index("column")


def segment_rates(df: pd.DataFrame, col: str, top: int | None = None) -> pd.DataFrame:
    """Row count, share and fraud rate per value of one column; NaN is its own segment."""
    s = df[col].astype(object)
    label = s.where(s.notna(), "NaN").rename(col)
    out = df.groupby(label)[TARGET].agg(n="size", fraud_rate="mean").sort_values("n", ascending=False)
    out.insert(1, "share", out["n"] / len(df))
    return out if top is None else out.head(top)


def compare_splits(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    as_categorical: Iterable[str] = (),
) -> pd.DataFrame:
    """How the current split differs from the reference: missing rates, and unseen category values.

    unseen_share is the share of non-missing current rows whose value never
    appears in the reference. It is computed for categorical columns and for
    numeric-coded categoricals listed in as_categorical (e.g. card1).
    """
    as_categorical = set(as_categorical)
    rows = []
    for c in columns:
        row = {"column": c, "miss_ref": reference[c].isna().mean(), "miss_cur": current[c].isna().mean()}
        if c in as_categorical or is_categorical(reference[c]) or is_categorical(current[c]):
            seen = set(reference[c].dropna().astype(str))
            values = current[c].dropna().astype(str)
            row["unseen_share"] = (~values.isin(seen)).mean() if len(values) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).set_index("column")
    out.insert(2, "miss_diff", out["miss_cur"] - out["miss_ref"])
    return out
