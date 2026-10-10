"""Load IEEE-CIS data, add time columns, and build leakage-aware splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"
ENTITY_COL = "entity_key"

# Vesta never published the real start date. 2017-12-01 is the community
# convention; only relative time matters for the analysis.
START_DATE = pd.Timestamp("2017-12-01")
SECONDS_PER_DAY = 86_400


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink dtypes: float64 -> float32, ints -> smallest, strings -> category."""
    # Build a new frame in one go: replacing ~400 columns one by one fragments it.
    columns = {}
    for col in df.columns:
        s = df[col]
        kind = s.dtype.kind
        if kind == "f":
            s = s.astype(np.float32)
        elif kind in "iu":
            s = pd.to_numeric(s, downcast="integer")
        elif kind == "O":
            s = s.astype("category")
        columns[col] = s
    return pd.DataFrame(columns, index=df.index)


def load_raw(split: str = "train", raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read the Kaggle CSVs for one split and left-join identity onto transactions."""
    tx = pd.read_csv(raw_dir / f"{split}_transaction.csv")
    identity = pd.read_csv(raw_dir / f"{split}_identity.csv")
    # The Kaggle test identity file names its columns "id-01" instead of "id_01".
    identity.columns = identity.columns.str.replace("-", "_")
    df = tx.merge(identity, on=ID_COL, how="left")
    return downcast(df)


def add_time_columns(df: pd.DataFrame, start_date: pd.Timestamp = START_DATE) -> pd.DataFrame:
    """Add day, week, month and an (assumed) calendar timestamp from TransactionDT."""
    seconds = df[TIME_COL].astype(np.int64)
    day = (seconds // SECONDS_PER_DAY).astype(np.int32)
    timestamp = start_date + pd.to_timedelta(seconds, unit="s")
    return df.assign(
        timestamp=timestamp,
        day=day,
        week=(day // 7).astype(np.int32),
        month=timestamp.dt.to_period("M").astype(str).astype("category"),
    )


def add_entity_key(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate card-holder key: card1 + addr1 + card issue day (day - D1).

    Used only to measure how much the model "memorises people" across splits.
    It is never a model feature.
    """
    issue_day = df["day"] - df["D1"]
    # pandas 3 keeps NaN through astype(str); spell missing parts out so a
    # missing addr1 or D1 never turns the whole key into NaN.
    parts = [s.astype(str).fillna("nan") for s in (df["card1"], df["addr1"], issue_day)]
    key = parts[0] + "_" + parts[1] + "_" + parts[2]
    return df.assign(**{ENTITY_COL: key.astype("category")})


def load_dataset(
    split: str = "train",
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load one split with time and entity columns, cached as parquet after the first run."""
    path = processed_dir / f"{split}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    df = add_entity_key(add_time_columns(load_raw(split, raw_dir)))
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def random_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle split. Kept only as the leaky baseline to compare against."""
    is_test = np.random.default_rng(seed).random(len(df)) < test_size
    return df[~is_test], df[is_test]


def time_split(
    df: pd.DataFrame, test_size: float = 0.2, gap_days: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earliest transactions, test on the latest.

    gap_days drops a window between train and test, e.g. to mimic labels that
    are not yet available when the model is trained.
    """
    cutoff = df[TIME_COL].quantile(1 - test_size)
    train = df[df[TIME_COL] <= cutoff]
    test = df[df[TIME_COL] > cutoff + gap_days * SECONDS_PER_DAY]
    return train, test


def split_by_week(
    df: pd.DataFrame, valid_start: int, test_start: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on weeks before valid_start, validate until test_start, test on the rest."""
    week = df["week"]
    return (
        df[week < valid_start],
        df[(week >= valid_start) & (week < test_start)],
        df[week >= test_start],
    )


def drop_seen_entities(
    train: pd.DataFrame, test: pd.DataFrame, key: str = ENTITY_COL
) -> pd.DataFrame:
    """Keep only test rows whose entity never appears in train."""
    return test[~test[key].isin(train[key].unique())]
