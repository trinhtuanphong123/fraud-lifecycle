import numpy as np
import pandas as pd
import pytest

from fraud.data import (
    ENTITY_COL,
    SECONDS_PER_DAY,
    TIME_COL,
    add_entity_key,
    add_time_columns,
    downcast,
    drop_seen_entities,
    load_raw,
    random_split,
    split_by_week,
    time_split,
)
from fraud.features import feature_columns


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 1_000
    raw = pd.DataFrame({
        "TransactionID": np.arange(n),
        "isFraud": rng.random(n) < 0.05,
        TIME_COL: np.sort(rng.integers(SECONDS_PER_DAY, 180 * SECONDS_PER_DAY, n)),
        "TransactionAmt": rng.exponential(100, n),
        "card1": rng.integers(1000, 1050, n),
        "addr1": rng.choice([100.0, 200.0, np.nan], n),
        "D1": rng.integers(0, 500, n).astype(float),
    })
    return add_entity_key(add_time_columns(raw))


def test_time_split_has_no_temporal_overlap(df):
    train, test = time_split(df, test_size=0.3)
    assert train[TIME_COL].max() < test[TIME_COL].min()
    assert len(train) + len(test) == len(df)


def test_time_split_gap_leaves_a_hole(df):
    train, test = time_split(df, test_size=0.3, gap_days=10)
    assert test[TIME_COL].min() - train[TIME_COL].max() > 10 * SECONDS_PER_DAY


def test_split_by_week_is_ordered_and_complete(df):
    train, valid, test = split_by_week(df, valid_start=10, test_start=20)
    assert train[TIME_COL].max() < valid[TIME_COL].min()
    assert valid[TIME_COL].max() < test[TIME_COL].min()
    assert len(train) + len(valid) + len(test) == len(df)
    assert set(valid["week"]) == set(range(10, 20))


def test_random_split_partitions_rows(df):
    train, test = random_split(df, test_size=0.3, seed=1)
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(test) == len(df)


def test_drop_seen_entities_removes_every_shared_entity(df):
    train, test = time_split(df, test_size=0.3)
    unseen = drop_seen_entities(train, test)
    assert not set(unseen[ENTITY_COL]) & set(train[ENTITY_COL])
    assert len(unseen) < len(test)  # the fixture has repeated card holders


def test_entity_key_is_never_missing(df):
    # Regression: pandas 3 propagated NaN in addr1 / D1 into the whole key.
    df.loc[df.index[:5], "D1"] = np.nan
    keys = add_entity_key(df)[ENTITY_COL]
    assert df["addr1"].isna().any()
    assert keys.notna().all()
    assert keys[df["addr1"].isna()].str.contains("_nan_").all()


def test_time_columns_follow_transaction_dt(df):
    assert (df["day"] == df[TIME_COL] // SECONDS_PER_DAY).all()
    assert (df["week"] == df["day"] // 7).all()
    assert df["month"].astype(str).iloc[0] == "2017-12"


def test_feature_columns_exclude_target_time_and_entity(df):
    cols = feature_columns(df)
    for leaky in ["isFraud", TIME_COL, "day", "week", "month", "timestamp", ENTITY_COL]:
        assert leaky not in cols
    assert "TransactionAmt" in cols


def test_downcast_shrinks_without_changing_values():
    raw = pd.DataFrame({"f": [1.5, 2.25], "i": [1, 2], "s": ["a", "b"]})
    out = downcast(raw.copy())
    assert out["f"].dtype == np.float32
    assert out["i"].dtype == np.int8
    assert out["s"].dtype == "category"
    np.testing.assert_allclose(out["f"], raw["f"])


def test_load_raw_joins_identity_and_fixes_test_column_names(tmp_path):
    pd.DataFrame({"TransactionID": [1, 2], TIME_COL: [86_400, 90_000]}).to_csv(
        tmp_path / "test_transaction.csv", index=False
    )
    pd.DataFrame({"TransactionID": [2], "id-01": [-5.0]}).to_csv(
        tmp_path / "test_identity.csv", index=False
    )
    out = load_raw("test", tmp_path)
    assert len(out) == 2
    assert "id_01" in out.columns
    assert out["id_01"].isna().sum() == 1
