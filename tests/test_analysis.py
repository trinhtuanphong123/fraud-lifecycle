import numpy as np
import pandas as pd
import pytest

from fraud.analysis import (
    amount_decimals,
    entity_fraud_profile,
    fraud_kind_by_week,
    loss_concentration,
    new_value_rate_by_week,
    rate_and_loss,
    wilson_ci,
)
from fraud.data import ENTITY_COL, SECONDS_PER_DAY, TARGET, TIME_COL


def test_wilson_ci_matches_hand_computation():
    # k=10, n=100, z=1.96: textbook interval (0.0552, 0.1744)
    low, high = wilson_ci(10, 100)
    assert low == pytest.approx(0.0552, abs=1e-4)
    assert high == pytest.approx(0.1744, abs=1e-4)
    low0, high0 = wilson_ci(0, 20)
    assert low0 == pytest.approx(0.0, abs=1e-12) and 0 < high0 < 0.2


def test_loss_concentration_runs_from_origin_to_one():
    curve = loss_concentration([100, 1, 10, 50], [1, 0, 1, 1])
    assert curve.iloc[0].tolist() == [0.0, 0.0]
    assert curve.iloc[-1].tolist() == [1.0, pytest.approx(1.0)]
    # The largest fraud (100 of 160) comes first.
    assert curve["share_of_fraud_usd"].iloc[1] == pytest.approx(100 / 160)


def test_rate_and_loss_separates_rate_from_dollars():
    df = pd.DataFrame({
        TARGET: [1, 0, 0, 0, 1, 1],
        "TransactionAmt": [1000.0, 5.0, 5.0, 5.0, 1.0, 1.0],
        "g": ["big", "big", "big", "big", "small", None],
    })
    out = rate_and_loss(df, "g")
    assert out.loc["big", "fraud_rate"] == 0.25
    assert out.loc["small", "fraud_rate"] == 1.0
    assert out.loc["big", "share_fraud_usd"] == pytest.approx(1000 / 1002)
    assert "NaN" in out.index


def test_amount_decimals():
    s = pd.Series(np.array([10.0, 49.95, 30.948, 12.5], dtype=np.float32))
    assert amount_decimals(s).tolist() == [0, 2, 3, 1]


def test_new_value_rate_by_week_on_a_hand_worked_example():
    df = pd.DataFrame({"week": [0, 0, 1, 1, 1, 2], "browser": ["a", "b", "a", "c", None, "c"]})
    rates = new_value_rate_by_week(df, "browser")
    assert rates.tolist() == [1.0, 0.5, 0.0]


def test_entity_fraud_profile_and_fraud_kinds():
    day = SECONDS_PER_DAY
    df = pd.DataFrame({
        ENTITY_COL: ["x", "x", "x", "x", "x", "y", "z"],
        TIME_COL: [0, 1 * day, 2 * day, 3 * day, 5 * day, 0, 0],
        TARGET: [0, 1, 0, 1, 0, 1, 0],
        "week": [0, 0, 0, 0, 0, 0, 0],
    })
    p = entity_fraud_profile(df)
    assert set(p.index) == {"x", "y"}
    x = p.loc["x"]
    assert (x["n_fraud"], x["n_legit_before"], x["n_legit_during"], x["n_legit_after"]) == (2, 1, 1, 1)
    assert x["streak_days"] == 2.0
    kinds = fraud_kind_by_week(df).loc[0]
    assert kinds["first_fraud"] == pytest.approx(2 / 7)   # first fraud of x and of y
    assert kinds["repeat_fraud"] == pytest.approx(1 / 7)  # second fraud of x
