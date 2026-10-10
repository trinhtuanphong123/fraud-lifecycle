import numpy as np
import pandas as pd
import pytest

from fraud.data import TARGET
from fraud.profiling import column_group, compare_splits, missing_blocks, missing_vs_target, segment_rates


def test_column_group_matches_whole_names_only():
    assert column_group("V12") == "V1-V339"
    assert column_group("D1") == "D1-D15"
    assert column_group("id_31") == "id_01-38"
    assert column_group("DeviceType") == "DeviceType"
    assert column_group("card4") == "card1-6"
    assert column_group("C1") == "C1-C14"


def test_missing_blocks_split_same_rate_but_different_rows():
    nan = np.nan
    df = pd.DataFrame({
        "a": [nan, nan, 1, 1],
        "b": [nan, nan, 2, 2],   # same rows missing as a
        "c": [1, 1, nan, nan],   # same rate as a, different rows
        "d": [1, 2, 3, 4],
    })
    blocks = missing_blocks(df, ["a", "b", "c", "d"])
    assert sorted(map(sorted, blocks["columns"])) == [["a", "b"], ["c"], ["d"]]
    assert blocks.set_index("first").loc["a", "missing_rate"] == 0.5


def test_missing_vs_target_on_a_hand_worked_example():
    df = pd.DataFrame({TARGET: [1, 1, 0, 0, 0, 1], "x": [np.nan, np.nan, np.nan, 1, 2, 3]})
    row = missing_vs_target(df, ["x"]).loc["x"]
    assert row["missing_rate"] == 0.5
    assert row["fraud_if_missing"] == pytest.approx(2 / 3)
    assert row["fraud_if_present"] == pytest.approx(1 / 3)


def test_segment_rates_keeps_missing_as_a_segment():
    df = pd.DataFrame({TARGET: [1, 0, 0, 1], "p": pd.Categorical(["W", "W", None, None])})
    out = segment_rates(df, "p")
    assert set(out.index) == {"W", "NaN"}
    assert out.loc["NaN", "fraud_rate"] == 0.5
    assert out["share"].sum() == pytest.approx(1.0)


def test_compare_splits_counts_unseen_values_and_missing_shift():
    ref = pd.DataFrame({"browser": pd.Categorical(["chrome 62", "chrome 63", None]), "card1": [1, 2, 3]})
    cur = pd.DataFrame({"browser": pd.Categorical(["chrome 70", "chrome 63", "chrome 71"]), "card1": [1, 9, 9]})
    out = compare_splits(ref, cur, ["browser", "card1"], as_categorical=["card1"])
    assert out.loc["browser", "unseen_share"] == pytest.approx(2 / 3)
    assert out.loc["browser", "miss_diff"] == pytest.approx(-1 / 3)
    assert out.loc["card1", "unseen_share"] == pytest.approx(2 / 3)
