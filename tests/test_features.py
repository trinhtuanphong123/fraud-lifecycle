import numpy as np
import pandas as pd

from fraud.data import TARGET, TIME_COL
from fraud.features import add_past_features, add_row_features, drop_correlated, mi_table

HOUR, DAY = 3_600, 86_400


def history(times, amounts, card=1):
    return pd.DataFrame({TIME_COL: times, "TransactionAmt": amounts, "card1": card})


def test_past_features_on_a_hand_worked_example():
    df = history([0, 1_800, 3_600, 90_000], [10.0, 20.0, 30.0, 40.0])
    out = add_past_features(df, keys=("card1",))

    # Window [t - W, t): the row itself and later rows never count.
    assert out["card1_n_1h"].tolist() == [0, 1, 2, 0]
    assert out["card1_n_24h"].tolist() == [0, 1, 2, 1]
    assert out["card1_n_7d"].tolist() == [0, 1, 2, 3]
    np.testing.assert_array_equal(out["card1_secs_since_prev"], [np.nan, 1_800, 1_800, 86_400])
    np.testing.assert_array_equal(out["card1_past_mean_amt"], [np.nan, 10, 15, 20])
    np.testing.assert_allclose(out["card1_amt_to_past_mean"], [np.nan, 2.0, 2.0, 2.0])


def test_past_features_ignore_the_future():
    past = history([100, 2_000, 5_000], [5.0, 50.0, 7.0])
    future = history([6_000, 6_500], [999.0, 1.0])
    alone = add_past_features(past, keys=("card1",))
    with_future = add_past_features(pd.concat([past, future], ignore_index=True), keys=("card1",)).iloc[:3]
    pd.testing.assert_frame_equal(alone, with_future)


def test_past_features_keep_cards_apart_and_ignore_same_second_rows():
    df = pd.DataFrame({
        TIME_COL: [1_000, 1_000, 1_500, 1_200],
        "TransactionAmt": [1.0, 2.0, 3.0, 4.0],
        "card1": [1, 1, 1, 2],
    })
    out = add_past_features(df, keys=("card1",))
    # Two rows at the same second are not each other's past; card 2 has no history.
    assert out["card1_n_1h"].tolist() == [0, 0, 2, 0]


def test_row_features():
    df = pd.DataFrame({
        TIME_COL: [DAY + 5 * HOUR], "day": [1], "TransactionAmt": np.array([49.95], dtype=np.float32),
    })
    out = add_row_features(df)
    assert out["amt_cents"].item() == 95
    assert out["hour"].item() == 5
    assert out["weekday"].item() == 1


def test_mi_keeps_signal_and_drops_noise_and_high_cardinality():
    rng = np.random.default_rng(0)
    n = 20_000
    y = (rng.random(n) < 0.1).astype(int)
    df = pd.DataFrame({
        TARGET: y,
        "signal": y + rng.normal(0, 0.5, n),
        "noise": rng.normal(size=n),
        # Many categories, no relation to y: inflated plug-in MI, but the null catches it.
        "many_levels": pd.Categorical(rng.integers(0, 2_000, n)),
    })
    table = mi_table(df, ["signal", "noise", "many_levels"], n_permutations=20)
    assert table.loc["signal", "keep"]
    assert not table.loc["noise", "keep"]
    assert not table.loc["many_levels", "keep"]
    assert table.loc["many_levels", "mi"] > table.loc["noise", "mi"]


def test_drop_correlated_keeps_the_first_of_near_duplicates():
    rng = np.random.default_rng(0)
    a = rng.normal(size=1_000)
    df = pd.DataFrame({"a": a, "a_copy": a * 2 + 1e-3 * rng.normal(size=1_000), "b": rng.normal(size=1_000)})
    assert drop_correlated(df, ["a", "a_copy", "b"]) == ["a", "b"]
    assert drop_correlated(df, ["a_copy", "a", "b"]) == ["a_copy", "b"]
