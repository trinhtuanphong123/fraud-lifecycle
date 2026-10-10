import numpy as np
import pandas as pd
import pytest

from fraud.data import ENTITY_COL, PROCESSED_DIR, TIME_COL
from fraud.features import PAST_KEYS, WINDOWS, add_past_features
from fraud.serving.history import HistoryStore

PAST_COLUMNS = [f"{k}_{n}" for k in PAST_KEYS for n in [f"n_{w}" for w in WINDOWS] +
                ["secs_since_prev", "past_mean_amt", "amt_to_past_mean"]]


def replay(store: HistoryStore, rows: pd.DataFrame) -> pd.DataFrame:
    """Score-then-record each row in order, as the API does."""
    out = []
    for r in rows.itertuples(index=False):
        keys = {k: getattr(r, k) for k in PAST_KEYS}
        out.append(store.features(getattr(r, TIME_COL), r.TransactionAmt, keys))
        store.record(getattr(r, TIME_COL), r.TransactionAmt, keys)
    return pd.DataFrame(out, index=rows.index)[PAST_COLUMNS]


def test_online_features_match_offline_on_a_hand_example():
    df = pd.DataFrame({
        TIME_COL: [0, 1_800, 1_800, 3_600, 90_000, 700_000],
        "TransactionAmt": np.array([10, 20, 25, 30, 40, 50], dtype=np.float32),
        "card1": [1, 1, 1, 1, 1, 1],
        ENTITY_COL: ["a", "a", "b", "a", "a", "a"],
    })
    offline = add_past_features(df)[PAST_COLUMNS]
    online = replay(HistoryStore(), df)
    pd.testing.assert_frame_equal(online.astype(np.float32), offline.astype(np.float32), check_exact=False, rtol=1e-6)
    assert online.loc[2, "card1_n_1h"] == 1          # the same-second row is not its past
    assert online.loc[5, "card1_n_7d"] == 0          # 700,000 s later: everything left the 7-day window
    assert online.loc[5, "card1_past_mean_amt"] == pytest.approx(25.0)  # ...but still counts for the mean


def test_seeded_store_continues_like_one_built_by_replay():
    rng = np.random.default_rng(0)
    n = 3_000
    df = pd.DataFrame({
        TIME_COL: np.sort(rng.integers(0, 40 * 86_400, n)),
        "TransactionAmt": rng.exponential(80, n).astype(np.float32),
        "card1": rng.integers(0, 50, n),
        ENTITY_COL: rng.integers(0, 300, n).astype(str),
    })
    cut = 2_000
    seeded = HistoryStore.from_frame(df.iloc[:cut])
    replayed = HistoryStore()
    replay(replayed, df.iloc[:cut])
    pd.testing.assert_frame_equal(replay(seeded, df.iloc[cut:]), replay(replayed, df.iloc[cut:]), rtol=1e-6)


@pytest.mark.data
@pytest.mark.skipif(not (PROCESSED_DIR / "train.parquet").exists(), reason="needs the IEEE-CIS data")
def test_online_features_match_offline_on_real_data():
    """Training-serving skew check: 2,000 real transactions through the online store vs the batch features."""
    df = pd.read_parquet(PROCESSED_DIR / "train.parquet", columns=[TIME_COL, "TransactionAmt", "card1", ENTITY_COL])
    offline = add_past_features(df)
    t0 = 140 * 86_400
    store = HistoryStore.from_frame(df[df[TIME_COL] < t0])
    rows = df[df[TIME_COL] >= t0].sort_values(TIME_COL, kind="stable").head(2_000)
    online = replay(store, rows)
    expected = offline.loc[rows.index, PAST_COLUMNS]
    assert (online.notna() == expected.notna()).all().all()
    np.testing.assert_allclose(online.to_numpy(dtype=float), expected.to_numpy(dtype=float), rtol=1e-5, equal_nan=True)
