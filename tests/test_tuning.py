import numpy as np
import pandas as pd
import pytest

from fraud.data import ENTITY_COL
from fraud.tuning import (
    TEST_START_WEEK,
    assert_no_test_weeks,
    group_halves,
    pick_simplest_within,
    prepare_training,
    random_configs,
    recency_weights,
    time_folds,
)


def test_recency_weight_halves_at_the_half_life():
    w = recency_weights([15, 11, 7], half_life=4, last_week=15)
    assert w.tolist() == pytest.approx([1.0, 0.5, 0.25])


def test_group_halves_never_split_a_card_holder():
    groups = np.repeat(np.arange(500), 4)
    half = group_halves(groups, seed=1)
    per_group = pd.Series(half).groupby(groups).nunique()
    assert (per_group == 1).all()
    assert 0.4 < half.mean() < 0.6


def test_folds_and_guard_stay_before_the_test_weeks():
    for fold in time_folds():
        assert fold.valid_weeks[1] < TEST_START_WEEK
    assert_no_test_weeks(pd.DataFrame({"week": [0, TEST_START_WEEK - 1]}))
    with pytest.raises(AssertionError):
        assert_no_test_weeks(pd.DataFrame({"week": [TEST_START_WEEK]}))


def test_unseen_protocol_drops_known_card_holders_from_early_stopping():
    train = pd.DataFrame({"week": [0, 5], ENTITY_COL: ["a", "b"]})
    es = pd.DataFrame({"week": [16, 16, 16], ENTITY_COL: ["a", "c", "d"]})
    _, es_unseen, weight = prepare_training(train, es, "all", "unseen")
    assert es_unseen[ENTITY_COL].tolist() == ["c", "d"] and weight is None
    tr, _, _ = prepare_training(train, es, "drop_weeks_0_3", "all")
    assert tr["week"].tolist() == [5]


def test_random_configs_are_distinct_and_from_the_space():
    space = {"a": [1, 2, 3], "b": [10, 20]}
    configs = random_configs(space, 6, seed=0)
    assert len({tuple(sorted(c.items())) for c in configs}) == 6
    assert all(c["a"] in space["a"] and c["b"] in space["b"] for c in configs)


def test_pick_simplest_within_tolerance():
    results = pd.DataFrame({"score": [0.540, 0.537, 0.520], "num_leaves": [255, 63, 31]})
    assert pick_simplest_within(results, "score", 0.005, ["num_leaves"]) == 1
    assert pick_simplest_within(results, "score", 0.001, ["num_leaves"]) == 0
