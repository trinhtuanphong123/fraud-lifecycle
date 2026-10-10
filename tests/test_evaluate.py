import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraud.evaluate import average_precision, bootstrap_ci, paired_bootstrap_diff


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    n = 5_000
    y = (rng.random(n) < 0.05).astype(int)
    score = np.round(y * 0.3 + rng.random(n), 2)  # rounding creates ties
    return y, score


def test_average_precision_matches_sklearn_with_ties_and_weights(scored):
    y, score = scored
    w = np.random.default_rng(1).integers(0, 4, len(y))
    assert average_precision(y, score) == pytest.approx(average_precision_score(y, score))
    assert average_precision(y, score, w) == pytest.approx(average_precision_score(y, score, sample_weight=w))


def test_bootstrap_ci_contains_the_estimate(scored):
    y, score = scored
    ci = bootstrap_ci(y, score, n_boot=200)
    assert ci.low < ci.estimate < ci.high


def test_cluster_bootstrap_is_wider_when_rows_repeat_inside_groups(scored):
    y, score = scored
    base = slice(0, 500)
    # Each base row appears 20 times: 10,000 rows but only 500 independent units.
    y_rep, s_rep = np.repeat(y[base], 20), np.repeat(score[base], 20)
    groups = np.repeat(np.arange(500), 20)
    rows = bootstrap_ci(y_rep, s_rep, n_boot=300)
    clusters = bootstrap_ci(y_rep, s_rep, groups=groups, n_boot=300)
    assert clusters.high - clusters.low > 2 * (rows.high - rows.low)


def test_missing_group_labels_form_their_own_group():
    from fraud.evaluate import _bootstrap_weights

    groups = np.array(["a", "a", None, None, "b"], dtype=object)
    draws = list(_bootstrap_weights(groups, len(groups), 50, np.random.default_rng(0)))
    assert all(w[0] == w[1] and w[2] == w[3] for w in draws)
    # Regression: with code -1 the missing rows silently reused the last group's count.
    assert any(w[2] != w[4] for w in draws)


def test_paired_difference_of_identical_models_is_zero(scored):
    y, score = scored
    ci = paired_bootstrap_diff(y, score, score, n_boot=100)
    assert ci == (0.0, 0.0, 0.0)


def test_value_capture_at_rate():
    from fraud.evaluate import value_capture_at_rate

    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.1, 0.2])
    amount = np.array([30.0, 1.0, 70.0, 1.0])
    assert value_capture_at_rate(y, score, amount, 0.25) == pytest.approx(0.3)
    assert value_capture_at_rate(y, score, amount, 1.0) == pytest.approx(1.0)


def test_paired_difference_detects_a_better_model(scored):
    y, score = scored
    better = score + 0.5 * y
    ci = paired_bootstrap_diff(y, score, better, n_boot=200)
    assert ci.low > 0
