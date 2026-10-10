import numpy as np
import pytest

from fraud.calibrate import brier_decomposition, expected_calibration_error, fit_isotonic, fit_platt


@pytest.fixture
def miscalibrated():
    rng = np.random.default_rng(0)
    true_p = rng.beta(0.5, 8, 20_000)
    y = (rng.random(len(true_p)) < true_p).astype(int)
    score = np.sqrt(true_p)  # same ranking, probabilities far too high
    return true_p, y, score


def test_brier_decomposition_is_exact_when_p_is_constant_per_bin():
    p = np.repeat([0.05, 0.35, 0.75], [600, 300, 100])
    rng = np.random.default_rng(1)
    y = (rng.random(len(p)) < p).astype(int)
    d = brier_decomposition(y, p, n_bins=10, strategy="uniform")
    assert d["residual"] == pytest.approx(0.0, abs=1e-12)
    assert d["brier"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"])


def test_calibrators_reduce_calibration_error_and_keep_ranking(miscalibrated):
    true_p, y, score = miscalibrated
    fit, test = slice(0, 10_000), slice(10_000, None)
    before = expected_calibration_error(y[test], score[test])
    for fitter in (fit_platt, fit_isotonic):
        cal = fitter(score[fit], y[fit])
        p = cal(score[test])
        assert expected_calibration_error(y[test], p) < before / 3
        assert np.all(np.diff(p[np.argsort(score[test])]) >= -1e-12)  # monotone in the score


def test_perfect_probabilities_have_small_ece(miscalibrated):
    true_p, y, _ = miscalibrated
    assert expected_calibration_error(y, true_p) < 0.01
