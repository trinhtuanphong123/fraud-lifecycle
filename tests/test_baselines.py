import numpy as np
import pandas as pd
import pytest

from fraud.baselines import fit_scorecard, score_scorecard


def test_single_attribute_scorecard_returns_the_smoothed_band_rate():
    bands = pd.DataFrame({"a": ["x"] * 10 + ["y"] * 90})
    y = np.r_[np.ones(5), np.zeros(5), np.ones(9), np.zeros(81)]  # x: 50% fraud, y: 10%
    model = fit_scorecard(bands, y, smoothing=1.0)
    p = score_scorecard(model, pd.DataFrame({"a": ["x", "y"]}))
    base = 14 / 100
    assert p[0] == pytest.approx((5 + base) / (10 + 1))
    assert p[1] == pytest.approx((9 + base) / (90 + 1))


def test_unseen_band_carries_no_evidence():
    bands = pd.DataFrame({"a": ["x", "x", "y", "y"]})
    model = fit_scorecard(bands, [1, 0, 0, 0])
    p = score_scorecard(model, pd.DataFrame({"a": ["never seen"]}))
    assert p[0] == pytest.approx(0.25)  # prior only


def test_evidence_adds_across_attributes():
    bands = pd.DataFrame({"a": ["x", "x", "y", "y"] * 25, "b": ["u", "v"] * 50})
    y = np.tile([1, 0, 0, 0], 25)
    model = fit_scorecard(bands, y)
    one = score_scorecard(model, pd.DataFrame({"a": ["x"], "b": ["u"]}))[0]
    logit = np.log(one / (1 - one))
    expected = model.prior_logit + model.woe["a"]["x"] + model.woe["b"]["u"]
    assert logit == pytest.approx(expected)
