import numpy as np
import pandas as pd
import pytest

from fraud.drift import benjamini_hochberg, categorical_shift, null_rate_pvalue, psi, psi_pvalue


def test_psi_is_zero_for_identical_samples_and_grows_with_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5_000)
    assert psi(x, x)[0] == pytest.approx(0.0, abs=1e-9)
    small, _ = psi(x, x + 0.1)
    large, _ = psi(x, x + 1.0)
    assert 0 < small < large


def test_psi_pvalues_are_roughly_uniform_when_nothing_changes():
    rng = np.random.default_rng(1)
    pvalues = []
    for _ in range(400):
        a, b = rng.exponential(size=4_000), rng.exponential(size=2_000)
        value, k = psi(a, b)
        pvalues.append(psi_pvalue(value, len(a), len(b), k))
    pvalues = np.array(pvalues)
    assert 0.02 < (pvalues < 0.05).mean() < 0.09
    assert 0.4 < np.median(pvalues) < 0.6


def test_benjamini_hochberg_textbook_example():
    # Benjamini & Hochberg (1995) style example: 4 smallest pass at alpha = 0.05.
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False, False, False, False, False, False]
    p2 = [0.01, 0.02, 0.03, 0.04]
    assert benjamini_hochberg(p2, 0.05).all()


def test_categorical_shift_measures_new_values():
    ref = pd.Series(["a"] * 800 + ["b"] * 200)
    same = categorical_shift(ref, ref)
    assert same["js"] == pytest.approx(0.0, abs=1e-12) and same["new_share"] == 0.0
    cur = pd.Series(["a"] * 640 + ["b"] * 160 + ["c"] * 200)  # 20% new value
    shifted = categorical_shift(ref, cur)
    assert shifted["new_share"] == pytest.approx(0.2)
    assert shifted["js"] > 0.05 and shifted["pvalue"] < 1e-10


def test_inject_drift_scenarios():
    from fraud.drift import inject_drift

    rng = np.random.default_rng(0)
    n = 10_000
    week = pd.DataFrame({
        "isFraud": (rng.random(n) < 0.05).astype(int),
        "TransactionAmt": rng.exponential(100, n).astype(np.float32),
        "C1": rng.integers(0, 5, n).astype(np.float32),
        "V1": rng.normal(size=n), "V2": rng.normal(size=n),
        "id_31": pd.Categorical(rng.choice(["chrome", "safari", None], n)),
    })
    piped = inject_drift(week, "pipeline", 0.3, rng, block=["V1", "V2"])
    assert piped["V1"].isna().mean() == pytest.approx(0.3, abs=0.02)
    assert (piped["V1"].isna() == piped["V2"].isna()).all()
    new = inject_drift(week, "new_category", 0.15, rng)
    present = week["id_31"].notna()
    assert (new.loc[present, "id_31"] == "injected browser 1.0").mean() == pytest.approx(0.15, abs=0.02)
    assert new["id_31"].isna().sum() == week["id_31"].isna().sum()
    prior = inject_drift(week, "prior", 1.0, rng)
    assert prior["isFraud"].sum() == 2 * week["isFraud"].sum()
    shifted = inject_drift(week, "covariate", 0.5, rng)
    assert shifted["TransactionAmt"].sum() == pytest.approx(1.5 * week["TransactionAmt"].sum(), rel=1e-4)


def test_alarm_thresholds_from_clean_weeks():
    from fraud.drift import apply_alarm_rules, choose_thresholds

    clean = pd.DataFrame({"max_new_share": [0.02, 0.03, 0.05, 0.04], "max_block_shift": [0.01, 0.05, 0.02, 0.03]})
    thresholds = choose_thresholds(clean, max_false_alarm=0.0, margin=1.1)
    assert thresholds["new_values"] == pytest.approx(0.055)
    alarms = apply_alarm_rules(pd.DataFrame({"max_new_share": [0.05, 0.2], "max_block_shift": [0.0, 0.0]}), thresholds)
    assert alarms["any"].tolist() == [False, True]


def test_null_rate_pvalue():
    assert null_rate_pvalue(100, 1_000, 100, 1_000) == pytest.approx(1.0)
    assert null_rate_pvalue(100, 1_000, 300, 1_000) < 1e-10
