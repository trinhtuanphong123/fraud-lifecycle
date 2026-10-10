import numpy as np
import pandas as pd
import pytest

from fraud.backtest import label_end, retrain_schedule, simulate

LAUNCH, WEEKS = 12, range(12, 27)


def test_label_end_waits_for_the_delay_after_launch():
    assert label_end(12, LAUNCH, 4) == 11      # only the historical labels
    assert label_end(15, LAUNCH, 4) == 11
    assert label_end(16, LAUNCH, 4) == 12      # first new week matures at the end of week 16
    assert label_end(26, LAUNCH, 17) == 11     # 120-day labels: nothing new in this window


def test_never_keeps_the_launch_model():
    s = retrain_schedule("never", WEEKS, LAUNCH, 4)
    assert (s["model_end"] == 11).all() and not s["retrain"].any()


def test_weekly_retrains_only_when_new_labels_exist_and_takes_effect_next_week():
    s = retrain_schedule("weekly", WEEKS, LAUNCH, 4)
    assert not s.loc[12:15, "retrain"].any()     # no new labels before the end of week 16
    assert s.loc[16, "retrain"] and s.loc[16, "model_end"] == 11
    assert s.loc[17, "model_end"] == 12          # new model in force from the next week


def test_periodic_and_drift_schedules():
    periodic = retrain_schedule("periodic", WEEKS, LAUNCH, 2, period=4)
    assert periodic.index[periodic["retrain"]].tolist() == [15, 19, 23]
    drift = retrain_schedule("drift", WEEKS, LAUNCH, 4, alarm_weeks=[13, 14, 18, 22])
    assert drift.index[drift["retrain"]].tolist() == [18, 22]   # 13 and 14 had no new labels yet
    assert drift.loc[19, "model_end"] == 14 and drift.loc[23, "model_end"] == 18


@pytest.mark.parametrize("policy", ["never", "periodic", "drift", "weekly"])
@pytest.mark.parametrize("delay", [2, 4, 17])
def test_a_model_never_scores_a_week_it_was_trained_on(policy, delay):
    s = retrain_schedule(policy, WEEKS, LAUNCH, delay, alarm_weeks=range(12, 27))
    assert (s["model_end"] < s.index).all()


def test_simulate_uses_the_model_in_force():
    evaluation = pd.DataFrame({"TransactionID": [1, 2], "week": [12, 13], "day": [84, 91],
                               "TransactionAmt": [100.0, 100.0], "isFraud": [1, 1]})
    snapshots = {11: pd.DataFrame({"TransactionID": [1, 2], "score": [0.9, 0.0], "p": [0.9, 0.0]}),
                 12: pd.DataFrame({"TransactionID": [2], "score": [0.9], "p": [0.9]})}
    meta = pd.DataFrame({"lambda_bayes": [0.0, 0.0], "lambda_3action": [0.0, 0.0]}, index=[11, 12])
    costs = {"missed_fraud": {"amount_share": 1.0, "chargeback_fee": 15.0},
             "false_positive": {"fixed": 5.0, "margin_share": 0.0},
             "manual_review": {"cost_per_case": 3.0, "daily_capacity": 10, "catch_rate": 0.9}}
    schedule = pd.DataFrame({"model_end": [11, 12], "retrain": [True, False]}, index=pd.Index([12, 13], name="week"))
    weekly, cost = simulate(evaluation, snapshots, meta, schedule, costs)
    assert cost.tolist() == [0.0, 0.0]   # both frauds declined: week 13 used the retrained model
    stale = schedule.assign(model_end=[11, 11])
    _, cost_stale = simulate(evaluation, snapshots, meta, stale, costs)
    assert cost_stale.tolist() == [0.0, 115.0]
