import numpy as np
import pytest

from fraud.decision import (
    APPROVE,
    DECLINE,
    REVIEW,
    bayes_policy,
    load_costs,
    paired_cost_diff,
    policy_summary,
    realized_cost,
    three_action_policy,
)

COSTS = {
    "missed_fraud": {"amount_share": 1.0, "chargeback_fee": 15.0},
    "false_positive": {"fixed": 5.0, "margin_share": 0.0},
    "manual_review": {"cost_per_case": 3.0, "daily_capacity": 2, "catch_rate": 0.9},
}


def test_bayes_threshold_falls_with_amount():
    # C_FN = amount + 15, C_FP = 5  ->  decline iff p > 5 / (5 + amount + 15)
    amount = np.array([5.0, 5.0, 980.0, 980.0])
    p = np.array([0.19, 0.21, 0.004, 0.006])  # thresholds: 0.2 and 0.005
    assert bayes_policy(p, amount, COSTS).tolist() == [APPROVE, DECLINE, APPROVE, DECLINE]


def test_realized_cost_by_action():
    action = np.array([APPROVE, APPROVE, DECLINE, DECLINE, REVIEW, REVIEW])
    y = np.array([1, 0, 1, 0, 1, 0])
    amount = np.full(6, 100.0)
    assert realized_cost(action, y, amount, COSTS).tolist() == pytest.approx([115, 0, 0, 5, 3 + 0.1 * 115, 3])


def test_three_action_policy_respects_daily_capacity():
    # p = 0.05, amount 100: approve 0.05*115 = 5.75, decline 0.95*5 = 4.75,
    # review 3 + 0.1*0.05*115 = 3.575 -> review is cheapest.
    p = np.full(6, 0.05)
    amount = np.full(6, 100.0)
    day = np.array([1, 1, 1, 2, 2, 2])
    action = three_action_policy(p, amount, day, COSTS)
    for d in (1, 2):
        assert (action[day == d] == REVIEW).sum() == 2
    # Over capacity, the next best action is DECLINE (4.75 < 5.75).
    assert set(action[action != REVIEW]) == {DECLINE}


def test_three_action_policy_skips_review_when_it_does_not_pay():
    # p = 0.3: decline costs 0.7*5 = 3.5 < review 3 + 0.1*0.3*115 = 6.45.
    action = three_action_policy(np.full(3, 0.3), np.full(3, 100.0), np.ones(3), COSTS)
    assert (action == DECLINE).all()


def test_policy_summary_counts_stopped_fraud_dollars():
    action = np.array([DECLINE, APPROVE, REVIEW])
    y = np.array([1, 1, 1])
    amount = np.array([100.0, 100.0, 200.0])
    s = policy_summary(action, y, amount, COSTS)
    assert s["fraud_usd_stopped"] == pytest.approx((100 + 0.9 * 200) / 400)
    assert s["fraud_loss"] == pytest.approx(115 + 0.1 * 215)


def test_penalty_and_threshold_meet_the_decline_cap():
    from fraud.decision import good_decline_rate, penalty_for_cap, threshold_for_cap, threshold_policy

    rng = np.random.default_rng(0)
    p = rng.beta(0.5, 6, 20_000)
    y = (rng.random(len(p)) < p).astype(int)
    amount = rng.exponential(100, len(p))
    unconstrained = good_decline_rate(bayes_policy(p, amount, COSTS), y)
    cap = unconstrained / 4
    lam = penalty_for_cap(lambda lam: bayes_policy(p, amount, COSTS, fp_penalty=lam), y, cap)
    rate = good_decline_rate(bayes_policy(p, amount, COSTS, fp_penalty=lam), y)
    assert lam > 0 and rate <= cap and rate > cap * 0.9   # tight, not just feasible
    t = threshold_for_cap(p, y, cap)
    assert good_decline_rate(threshold_policy(p, t), y) <= cap


def test_expected_good_decline_rate_on_a_hand_example():
    from fraud.decision import expected_good_decline_rate

    action = np.array([DECLINE, APPROVE, APPROVE, DECLINE])
    p = np.array([0.8, 0.1, 0.0, 0.5])
    # declined good customers expected: 0.2 + 0.5 = 0.7, good customers expected: 0.2 + 0.9 + 1 + 0.5 = 2.6
    assert expected_good_decline_rate(action, p) == pytest.approx(0.7 / 2.6)


def test_label_free_penalty_meets_the_expected_cap_and_tracks_the_true_one_when_calibrated():
    from fraud.decision import expected_good_decline_rate, good_decline_rate, penalty_for_expected_cap

    rng = np.random.default_rng(3)
    p = rng.beta(0.5, 6, 50_000)
    y = (rng.random(len(p)) < p).astype(int)   # p is calibrated by construction
    amount = rng.exponential(100, len(p))
    lam = penalty_for_expected_cap(p, amount, COSTS, 0.01)
    action = bayes_policy(p, amount, COSTS, fp_penalty=lam)
    assert expected_good_decline_rate(action, p) <= 0.01
    assert good_decline_rate(action, y) == pytest.approx(0.01, abs=0.0015)


def test_penalty_is_zero_when_the_cap_already_holds():
    from fraud.decision import penalty_for_cap

    p, y, amount = np.array([0.01, 0.9]), np.array([0, 1]), np.array([10.0, 10.0])
    assert penalty_for_cap(lambda lam: bayes_policy(p, amount, COSTS, fp_penalty=lam), y, 0.5) == 0.0


def test_paired_cost_diff_of_identical_policies_is_zero():
    cost = np.arange(10.0)
    assert paired_cost_diff(cost, cost, n_boot=50) == (0.0, 0.0, 0.0)


def test_load_costs_overrides_do_not_touch_the_file():
    a = load_costs(fp_fixed=50)
    b = load_costs()
    assert a["false_positive"]["fixed"] == 50
    assert b["false_positive"]["fixed"] == 5.0
