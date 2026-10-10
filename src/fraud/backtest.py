"""Replay the labelled weeks as if the system were live, to compare retraining policies.

Timeline conventions (one step = one week):
- the system launches at week `launch` with a model trained on weeks 0..launch-1,
  whose labels are assumed mature (historical data);
- a transaction from week w gets its label at the end of week w + delay;
- retraining happens at the end of week t, with labels up to week
  label_end(t) = max(launch - 1, t - delay); the new model scores from week t + 1.
So a model never scores a week it was trained or calibrated on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.calibrate import fit_platt
from fraud.data import ENTITY_COL, TARGET
from fraud.decision import (bayes_policy, penalty_for_cap, policy_summary, realized_cost, three_action_policy)
from fraud.evaluate import average_precision
from fraud.train import fit_lgbm, predict
from fraud.tuning import group_halves, recency_weights

POLICIES = ("never", "periodic", "drift", "weekly")


def label_end(t: int, launch: int, delay: int) -> int:
    """Last week whose labels are known at the end of week t."""
    return max(launch - 1, t - delay)


def retrain_schedule(policy: str, weeks, launch: int, delay: int, alarm_weeks=(), period: int = 4) -> pd.DataFrame:
    """Per week: the label end of the model in force, and whether the policy retrains at the end of the week.

    A retrain only happens if labels newer than the current model exist.
    """
    current = launch - 1
    rows = []
    for t in weeks:
        due = {
            "never": False,
            "periodic": (t - launch + 1) % period == 0,
            "drift": t in set(alarm_weeks),
            "weekly": True,
        }[policy]
        available = label_end(t, launch, delay)
        retrain = bool(due and available > current)
        rows.append({"week": t, "model_end": current, "retrain": retrain})
        if retrain:
            current = available
    return pd.DataFrame(rows).set_index("week")


def fit_snapshot(df: pd.DataFrame, end: int, features: list[str], lgbm_cfg: dict, overrides: dict, costs: dict,
                 cap: float, half_life: float = 4.0, seed: int = 0) -> dict:
    """Train the model a live system would have built with labels up to week `end`.

    Weeks 0..end-2 train it (recency-weighted); weeks end-1..end are split by
    card holder into an early-stopping half and a calibration half, on which
    Platt scaling and the decline-cap penalties are fitted.
    """
    train = df[df["week"] <= end - 2]
    holdout = df[df["week"].between(end - 1, end)]
    es_half = group_halves(holdout[ENTITY_COL], seed)
    es, cal = holdout[es_half], holdout[~es_half]
    weight = recency_weights(train["week"], half_life, end - 2)
    model = fit_lgbm(train, es, features, lgbm_cfg, seed=seed, sample_weight=weight, overrides=overrides)

    s_cal = predict(model, cal, features)
    calibrate = fit_platt(s_cal, cal[TARGET])
    p_cal = calibrate(s_cal)
    y_cal, amt_cal, day_cal = cal[TARGET].to_numpy(), cal["TransactionAmt"].to_numpy(dtype=float), cal["day"].to_numpy()
    lam = penalty_for_cap(lambda l: bayes_policy(p_cal, amt_cal, costs, fp_penalty=l), y_cal, cap)

    # The calibration half carries a fraction of live daily volume; scale the review capacity accordingly.
    costs_cal = {**costs, "manual_review": dict(costs["manual_review"])}
    costs_cal["manual_review"]["daily_capacity"] = max(1, int(round(costs["manual_review"]["daily_capacity"] * len(cal) / len(holdout))))
    lam3 = penalty_for_cap(lambda l: three_action_policy(p_cal, amt_cal, day_cal, costs_cal, fp_penalty=l), y_cal, cap)

    scored = df[df["week"] > end]
    s = predict(model, scored, features)
    return {
        "scores": pd.DataFrame({"TransactionID": scored["TransactionID"].to_numpy(), "week": scored["week"].to_numpy(),
                                "score": s, "p": calibrate(s)}),
        "meta": {"end": end, "n_trees": int(model.best_iteration_), "train_rows": len(train),
                 "lambda_bayes": lam, "lambda_3action": lam3},
    }


def simulate(evaluation: pd.DataFrame, snapshots: dict[int, pd.DataFrame], meta: pd.DataFrame,
             schedule: pd.DataFrame, costs: dict, rule: str = "bayes") -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the decision rule week by week with the model the schedule has in force.

    `evaluation` holds the live weeks (TransactionID, week, day, amount, label);
    `snapshots[end]` holds each model's calibrated scores; `meta` its penalties.
    Returns a weekly summary and the realised cost of every transaction (in
    `evaluation` order), for paired comparisons between policies.
    """
    cost = np.zeros(len(evaluation))
    rows = []
    for t, row in schedule.iterrows():
        mask = (evaluation["week"] == t).to_numpy()
        if not mask.any():
            continue
        week = evaluation[mask]
        s = snapshots[row["model_end"]].set_index("TransactionID").loc[week["TransactionID"]]
        p, y = s["p"].to_numpy(), week[TARGET].to_numpy()
        amount, day = week["TransactionAmt"].to_numpy(dtype=float), week["day"].to_numpy()
        if rule == "bayes":
            action = bayes_policy(p, amount, costs, fp_penalty=meta.loc[row["model_end"], "lambda_bayes"])
        else:
            action = three_action_policy(p, amount, day, costs, fp_penalty=meta.loc[row["model_end"], "lambda_3action"])
        cost[mask] = realized_cost(action, y, amount, costs)
        summary = policy_summary(action, y, amount, costs)
        rows.append({"week": t, "model_end": row["model_end"], "model_age": t - row["model_end"], "retrain": row["retrain"],
                     "cost": summary["total_cost"], "good_customers_declined": summary["good_customers_declined"],
                     "fraud_usd_stopped": summary["fraud_usd_stopped"], "pr_auc": average_precision(y, s["score"].to_numpy()),
                     "n": len(week)})
    return pd.DataFrame(rows).set_index("week"), cost
