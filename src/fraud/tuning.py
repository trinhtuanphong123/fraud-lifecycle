"""Tuning without touching the test weeks.

Every selection uses weeks before the test start only (checked by assert).
Inside each validation window, card holders are split in two: one half drives
early stopping, the other half scores the configuration, so no configuration
is judged on the rows that decided when it stopped training.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np
import pandas as pd

from fraud.config import load_config
from fraud.data import ENTITY_COL, TARGET
from fraud.evaluate import average_precision, summarize
from fraud.train import fit_lgbm, predict

TEST_START_WEEK = load_config("model")["split"]["test_start_week"]


class Fold(NamedTuple):
    name: str
    train_last_week: int
    valid_weeks: tuple[int, int]  # inclusive


def time_folds() -> list[Fold]:
    """Rolling-origin folds inside the tuning period (weeks 0-19)."""
    return [Fold("F1", 11, (12, 15)), Fold("F2", 15, (16, 19))]


def assert_no_test_weeks(df: pd.DataFrame) -> None:
    assert df["week"].max() < TEST_START_WEEK, "tuning must not see the test weeks"


def group_halves(groups, seed: int = 0) -> np.ndarray:
    """True for rows whose group (card holder) falls in the early-stopping half; groups never straddle halves."""
    codes, uniques = pd.factorize(pd.Series(groups), use_na_sentinel=False)
    in_first = np.random.default_rng(seed).random(len(uniques)) < 0.5
    return in_first[codes]


def recency_weights(week, half_life: float, last_week: int) -> np.ndarray:
    """Weight 1 for the last training week, halving every `half_life` weeks back."""
    return 0.5 ** ((last_week - np.asarray(week, dtype=float)) / half_life)


def split_fold(df: pd.DataFrame, fold: Fold, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, early-stopping half, evaluation half) for one fold."""
    train = df[df["week"] <= fold.train_last_week]
    valid = df[df["week"].between(*fold.valid_weeks)]
    assert_no_test_weeks(train)
    assert_no_test_weeks(valid)
    es_half = group_halves(valid[ENTITY_COL], seed)
    return train, valid[es_half], valid[~es_half]


def prepare_training(train: pd.DataFrame, es: pd.DataFrame, window: str, es_protocol: str):
    """Apply a training-window choice and an early-stopping protocol.

    window: "all", "drop_weeks_0_3", or "half_life_<w>" (recency weights).
    es_protocol: "all" (every card holder in the early-stopping half) or
    "unseen" (only card holders absent from training, so stopping does not
    reward recognising known people).
    """
    weight = None
    if window == "drop_weeks_0_3":
        train = train[train["week"] >= 4]
    elif window.startswith("half_life_"):
        weight = recency_weights(train["week"], float(window.rsplit("_", 1)[1]), int(train["week"].max()))
    elif window != "all":
        raise ValueError(window)
    if es_protocol == "unseen":
        es = es[~es[ENTITY_COL].isin(train[ENTITY_COL].unique())]
    elif es_protocol != "all":
        raise ValueError(es_protocol)
    return train, es, weight


def evaluate_config(df: pd.DataFrame, fold: Fold, features: list[str], lgbm_cfg: dict, overrides: dict | None = None,
                    window: str = "all", es_protocol: str = "all", seed: int = 0) -> dict:
    """Fit one configuration on one fold and score it on the evaluation half."""
    train, es, evaluation = split_fold(df, fold, seed)
    train, es, weight = prepare_training(train, es, window, es_protocol)
    t0 = time.time()
    model = fit_lgbm(train, es, features, lgbm_cfg, seed=seed, sample_weight=weight, overrides=overrides)
    score = predict(model, evaluation, features)
    y, amount = evaluation[TARGET].to_numpy(), evaluation["TransactionAmt"].to_numpy(dtype=float)
    return {
        "fold": fold.name,
        "pr_auc": average_precision(y, score),
        "usd_pr_auc": average_precision(y, score, amount),
        "log_loss": summarize(y, score)["log_loss"],
        "n_trees": model.best_iteration_,
        "seconds": round(time.time() - t0),
    }


def random_configs(space: dict[str, list], n: int, seed: int = 0) -> list[dict]:
    """n distinct random draws from a grid of parameter values."""
    rng = np.random.default_rng(seed)
    configs, seen = [], set()
    while len(configs) < n:
        config = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            configs.append({k: (v.item() if hasattr(v, "item") else v) for k, v in config.items()})
    return configs


def pick_simplest_within(results: pd.DataFrame, score: str, tolerance: float, complexity: list[str]):
    """Index of the simplest row whose score is within `tolerance` of the best (ties: higher score)."""
    best = results[score].max()
    close = results[results[score] >= best - tolerance]
    return close.sort_values(complexity + [score], ascending=[True] * len(complexity) + [False]).index[0]
