"""Model training: logistic baseline, LightGBM with early stopping, MLflow logging."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud.data import TARGET
from fraud.features import is_categorical

REPO_DIR = Path(__file__).resolve().parents[2]
MLFLOW_DB = REPO_DIR / "mlflow.db"
MLFLOW_ARTIFACTS = REPO_DIR / "mlartifacts"

# Fixed configuration used in M1 to compare split strategies.
BASELINE_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.5,
    verbose=-1,
)


def fit_baseline(train: pd.DataFrame, features: list[str], seed: int = 0) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**BASELINE_PARAMS, random_state=seed)
    model.fit(train[features], train[TARGET])
    return model


def pr_auc(model, test: pd.DataFrame, features: list[str]) -> float:
    return average_precision_score(test[TARGET], predict(model, test, features))


def predict(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(df[features])[:, 1]


def fit_lgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    cfg: dict,
    num_leaves: int | None = None,
    seed: int = 0,
    sample_weight=None,
    overrides: dict | None = None,
) -> lgb.LGBMClassifier:
    """LightGBM with early stopping on validation PR-AUC (unweighted, also when training is weighted).

    `overrides` replaces any LightGBM parameter from the config, e.g. during tuning.
    """
    params = {k: v for k, v in cfg.items() if k not in ("early_stopping_rounds", "num_leaves_grid", "num_leaves")}
    params["num_leaves"] = num_leaves or cfg["num_leaves"]
    params |= overrides or {}
    model = lgb.LGBMClassifier(**params, metric="average_precision", random_state=seed, verbose=-1)
    model.fit(
        train[features], train[TARGET],
        sample_weight=sample_weight,
        eval_X=(valid[features],),
        eval_y=(valid[TARGET],),
        callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)],
    )
    return model


def _signed_log(x):
    # Counts and amounts are heavy-tailed; a linear model needs them compressed.
    return np.sign(x) * np.log1p(np.abs(x))


def _as_str(x):
    return x.astype(str)


def fit_logistic(
    train: pd.DataFrame, features: list[str], C: float = 1.0, max_rows: int | None = None, seed: int = 0
) -> Pipeline:
    """Linear baseline: signed log, median imputation with missing flags, scaling, one-hot for categories."""
    if max_rows is not None and len(train) > max_rows:
        # A uniform row sample keeps the class balance, so probabilities stay unbiased.
        train = train.sample(max_rows, random_state=seed)
    categorical = [f for f in features if is_categorical(train[f])]
    numeric = [f for f in features if f not in categorical]
    preprocess = ColumnTransformer([
        ("num", make_pipeline(
            FunctionTransformer(_signed_log),
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
        ), numeric),
        ("cat", make_pipeline(
            FunctionTransformer(_as_str),
            OneHotEncoder(max_categories=20, handle_unknown="infrequent_if_exist"),
        ), categorical),
    ])
    model = make_pipeline(preprocess, LogisticRegression(C=C, max_iter=2000))
    model.fit(train[features], train[TARGET])
    return model


def log_run(name: str, params: dict, metrics: dict, features: list[str], experiment: str = "m2-modeling") -> None:
    """Record one model in the local MLflow store (mlflow.db + mlartifacts/)."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        # Pin artifacts to the repo; the default is relative to the working directory.
        mlflow.create_experiment(experiment, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_text("\n".join(features), "features.txt")
