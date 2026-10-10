"""The serving bundle: everything the API needs, built once from the research outputs.

    uv run python -m fraud.serving.bundle            # build artifacts/bundle and register it in MLflow
    uv run python -m fraud.serving.bundle --no-register

Contents: model.txt (tuned LightGBM), features.json (feature order, raw column
dtypes), calibration.json (Platt a, b), decision.json (cost table, decline-cap
penalty), history_*.parquet (online history state after the training data).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud.calibrate import fit_platt_params, platt
from fraud.data import ENTITY_COL, PROCESSED_DIR, TARGET
from fraud.decision import bayes_policy, load_costs, penalty_for_cap
from fraud.features import add_past_features, add_row_features, feature_columns
from fraud.serving.history import HistoryStore

REPO_DIR = Path(__file__).resolve().parents[3]
BUNDLE_DIR = REPO_DIR / "artifacts" / "bundle"
DERIVED = {"timestamp", "day", "week", "month", ENTITY_COL, TARGET}


@dataclass
class Bundle:
    booster: lgb.Booster
    features: list[str]
    raw_dtypes: dict[str, str]
    platt_a: float
    platt_b: float
    fp_penalty: float
    costs: dict
    version: str


def build_bundle(out_dir: Path, booster: lgb.Booster, raw: pd.DataFrame, valid_score, valid_y, valid_amount,
                 costs: dict, cap: float, version: str) -> Path:
    """Write a bundle. `raw` is the training table before feature engineering (dtypes and history come from it)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = feature_columns(add_past_features(add_row_features(raw.head(5))))
    assert booster.feature_name() == expected, "model features differ from the feature pipeline"

    booster.save_model(str(out_dir / "model.txt"))
    raw_dtypes = {c: str(raw[c].dtype) for c in raw.columns if c not in DERIVED}
    (out_dir / "features.json").write_text(json.dumps({"features": expected, "raw_dtypes": raw_dtypes}, indent=1))

    a, b = fit_platt_params(valid_score, valid_y)
    p = platt(valid_score, a, b)
    amount = np.asarray(valid_amount, dtype=float)
    lam = penalty_for_cap(lambda l: bayes_policy(p, amount, costs, fp_penalty=l), np.asarray(valid_y), cap)
    (out_dir / "calibration.json").write_text(json.dumps({"method": "platt", "a": a, "b": b}, indent=1))
    (out_dir / "decision.json").write_text(json.dumps(
        {"version": version, "max_good_decline_rate": cap, "fp_penalty": lam, "costs": costs}, indent=1))
    HistoryStore.save_frame(raw, out_dir)
    return out_dir


def load_bundle(directory: Path = BUNDLE_DIR) -> Bundle:
    directory = Path(directory)
    feats = json.loads((directory / "features.json").read_text())
    cal = json.loads((directory / "calibration.json").read_text())
    dec = json.loads((directory / "decision.json").read_text())
    return Bundle(
        booster=lgb.Booster(model_file=str(directory / "model.txt")),
        features=feats["features"], raw_dtypes=feats["raw_dtypes"],
        platt_a=cal["a"], platt_b=cal["b"], fp_penalty=dec["fp_penalty"], costs=dec["costs"], version=dec["version"],
    )


def register(out_dir: Path, booster: lgb.Booster, version: str) -> None:
    """Log the bundle to MLflow and register the model as `fraud-scorer`."""
    import mlflow

    from fraud.train import MLFLOW_ARTIFACTS, MLFLOW_DB

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    if mlflow.get_experiment_by_name("serving") is None:
        mlflow.create_experiment("serving", artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment("serving")
    with mlflow.start_run(run_name=f"bundle {version}"):
        mlflow.lightgbm.log_model(booster, name="model", registered_model_name="fraud-scorer")
        for f in ["features.json", "calibration.json", "decision.json"]:
            mlflow.log_artifact(str(Path(out_dir) / f), artifact_path="bundle")
        mlflow.log_params({"version": version, **json.loads((Path(out_dir) / "calibration.json").read_text())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=BUNDLE_DIR)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    raw = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    booster = lgb.Booster(model_file=str(PROCESSED_DIR / "model_tuned.txt"))
    scores = pd.read_parquet(PROCESSED_DIR / "scores_tuned.parquet")
    valid = scores[scores["split"] == "valid_scoring_half"]
    costs = load_costs()
    version = "tuned-" + pd.Timestamp.now().strftime("%Y%m%d")
    build_bundle(args.out, booster, raw, valid["score"], valid[TARGET], valid["TransactionAmt"],
                 costs, costs["constraints"]["max_good_decline_rate"], version)
    print(f"bundle written to {args.out}")
    if not args.no_register:
        register(args.out, booster, version)
        print("registered in MLflow as 'fraud-scorer'")


if __name__ == "__main__":
    main()
