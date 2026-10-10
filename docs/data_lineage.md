# Data lineage: generated files

Nothing below is committed: `data/`, `artifacts/`, `mlflow.db` and `mlartifacts/` are in `.gitignore`. This page says which step creates each file, what reads it, and what it costs to rebuild. Sizes are from a full run.

## Order of creation

```
data/raw (Kaggle download)
  └─ train.parquet, test.parquet ─────────────── any notebook (load_dataset)
       ├─ 02 → scores_m2 ─→ 04 → scores_p2 ─→ 05
       ├─ 03 → scores_p1 ─────────────────────→ 06 (reference model S4)
       ├─ 06 → model_tuned, scores_tuned ─────→ 07, fraud.serving.bundle → artifacts/bundle → API / Docker
       ├─ 08 → tuning_monitor_replay ─────────→ 09, 10
       └─ 09 → backtest_snapshots/ ───────────→ 10
```

## Files

| File | Size | Created by | Read by | Rebuild |
|---|---|---|---|---|
| `data/raw/*.csv` | 1.29 GB | Kaggle download (see README) | `data.load_raw` | manual download |
| `data/processed/train.parquet` | 87 MB | `data.load_dataset("train")`, first call from any notebook | every notebook, `fraud.serving.bundle`, data tests | ~2 min |
| `data/processed/test.parquet` | 75 MB | `data.load_dataset("test")` (notebook 00) | notebook 00 (train vs test), API latency checks | ~2 min |
| `data/processed/scores_m2.parquet` | 5 MB | notebook 02 | notebook 04 | ~25 min (notebook 02) |
| `data/processed/scores_p1.parquet` | 10 MB | notebook 03 | notebook 06 (S4 baseline for the paired test comparison) | ~20 min |
| `data/processed/scores_p2.parquet` | 7 MB | notebook 04 | notebook 05 (CBPE section) | < 1 min |
| `data/processed/tuning_stage_a.parquet`, `tuning_stage_b.parquet` | < 1 MB | notebook 06, one row per fit (resume cache) | notebook 06 | ~2 h; deleting forces a full re-run |
| `data/processed/model_tuned.txt` | 65 MB | notebook 06 | `fraud.serving.bundle` | with notebook 06 |
| `data/processed/scores_tuned.parquet` | 4 MB | notebook 06 | notebook 07, `fraud.serving.bundle`, serving parity test | with notebook 06 |
| `data/processed/tuned_config.json` | < 1 KB | notebook 06 | nothing (record; the settings live in `configs/model.yaml → tuned`) | with notebook 06 |
| `data/processed/tuning_monitor_runs.parquet` | < 1 MB | notebook 08 (resume cache) | notebook 08 | ~35 min |
| `data/processed/tuning_monitor_replay.parquet` | < 1 MB | notebook 08 | notebooks 09, 10 (drift alarms) | with notebook 08 |
| `data/processed/backtest_snapshots/` | 50 MB | notebook 09, one model per label-end week (resume cache) | notebooks 09, 10 | ~1 h |
| `artifacts/bundle/` | 71 MB | `uv run python -m fraud.serving.bundle` | API, Docker image, `fraud.serving.cap`, serving parity test | ~2 min |
| `mlflow.db`, `mlartifacts/` | 1 MB, 65 MB | `fraud.train.log_run` (notebooks 02, 03, 06), `fraud.serving.bundle` (registry: `fraud-scorer`) | `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` | re-run the notebooks; registry: re-run the bundle |

Notebook 02 also writes `model_m2.txt` and `features_m2.json`. No later step reads them (notebook 04 needs only `scores_m2.parquet`), so they can be deleted.

## What can be deleted safely

- **Resume caches** (`tuning_stage_*.parquet`, `tuning_monitor_runs.parquet`, `backtest_snapshots/`): only to force a fresh run. Notebooks 06, 08 and 09 then recompute everything, which takes hours.
- **Everything in `data/processed/`**: it is all derived from `data/raw/`. Re-run the notebooks in order (README → Quickstart).
- **`artifacts/bundle/`**: rebuilt in ~2 minutes from `model_tuned.txt` and `scores_tuned.parquet`.
