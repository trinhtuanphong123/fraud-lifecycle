# fraud-lifecycle

Fraud detection treated as a lifecycle, not a leaderboard: **decide** with costs, **monitor** before labels arrive, **retrain** when it pays off. Built on the IEEE-CIS (Vesta) e-commerce transactions dataset.

The project follows the research order **problem → data → analysis → method → implementation**, and the docs below follow the same order.

## 1. Problem

Vesta guarantees the payments it approves: every approved fraud becomes a chargeback Vesta pays, and every declined good customer is lost revenue. The goal is a decision system that **minimises total cost** per transaction and **keeps working over time**, although chargebacks (the labels) arrive up to ~120 days late.

| | Sub-problem | Success measure | Module |
|---|---|---|---|
| P1 | Score each transaction: p(fraud) | PR-AUC, calibration | M1, M2 |
| P2 | Decide: approve / review / decline | Total cost in $ | M3 |
| P3 | Detect degradation **before** labels arrive | Detection delay, false-alarm rate | M4 |
| P4 | Decide when to retrain | Total cost incl. retraining | M5 |

Full framing, constraints, why this dataset fits and what it cannot provide: [`docs/problem_framing.md`](docs/problem_framing.md).

## 2. Data

- **What is in the data**, how it is missing, how it is distributed, and how test differs from train: [`docs/data_dictionary.md`](docs/data_dictionary.md), reproduced by `notebooks/00_data_overview.ipynb`.
- **What the data says about the problem**: where the money is lost, where the signal is, what changes over time, how labels are structured: [`reports/data_analysis.md`](reports/data_analysis.md), reproduced by `notebooks/00b_problem_data_analysis.ipynb`.

The labelled train file covers about 6 months, so monitoring runs **weekly**. The unlabelled Kaggle test file (the following ~6 months) is used as a "production" period where only label-free signals exist.

To get the data:
1. Accept the competition rules at <https://www.kaggle.com/competitions/ieee-fraud-detection>.
2. Download and unzip into `data/raw/`:
   ```
   kaggle competitions download -c ieee-fraud-detection -p data/raw
   ```
   Expected files: `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`.

## 3. Method

Candidate methods, the choice for each sub-problem, and the evidence behind it: [`docs/methodology.md`](docs/methodology.md). First, untuned results of every candidate: [`reports/method_research.md`](reports/method_research.md). Tuning, with the test weeks used only once: [`reports/tuning.md`](reports/tuning.md). Hypotheses are written down before each experiment: [`docs/hypotheses.md`](docs/hypotheses.md).

In short: time-based splits and card-holder bootstrap CIs everywhere; LightGBM with past-only behavioural features (P1); calibrated probabilities and a Bayes cost rule (P2); permutation-null PSI/JS with FDR control, block-level null rates, score drift, CBPE and a domain classifier (P3); a retraining backtest under label delay (P4).

## 4. Results so far

| Question | Answer |
|---|---|
| Does a random split mislead? | Yes. PR-AUC 0.766 [0.751, 0.780] with a random split vs 0.567 [0.537, 0.599] split by time (M1). |
| Best scoring model | Tuned LightGBM with past-only features and recency weights: PR-AUC 0.545 [0.515, 0.574] on the last 7 weeks; +0.012 [0.003, 0.020] over the untuned model, +0.14 over logistic regression. |
| Are its probabilities usable as is? | No. Raw scores underestimate risk (mean 1.5% vs 3.45% observed); Platt calibration brings ECE to 0.002. |
| At what threshold should we decline to minimise total loss? | A per-transaction cost rule on calibrated probabilities, capped at 1% of good customers declined: saves $310k [263k, 365k] on the last 7 weeks vs $148k for threshold 0.5, and $44k [22k, 70k] more than a score threshold at the same decline rate. With manual review, $478k. |
| How do we know the model is degrading **before** chargebacks arrive? | Four tuned label-free rules (never-seen values, PSI column count, V-block missing rate, domain classifier) flag every known data change and raise no alarm in 10 clean weeks. Prior and concept shift still need labels. |
| When should we retrain? | Whenever new labels mature. In a 15-week replay with a 4-week label delay, weekly retraining saved $134k [91k, 180k] vs never retraining, more than every-4-weeks ($91k) or drift-triggered ($114k) retraining; a retrain pays off if it costs under ~$12k. Drift alarms did not precede performance drops, so they serve for investigation, not as the trigger. With 120-day labels no retraining is possible for months: faster labels (review outcomes) are needed. |

All intervals are 95% bootstrap CIs over card holders.

## 5. Serving

A FastAPI service scores one raw transaction. It returns the calibrated fraud probability, the decision under the capped cost rule with the transaction's own threshold, a review suggestion and, for declines, the top-3 reasons from TreeSHAP. Details: [`reports/serving.md`](reports/serving.md).

- **Training-serving parity is tested on real data.** Past-only features computed online match the batch features on 2,000 transactions, and API scores match the offline pipeline on 200 transactions (within 1e-5).
- **Latency in Docker** on 1,000 real test-period transactions: p50 53 ms, p95 65 ms (declines, which are explained, at p99 ≈ 0.65 s); 424 MiB of memory.
- The model is registered in MLflow as `fraud-scorer`.

```
uv run python -m fraud.serving.bundle        # build artifacts/bundle from the tuned model, register in MLflow
docker build -t fraud-scorer:1.0 . && docker run -p 8000:8000 fraud-scorer:1.0
curl -X POST "localhost:8000/score?explain=always" -H "Content-Type: application/json" \
     -d '{"TransactionID": 1, "TransactionDT": 18500000, "TransactionAmt": 250.0, "card1": 9500, "ProductCD": "C"}'
```

## Quickstart

```
uv sync
uv run pytest
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_overview.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00b_problem_data_analysis.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_and_time_split.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_modeling.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/03_method_scoring.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_method_decision.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/05_method_monitoring.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/06_tuning_scoring.ipynb   # ~2 h, resumable
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_tuning_decision.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/08_tuning_monitoring.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/09_backtest.ipynb   # ~1 h, resumable
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Layout

```
configs/      cost, monitoring and model settings
docs/         problem framing, data dictionary, methodology, pre-registered hypotheses
notebooks/    one story per notebook; logic lives in src/
reports/      data analysis, method research, tuning, backtest and serving reports, figures
src/fraud/    data.py, profiling.py, analysis.py, features.py, baselines.py, train.py, evaluate.py,
              calibrate.py, decision.py, drift.py, tuning.py, backtest.py, config.py
src/fraud/serving/   bundle.py, history.py, scorer.py, reasons.py, app.py (FastAPI)
Dockerfile    scoring API image (runtime deps in requirements-serve.txt)
tests/        leakage and correctness tests
```

## Roadmap

| Module | Content | Status |
|---|---|---|
| M0 | Data overview: files, columns, missingness blocks, distributions, train vs. test drift; problem-oriented data analysis | done |
| M1 | Loading, weekly time split, random vs. time vs. entity-disjoint split | done |
| M2 | Past-only features, MI screening vs. permutation null, logistic vs. LightGBM, PR-AUC with paired cluster-bootstrap CIs | done |
| M3 | Calibration, Bayes-optimal cost policy with a decline-rate cap, cost sensitivity, 3-action policy | done (tuned) |
| M4 | PSI test + BH-FDR, block null rates, unseen-value share, domain classifier; real and injected drift | done (tuned); CBPE preliminary |
| M5 | Retraining policy backtest with 2-, 4- and 17-week label delay | done |
| M6 | FastAPI + TreeSHAP reason codes + Docker + MLflow registry, with training-serving parity tests | done |

## Assumptions and limitations

- The real start date of `TransactionDT` is unpublished; `2017-12-01` is assumed. Only relative time matters.
- `entity_key` (card1 + addr1 + card issue day) approximates a card holder. The key itself is never a model input; it is used to measure leakage, to group bootstrap resamples, and to compute past-only velocity features.
- Costs are assumptions (`configs/costs.yaml`); cost conclusions are reported across a sensitivity grid.
- The data likely contains only transactions Vesta processed, not those its existing system declined, and blocked transactions never receive labels (feedback loop). The offline backtest cannot fully reproduce this.
