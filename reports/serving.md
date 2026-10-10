# Serving: one transaction in, decision and reasons out (M6)

The ONLINE half of the architecture: a FastAPI service that scores a single transaction with the tuned model, applies the capped cost rule, and explains declines. Code in `src/fraud/serving/`, tests in `tests/test_history.py` and `tests/test_serving.py`.

```
POST /score  {raw transaction}
   │
   ├─ same feature code as training (add_time_columns, add_entity_key, add_row_features)
   ├─ past-only features from the online HistoryStore (per card and per card holder)
   ├─ tuned LightGBM → raw score → Platt calibration → p
   ├─ cost rule with the 1% decline-cap penalty → approve / decline (+ review suggestion)
   ├─ declines: top-3 reasons from TreeSHAP
   └─ record the transaction in the history
```

## Components

| File | Role |
|---|---|
| `bundle.py` | Builds `artifacts/bundle/`: tuned model, feature order and raw dtypes, Platt (a, b), cost table and cap penalty λ, history state. Registers the model in MLflow as **`fraud-scorer` (version 1)**. |
| `history.py` | `HistoryStore`: incremental version of the past-only features (counts in 1 h / 24 h / 7 d, time since previous, past mean amount), per card and per card holder. |
| `scorer.py` | One-row feature frame built with the training functions, prediction, calibration, decision, reasons. |
| `reasons.py` | Reason codes from LightGBM's native TreeSHAP (`pred_contrib=True`), with readable descriptions. |
| `app.py` | FastAPI: `GET /health`, `POST /score?record=&explain=`. |

Bundle values: Platt a = 0.540, b = 0.808. λ = $46.3 for the 1% cap; notebook 07 found $47.3 with cross-fitted probabilities, while the bundle fits Platt on the whole validation half.

## Training-serving skew: tested, not assumed

The classic failure of a deployed model is computing features differently online than in training. Two tests replay **real** transactions:

1. **Features:** 2,000 consecutive transactions are scored by the online history store, seeded with all earlier transactions. All 12 past-only features match the batch computation of `features.add_past_features` (relative tolerance 1e-5, identical missing values).
2. **Scores:** 200 test-week transactions go through the full online path (raw fields → features → model). Their scores match the offline scores from notebook 06 within 1e-5.

Smaller tests check, among others:
- same-second transactions are not each other's past;
- the 7-day window drops old transactions, which still count for the past mean;
- the SHAP contributions plus the bias sum exactly to the log-odds of the raw score;
- the decision is "decline" exactly when p exceeds the returned per-transaction threshold;
- recording a transaction changes the next one's features.

## Example

A real transaction from the unlabelled Kaggle test period ($117):

```json
{
  "fraud_probability": 0.010699,
  "decision": "approve",
  "decline_threshold": 0.29355,
  "expected_cost": {"approve": 1.41, "decline": 8.42, "review": 3.14},
  "review_recommended": false,
  "explained": false,
  "model_version": "tuned-20261010"
}
```

The threshold matches the cost rule: (C_FP + λ) / (C_FP + λ + C_FN) = (8.51 + 46.34) / (8.51 + 46.34 + 132) = 0.2936. With `explain=always`, the top reasons for this transaction were C1 = 5, C2 = 4 and D11 = 16, which are masked Vesta counts and time deltas. Many reasons can only name a column family, because Vesta masks the column meanings (see `docs/problem_framing.md`).

## Latency and footprint

Measured in Docker on 1,000 real test-period transactions sent in time order (history recorded):

| | Value |
|---|---|
| p50 / p95 / p99 latency | **53 ms / 65 ms / 645 ms** |
| Declined (explained) | 1.5% of requests |
| Review suggested | 12.5% of requests |
| Container memory | 424 MiB |
| Startup (load model + 231k history keys) | about 23 s |
| Image size | 1.01 GB |

In-process breakdown per call:
- building the feature frame: 25 ms;
- prediction: 34 ms on one thread (54 ms with all threads, since threads cost more than they save on one row);
- **TreeSHAP: about 650 ms**, because the model has 2,314 trees of 255 leaves.

**A design change came out of this measurement.** The first version explained every transaction, which took **1.2 s per call**. Reasons are now computed only for **declines**, the decision a customer sees at checkout. Review cases get their reasons when a reviewer opens them (`explain=always`), off the payment path. This took p95 from 680 ms to 65 ms; the p99 is the declines.

## Keeping the decline cap in production

The bundle starts with λ from the labelled calibration half. Every week, `python -m fraud.serving.cap --log <last week's responses>` resets λ from the logged `fraud_probability` and `TransactionAmt`, so the *expected* share of good customers declined equals the 0.9% target. No labels are needed. The previous values are kept in `decision.json → penalty_history`, and the API uses the new λ after a reload. Replayed over 15 weeks, this kept the realised rate at about 1% on average, with the worst week at 1.4% instead of 1.9% ([`backtest.md`](backtest.md)).

## Limitations

- **The history store lives in memory, in one process.** It starts from the bundle (history up to the end of the training data) and loses what it recorded on restart. Several replicas would need a shared store (for example Redis) with the same semantics. The parity tests define those semantics.
- **Transactions are assumed to arrive in time order.** Late arrivals would be counted in the past of transactions already scored.
- **The review suggestion ignores capacity.** It is a per-transaction cost comparison (12.5% of requests). Choosing the 200 reviews per day with the largest expected saving, as the 3-action rule does, belongs to the review-queue system.
- **Unseen category values are treated as missing** (LightGBM maps them to NaN), the same as for categories absent from training.
- **Reasons are attributions on the raw score's log-odds.** Platt is monotone, so the ranking of reasons is unchanged; the probability itself is calibrated.

## Run

```
uv run python -m fraud.serving.bundle              # build artifacts/bundle, register in MLflow
uv run uvicorn fraud.serving.app:app               # local, http://localhost:8000/docs
docker build -t fraud-scorer:1.0 .
docker run -p 8000:8000 fraud-scorer:1.0
curl -X POST "localhost:8000/score?explain=always" -H "Content-Type: application/json" \
     -d '{"TransactionID": 1, "TransactionDT": 18500000, "TransactionAmt": 250.0, "card1": 9500, "ProductCD": "C"}'
```
