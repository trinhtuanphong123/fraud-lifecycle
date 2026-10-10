# Problem framing

Step 1 of the project: who has the problem, which decision has to be made, how success is measured, and why the IEEE-CIS data can (and cannot) answer it. The data itself is described in [`data_dictionary.md`](data_dictionary.md); how each sub-problem is solved is in [`methodology.md`](methodology.md).

## 1. Business context

Vesta sells **Guaranteed Payments** to e-commerce merchants: when Vesta approves a card-not-present transaction, it takes on the full liability for any fraud chargeback that follows ([Green Sheet](https://www.greensheet.com/emagazine&article_id=6132), [Fintech News](https://fintechnews.sg/80590/payments/vesta-partners-stripe-to-shield-merchants-from-fraudulent-transaction-liability/)). Every approved fraudulent transaction is therefore money Vesta loses directly, and every good transaction it declines is revenue lost for the merchant and for Vesta.

Each transaction needs a decision **at checkout, in real time**: approve, decline, or send to a human reviewer. Two operational facts make this harder than a one-off classification task:

- **Labels arrive late.** A transaction is only known to be fraud when a chargeback is filed, and cardholders have up to ~120 days to file one. Performance measured today describes the model of months ago.
- **The world moves.** Fraudsters adapt, browsers and devices change, the product mix shifts. A model that is good at launch degrades.

## 2. Problem statement

> Build and operate a decision system that **minimises total cost** (fraud losses + cost of declining good customers + operating cost) per transaction, and **keeps that performance over time** although labels arrive up to 120 days late.

The top-level metric is money. Ranking metrics such as PR-AUC are intermediate measures for one component.

## 3. Sub-problems

| | Sub-problem | Type | Input → output | Success measure | Baseline | Module |
|---|---|---|---|---|---|---|
| P1 | **Scoring** | Probabilistic binary classification | Transaction features at decision time → p(fraud) | PR-AUC (ranking), log loss / calibration (probability quality) | Logistic regression | M1, M2 |
| P2 | **Decision** | Decision under asymmetric costs | p, amount, cost assumptions → approve / review / decline | Total cost in $, share of good customers declined | Threshold 0.5; approve all | M3 |
| P3 | **Monitoring** | Distribution-shift detection without labels | Logged features and scores → alert | Detection delay, false-alarm rate | — | M4 |
| P4 | **Maintenance** | Retraining policy under label delay | Alerts, delayed labels → retrain or not | Total cost including retraining cost | Never retrain | M5 |

## 4. Constraints

- **Point-in-time information only.** A feature may use only what is known when the transaction arrives. Velocity features look strictly into the past; splits are by time.
- **Delayed labels.** Training and evaluation can only use transactions old enough for their label to have matured.
- **Class imbalance.** 3.5% fraud: accuracy and ROC-AUC look good for weak models.
- **Explainability.** A declined customer or a reviewer needs a reason (reason codes, M6).
- **Latency.** Scoring sits in the checkout path, so the model must score one transaction quickly.

## 5. Does the data fit the problem?

| The problem needs | IEEE-CIS provides | Evidence |
|---|---|---|
| Real transactions with fraud labels | 590,540 real, labelled transactions (not simulated) | M0 |
| The amount of each transaction, to price decisions (P2) | `TransactionAmt` in USD | M0 |
| Enough time to see change (P3, P4) | ~26 labelled weeks, a 30-day gap, ~26 unlabelled weeks | M0 |
| A period where labels have not arrived yet (P3) | The Kaggle test file: same features, no labels | M0 |
| Real drift to test a monitor on | 60.6% of test browser values (`id_31`) never seen in train; the V35–V52 block drops from 28.6% to 15.2% missing; high-fraud product C rises from 11.6% to 13.7% | M0 |
| A known label definition, to simulate label delay | Chargeback, propagated to linked accounts, 120-day window | M0, M1 |
| A way to recognise the same customer (velocity features, leakage analysis) | An approximate card-holder key from `card1`, `addr1` and the card issue day | M1, M2 |

## 6. What the data cannot provide, and how the project handles it

| Missing | Consequence | Handling |
|---|---|---|
| Real costs (chargeback fees, customer value) | Savings in $ are not exact | Explicit assumptions in `configs/costs.yaml`; every cost result is reported across a sensitivity grid |
| Chargeback dates | Exact label arrival times are unknown | Simulate label delay: 4 weeks, and a 120-day worst case |
| Transactions Vesta's existing system declined | The data likely contains only processed transactions: selection bias; counterfactual outcomes of declines are unknown | Stated as a limitation; not modelled |
| Real column names (masked) | Reason codes such as "V258" mean little to a reviewer | Report reasons at column-family level |
| Test labels | Real drift in the test period can be detected but its effect on performance cannot be verified | Validate the monitor on the labelled period and on injected drift; on the test period, report signals and estimated performance only |
| Real timestamps | Seasonality is approximate | Assume 2017-12-01 as day 0, supported by two holiday-season peaks one year apart |

The label definition (chargeback, propagation to transactions linked by account, email or billing address, legitimate after 120 days without report) follows Vesta's description in the competition discussion. It could not be re-checked against the original post while writing this; the 120-day window matches card-network chargeback rules ([J.P. Morgan](https://www.jpmorgan.com/insights/payments/security-trust/identify-report-manage-fraud-ease), [Chargeback Gurus](https://www.chargebackgurus.com/faq)).

## 7. Why this dataset and not another

| Dataset | Why not the main dataset |
|---|---|
| ULB Credit Card Fraud | Two days of data, features already PCA-transformed: too short for drift, no entity information, no meaningful reason codes. |
| PaySim | Simulated, with a known leak through the balance columns; conclusions would not transfer. |
| **IEEE-CIS** | One of few public sets with all four ingredients: real data, amounts, a long time span, and an unlabelled future period. |

## 8. Scope

**In scope:** P1–P4 on IEEE-CIS; a scoring API with reason codes (M6).

**Out of scope, on purpose:** chasing the Kaggle leaderboard or reconstructing user IDs with community tricks (rewards memorising people, see M1); deep learning on tabular data; streaming infrastructure (Kafka); dashboards before monitoring works. Each costs time without changing the answers to P1–P4.
