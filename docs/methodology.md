# Methodology

Step 4 of the project: for each sub-problem in [`problem_framing.md`](problem_framing.md), the candidate methods, the one chosen, why, and the evidence. Candidates come from the findings in [`reports/data_analysis.md`](../reports/data_analysis.md); each was run once without tuning, with results in [`reports/method_research.md`](../reports/method_research.md). Open questions are pre-registered hypotheses in [`hypotheses.md`](hypotheses.md).

## From data-analysis findings to candidate methods

| Finding | Requirement on the method | Candidates tried |
|---|---|---|
| Thin signal in small slices; non-monotonic relations (V258 ≥ 3, C1) | Non-linear model that captures interactions | Scorecard → logistic → LightGBM |
| Missingness is informative | Treat NaN as a value | LightGBM (native NaN) vs logistic (imputation + flags) |
| Young cards are risky; 78% of fraud is repeat fraud | Behaviour-over-time features, with memorisation kept in check | With / without card-holder features, scored on seen vs new card holders |
| December 2017 is a different regime | Choose the training window | Train without weeks 0–3 |
| Loss is concentrated in large transactions; rate ≠ loss | Judge in dollars; consider amount in training | Amount-weighted training; dollar-weighted metrics |
| Best-ranking model has the worst log loss | Decide on calibrated probabilities, priced in dollars | 3 calibrations × 8 decision policies, cost sensitivity grid |
| Step change at week 4; browser releases; 1.9–5.1% weekly noise | Detect real changes, few alarms otherwise | 6 drift monitors scored against the known events |
| 14 missing-together blocks; heavy redundancy | One change should not fire dozens of alarms | Block null rates, FDR control, multivariate domain classifier |

## Evaluation design shared by all modules

- **Split by time, never at random.** Train weeks 0–15, validation 16–19, test 20–26 (`configs/model.yaml`). A random split gave PR-AUC 0.766 [0.751, 0.780] against 0.567 [0.537, 0.599] for a time split (M1): random splitting overstates performance.
- **Resample groups, not rows.** Transactions of one card holder are correlated and fraud comes in bursts. Bootstrap CIs resample card holders (or days); these CIs are about twice as wide as row-level ones (0.058 and 0.060 vs 0.031 for PR-AUC, M2).
- **Compare models with a paired bootstrap** on the same resamples, and call a difference real only if its CI excludes 0.
- **Pre-register hypotheses** before running each experiment.

## P1. Scoring

| Candidate | Verdict |
|---|---|
| Hand-written rules | Useful as a sanity baseline; cannot use hundreds of masked features. |
| Logistic regression | **Baseline.** Interpretable, but needs imputation and cannot use missingness patterns without explicit flags. |
| **Gradient-boosted trees (LightGBM)** | **Chosen.** Tabular data with mixed types, routes NaN natively (missingness is informative, M0), captures interactions, fast to score. |
| XGBoost / CatBoost | Same family; not expected to change any conclusion, not worth the time. |
| Deep learning on tabular data | Out of scope: no expected gain on this kind of data, slower to iterate. |
| Unsupervised anomaly detection | Not needed while labels exist; a candidate for new fraud patterns, out of scope. |
| Oversampling (SMOTE) | Rejected: distorts predicted probabilities, which P2 depends on. |

**Features.** Raw columns plus past-only behavioural features per card (`card1`) and per card holder (`entity_key`): transaction counts in the last 1 h / 24 h / 7 d, time since the previous transaction, ratio of amount to past mean amount. They are computed from strictly earlier transactions only (tested in `tests/test_features.py`).

**Evidence (method research, test weeks, PR-AUC with card-holder bootstrap CIs):**
- LightGBM + past-only features (S4): PR-AUC 0.534 [0.504, 0.561]. Every other candidate is significantly lower.
- Scorecard from the analysis: 0.288. Logistic regression: 0.404. LightGBM on raw columns: 0.505.
- The card-holder features mostly help on known card holders: PR-AUC on seen card holders 0.637 with them vs 0.546 without; on new card holders 0.526 vs 0.521.
- Training without December or with amount weights brings no clear gain.

**Status: tuned** ([`reports/tuning.md`](../reports/tuning.md)).
- Early stopping on all card holders: stopping on unseen card holders only lost 0.04 PR-AUC, so the memorisation hypothesis was rejected.
- Recency weights with a half-life of 4 weeks.
- 255 leaves, learning rate 0.05.
- Test PR-AUC 0.545, +0.012 (+0.003, +0.020) over the untuned model.

## P2. Decision

| Candidate | Verdict |
|---|---|
| Fixed threshold 0.5 | **Baseline.** Ignores both costs and amounts. |
| F1-optimal threshold | Baseline. Balances precision and recall, which is not what costs money. |
| Top-k by review capacity | Used inside the 3-action policy below. |
| Cost-weighted training | Rejected as the main tool: bakes one cost assumption into the model and breaks probabilities. |
| **Calibrated probability + Bayes decision rule** | **Chosen.** |

**Bayes rule.** Decline when the expected cost of approving exceeds the expected cost of declining:

  p · C_FN(amount) > (1 − p) · C_FP(amount)

With the cost structure in `configs/costs.yaml`:
- C_FN = amount + chargeback fee.
- C_FP = fixed friction cost + margin share × amount.

This gives a **per-transaction** threshold p* = C_FP / (C_FP + C_FN), which depends on the amount.

**Why calibration comes first.** The rule is only optimal if p is a true probability. In M2, the best-ranking model had the worst log loss (0.171, against 0.110 for logistic regression), so its p cannot be used as is. Candidates are Platt scaling and isotonic regression, fitted on the validation weeks and compared on the test weeks:
- log loss and Brier score, the Brier score decomposed into reliability and resolution;
- reliability diagrams.

**3-action extension.** Approve / review / decline. Review costs a fixed amount and is limited by a daily capacity. Transactions are sent to review in order of expected savings until capacity is reached.

**Robustness.** Costs are assumptions, so every policy comparison is repeated over the sensitivity grid in `configs/costs.yaml` (hypothesis H2).

**Evidence (method research, test weeks):**
- Calibration fixes the underestimated raw scores: ECE 0.0197 → 0.0019 (Platt), log loss 0.171 → 0.092.
- Against approve-all, the Bayes rule on calibrated p saves $387k [331k, 446k], threshold 0.5 saves $149k [130k, 168k], and the 3-action policy saves $480k [423k, 542k].
- A score threshold is never the cheapest policy in any of the 30 cost scenarios.

**Status: tuned.**
- Platt calibration.
- A 1% cap on declined good customers, enforced with a Lagrange penalty (shadow price $47 per declined good customer).
- At the same decline rate the cost rule is $44k cheaper than a score threshold (22k–70k), and cheaper in all 30 cost scenarios. The 3-action policy with the cap saves $478k.
- Open items: the cap is slightly exceeded out of sample (1.17%), and the review assumptions need business input.

## P3. Monitoring without labels

| Candidate | Verdict |
|---|---|
| Wait for labels, then measure PR-AUC | Not usable alone: labels take up to 120 days. Kept as the delayed ground truth. |
| KS test per feature | Rejected as the alarm: with ~20k transactions a week almost any difference is "significant". Significance is not effect size. |
| PSI with the 0.25 rule of thumb | Rejected: alarms in 14 of 22 weeks and misses the browser releases. |
| **PSI (= symmetric KL, J-divergence) with a significance test and an effect-size floor** | **To tune.** The p-value uses n_eff·PSI ~ χ²(k−1), checked on random halves of one week: 1.1% of p-values below 0.05, so the test is conservative. Benjamini–Hochberg FDR across columns, plus PSI > 0.1. It catches the week-4 step and an unexplained change in weeks 23–25, but flags 1–2 slowly drifting columns most weeks; the alarm needs a count of flagged columns. |
| JS divergence for categorical codes | Rejected as the alarm: browser versions churn every week, so JS on `id_31` is high every week. |
| **Share of never-seen categorical values** | **Chosen.** 0.02–0.06 in ordinary weeks, 0.19 and 0.32 in the two browser-release weeks. |
| **Null-rate drift per missing-together block** | **Chosen.** 339 V columns form 14 blocks (M0). It catches the week-4 step; its other alarms (weeks 13–14 and 18) are real block changes. |
| Score drift | Kept as a signal, not as the alarm: score PSI stayed at or below 0.008 while weekly PR-AUC moved between 0.49 and 0.57. |
| **Confidence-based performance estimation (CBPE)** | **To validate.** Estimates precision, recall and cost from calibrated probabilities; valid under covariate shift only (H4). First check: weekly fraud rate estimated with error 0.33 points, correlation 0.79 over 7 weeks. |
| Domain classifier | To tune: a model trained to tell the current week from the reference. With balanced classes its optimal cross-entropy is ln 2 − JS, so it estimates a multivariate JS divergence. It separates every week (AUC 0.67–0.88) because many columns are time deltas that drift by construction; it needs those columns removed or a relative threshold. |

**Status: tuned.** Four complementary alarm rules (never-seen values > 0.058; ≥ 3 PSI-flagged columns; V-block missing shift > 0.097; domain-classifier AUC > 0.815 without time-trend columns).
- No alarm in 10 clean weeks; every known event and every change found earlier is flagged.
- Injected pipeline and covariate shifts are caught in 100% of weeks.
- Prior and concept shift are not reliably caught without labels.

**Validation.** Weekly monitoring on the labelled period (do alerts precede PR-AUC drops? H3). Then injected drift with known onset (covariate shift, prior shift, concept shift, pipeline breakage), to measure detection delay and false-alarm rate. Finally, the unlabelled test period, where real drift is already visible (M0).

## P4. Maintenance

| Candidate | Verdict |
|---|---|
| Never retrain | **Baseline.** |
| Periodic (monthly) | Compared. |
| Drift-triggered | Compared. |
| Online learning | Out of scope; labels arrive too late for it to help. |

**Status: done** ([`reports/backtest.md`](../reports/backtest.md)).
- With a 4-week label delay, retraining every week saves $134k (91k–180k) over 15 weeks against never retraining; every 4 weeks saves $91k; on drift alarms $114k.
- Break-even cost per retrain is at least $12k, so retrain whenever new labels mature.
- H3 and H5 are not supported: alarms did not precede performance drops, and drift-triggered retraining needed more retrains than periodic.
- Drift alarms are kept for investigation (for example, pipeline breaks), not as the trigger.
- The decline cap drifted when λ was estimated on two labelled weeks (worst week 1.9%). It is now reset weekly by a label-free controller targeting 0.9%: mean realised rate about 1.0%, worst week 1.4%, at about $22–24k over 15 weeks.
- Open item: a 120-day label window prevents any retraining for months.

**Backtest.** Replay the labelled weeks in order. At each week, only labels older than the delay are available for retraining. Total cost = fraud loss + false-positive cost + retraining cost. Label delays of 4 weeks and a 120-day worst case. The 120-day case leaves few labelled weeks to train on, which is itself a finding about how long a model must survive between retrains (H5).

## What the experiments changed in the method

| Plan before | Evidence | Change |
|---|---|---|
| Screen features with mutual information + correlation pruning | Screened model worse by −0.027 [−0.041, −0.013] PR-AUC (M2) | Dropped. Marginal statistics miss interactions and missingness patterns that trees use. |
| Row-level bootstrap CIs | Card-holder and day CIs are about 2× wider (M2) | All CIs resample card holders. |
| Monitor 339 V columns one by one | They form 14 missing-together blocks (M0) | Monitor null rates per block. |
| Monitor with injected drift only | Real drift already exists in the test period (M0) | Use both. |
| Single 4-week label delay | Labels mature over 120 days (M0) | Add a 120-day scenario. |
| Weight training by amount to save dollars | No gain in dollar-weighted PR-AUC, fewer dollars caught at a 3% budget (method research) | Amount enters the decision rule, not training. |
| Pure cost-minimising decision | 6.7% of good customers declined at the default costs (method research) | Add a cap on the good-customer decline rate. |
| JS divergence as the categorical alarm | Alarms every week because of version churn (method research) | Alarm on the share of never-seen values. |
| Early stopping on the validation weeks | 2,488 trees, worst log loss; card-holder features help mainly on known card holders (method research) | Tune the validation protocol (e.g. card-holder-disjoint validation). |
