# Pre-registered hypotheses

Written **before** running the experiments. Results are recorded below each hypothesis whether they confirm it or not. A negative result is still a result.

Each hypothesis tests a method choice explained in [`methodology.md`](methodology.md).

## H1 (M1): a random split inflates performance

A random split gives a higher PR-AUC than a time split, and the time split gives a higher PR-AUC than a time split with entities seen in training removed.

- Metric: PR-AUC on each test set, reported next to the test fraud rate (the PR-AUC of a random model).
- **Result: partly confirmed** (notebooks 01–02; 95% CIs over card holders).
  - A random split inflates performance: PR-AUC 0.766 [0.751, 0.780] vs 0.567 [0.537, 0.599] for a time split, CIs not overlapping (lift 21.8 vs 16.5).
  - The second half is **not confirmed on PR-AUC**: on test card holders unseen in training, PR-AUC is 0.560 [0.528, 0.592]. Its lift is lower (12.3), but that group also has a higher fraud rate (4.6% vs 3.4%), so the two numbers are not directly comparable.
  - The first version of this result was wrong because of a bug: under pandas 3, a missing `addr1` or `D1` made the card-holder key NaN. It was fixed and re-run before these numbers.

## H2 (M3): a cost-based policy beats the 0.5 threshold

A calibrated, Bayes-optimal policy has a lower total cost than both the 0.5 threshold and the F1-optimal threshold across the whole sensitivity grid in `configs/costs.yaml`.

- **Result: confirmed** ([`method_research.md`](../reports/method_research.md), [`tuning.md`](../reports/tuning.md)).
  - In all 30 cost scenarios, no score threshold (0.5, F1-optimal, top 2%) was the cheapest policy. The calibrated Bayes rule cost $41k–$378k less than the best of them on the test weeks.
  - With a cap of 1% declined good customers on both sides, the Bayes rule is still $44k [22k, 70k] cheaper at the default costs, and cheaper in 30 of 30 scenarios.
  - Caveat: without the cap, the cost-optimal rule declines 6–7% of good customers, which is why the cap was added.

## H3 (M4): drift alerts precede performance drops

On the labelled period, weeks flagged by the label-free monitor are followed by a drop in realised PR-AUC or a rise in realised cost.

- This may well be false on 26 weeks of data. The injected-drift experiments (covariate, prior, concept, pipeline bug) are the controlled test.
- **Result: not supported** ([`backtest.md`](../reports/backtest.md)).
  - In the replay with a 4-week label delay, the staleness gap (PR-AUC of a weekly-retrained model minus a never-retrained one) averaged 0.039 in the week after an alarm and 0.032 after a week without one.
  - Performance decayed with model age (correlation −0.44) whether or not an alarm fired.
  - The tuned monitors do detect data changes reliably (no alarm in 10 clean weeks, injected pipeline and covariate shifts caught 100%), but those changes were not what made the model worse.

## H4 (M4): CBPE tracks realised performance under covariate shift only

Performance estimated from calibrated probabilities matches realised performance under covariate shift and diverges under concept shift.

- **Result: partly tested, still open** ([`method_research.md`](../reports/method_research.md), [`tuning.md`](../reports/tuning.md)).
  - On the 7 test weeks, the weekly fraud rate estimated from calibrated probabilities tracked the observed one: correlation 0.79, mean absolute error 0.33 points. Precision of the top 2% was overestimated by 0.07–0.14 in three weeks.
  - Concept shift was shown to leave every label-free *monitor* statistic unchanged, but CBPE itself was not run on injected concept shift. That test is still to do.

## H5 (M5): drift-triggered retraining is cost-efficient

Drift-triggered retraining reaches a total cost (fraud loss + retraining cost) no worse than monthly retraining with fewer retrains.

- **Result: not supported as stated** ([`backtest.md`](../reports/backtest.md); 4-week label delay, weeks 12–26).
  - Drift-triggered retraining saved *more* than retraining every 4 weeks, +$23k [4k, 43k], but needed *more* retrains (5 vs 2), because alarms clustered in weeks 22–25.
  - Retraining every week beat both: +$134k [91k, 180k] vs never, about $12k per retrain.
  - Since retraining is cheap compared with that, the binding constraint is label availability, not drift detection.
