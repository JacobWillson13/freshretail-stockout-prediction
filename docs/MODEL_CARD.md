# Model card: next-day stockout alerts (LightGBM, `full` features)

All numbers come from `reports/tables/` (test: `headline_test.csv`, `test_metrics_ci.csv`,
`test_paired_differences.csv`, `calibration_test.csv`, `slices_test.csv`; backtest:
`backtest_summary.csv`). Data: FreshRetailNet-50K.

## Intended use

- **Task:** at the end of day D, rank every store-product series by its risk of a stockout on
  D+1, so a replenishment team can review the top of the list overnight and in the morning
  (reorder, move stock, adjust a markdown).
- **Operating point:** a fixed daily review capacity. The reference point is the top 10% of
  series per day (5,000 of 50,000 alerts). `python -m src.predict --date D --top-k 5000` writes
  that list with a calibrated probability and the top 3 contributors per alert.
- **Not intended for:** automatic ordering without review, estimating the effect of a discount or
  promotion (the model is not causal), or series and stores outside the training population.

## Model

- LightGBM binary classifier (63 leaves, learning rate 0.05, early stopping on validation
  PR-AUC, 228 trees), 77 features: daily, weekly, category and calendar features, plus the
  intraday, history, entity and demand groups (`src/modeling.py`).
- **Selection:** highest mean PR-AUC in a 5-fold rolling-origin backtest over the train period
  (0.592, sd 0.036 across folds). The test week played no part in the choice.
- **Saved artifacts** (`models/`, not committed): `best_model.joblib` and `best_model.json`
  (features, categorical features, F1 threshold, calibration map, training window).

## Training data and window

| Part | Label dates (D+1) | Rows | Use |
|---|---|---|---|
| train | 2024-04-11 to 2024-06-11 | 3,100,000 | fitting |
| valid | 2024-06-12 to 2024-06-25 | 700,000 | early stopping, threshold, calibration |
| test | 2024-06-26 to 2024-07-02 | 350,000 | scored once |

50,000 series (898 stores, 865 products), one row per series and prediction day. The first 13
days of each series are a warm-up and carry no row.

## Target

`label = 1` when D+1 has **more than 2 out-of-stock hours in the 06:00 to 22:00 window**
(16 hours). Prevalence: 36.6% train, 37.4% valid, 35.0% test. Overnight hours are not part of
the target.

Why more than 2 hours: a single short gap can be a counting or shelf-refill artifact, while three
or more lost hours is about a fifth of the selling day and the kind of event a replenishment team
would want to prevent.

## Leakage rule

Every feature is known at the end of D. Nothing realized on D+1 is a feature, except the D+1
holiday flag (a calendar fact). The late-night hours 22 and 23 of D are used as features: they
are outside the target window and known at the end of D. `tests/test_leakage.py` scrambles all
raw data after a cutoff, rebuilds the warehouse and checks that no feature up to the cutoff
changes, for every feature of every config.

The product and store x second-category stockout rates are leave-one-out: they exclude the
row's own store (product rate) or own product (category rate), so they describe related series
only. A test flips one series' own stockout on day D and checks that its own rates do not move.

## Metrics (test week, 95% store-cluster bootstrap CIs, 1,000 resamples)

Precision@k is the share of real stockouts among the top k% of series by score each day,
averaged over the 7 days.

| Model | Precision@5% | Precision@10% | Precision@20% | Recall@10% | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Best baseline (backtest): 7-day mean OOS hours | 0.585 | 0.549 (0.543 to 0.555) | 0.504 | 0.159 | 0.466 (0.462 to 0.471) | 0.636 |
| Rule baseline: today's OOS hours | 0.602 | 0.576 (0.570 to 0.581) | 0.543 | 0.166 | 0.470 (0.467 to 0.474) | 0.649 |
| LightGBM, combined | 0.636 | 0.615 (0.610 to 0.620) | 0.575 | 0.178 | 0.538 (0.535 to 0.542) | 0.708 |
| **LightGBM, full (this model)** | **0.680** | **0.651 (0.646 to 0.656)** | **0.604** | **0.189** | **0.567 (0.563 to 0.571)** | **0.729** |

Paired bootstrap differences (same store resamples for both models):

- vs the best baseline: precision@10% +0.102 (0.096 to 0.108), PR-AUC +0.101 (0.098 to 0.104);
- vs the rule baseline: precision@10% +0.076 (0.070 to 0.082), PR-AUC +0.097 (0.094 to 0.099);
- vs LightGBM combined: precision@10% +0.037 (0.032 to 0.041), PR-AUC +0.028 (0.027 to 0.030).

The full model also beats combined in each of the 5 backtest folds (PR-AUC +0.018 to +0.032,
precision@10% +0.012 to +0.040).

At the validation F1-optimal threshold (0.291) the model flags 49% of test rows (precision
0.503, recall 0.708). That is far beyond any review capacity, which is why the capacity view is
the headline.

## Calibration

Raw scores under-predict on test (mean score 0.313 against an observed rate of 0.350). Platt
and isotonic maps were fitted on validation; Platt was chosen by store-grouped 5-fold
cross-validation on validation (mean Brier 0.192988 against 0.192992 for isotonic and 0.193784
raw; the two maps are practically tied).

| Calibration | Brier (test) | ECE, 10 bins (test) |
|---|---|---|
| raw | 0.1963 | 0.037 |
| Platt (selected) | 0.1951 | 0.021 |
| isotonic | 0.1950 | 0.020 |

Ranking uses the raw score; the calibrated probability is reported next to it. Calibration
does not remove the shift to test: calibrated probabilities still average 0.335 against an
observed rate of 0.350.

## Known weak slices (test)

- **Rare-stockout categories.** In first categories with low prevalence the model ranks poorly:
  category 28 (prevalence 3.1%, PR-AUC 0.187, precision@10% 0.146), category 22 (6.2%, 0.248),
  category 17 (6.7%, 0.238) and category 6 (9.0%, 0.222). Alerts there are mostly false.
- **Small gain over the baseline** in categories 30, 28 and 29 (PR-AUC 0.015 to 0.036 above
  the best baseline, against about 0.10 overall).
- **Series with a mid-range stockout history** (23% to 39% of train days) reach PR-AUC 0.51 to
  0.52, against 0.67 for the series that are out of stock most often.
- **Day to day variation:** PR-AUC ranges from 0.455 (prediction day 2024-06-27, prevalence
  26%) to 0.654 (2024-06-29, prevalence 44%).
- **Store volume** does not matter much: PR-AUC is 0.556 to 0.579 across store volume deciles.

## What the model relies on

Mean absolute SHAP on 20,000 test rows. The top features are the product's stockout rate in the
other stores on D (`prod_oos_rate`), the late-night out-of-stock hours of D, out of stock at
23:00, the third-level category and the average temperature. Roles: response triggers (recent
and intraday stockouts, entity rates), descriptive (category, calendar, weather) and actionable
(discount, promotion; discount is the 6th feature). **Importance is not
causation**: a high SHAP value for discount does not mean that changing the discount would
change the risk.

Temperature moved up the ranking after the days-observed count (a date index) was removed.
Temperature follows the season, so part of its weight may be a time trend rather than a
weather effect; with about 3 months of data the two cannot be separated.

## Limitations

- **Short test window.** One week of test labels. Store-cluster CIs capture variation across
  stores, not across weeks; the backtest spread (PR-AUC sd 0.036 over 5 weeks) is the better
  guide to week-to-week variation. Validation is more optimistic than test for every model,
  the baselines included, which points to a shift in late June.
- **Censored demand.** Sales stop when the shelf is empty, so sales-based features understate
  demand on stockout days. The demand features are a simple in-stock-rate estimate; removing
  them lowers test PR-AUC by less than 0.001.
- **Associations, not causes.** Features and SHAP values describe what predicts a stockout in
  this data, not what would prevent one.
- **Single dataset, single period.** Trained and tested on about 3 months of one retailer's
  perishable goods; seasonality beyond that window is not covered.
