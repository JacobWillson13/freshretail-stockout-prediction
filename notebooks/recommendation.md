### What the numbers say

Test set: 350,000 store-product-days (7 label days x 50,000 series), prevalence 35.0%. The PR-AUC
of a random ranking equals prevalence. Intervals are 95% store-cluster bootstrap CIs (1,000
resamples). Precision@10% is the share of real stockouts among the top 10% of series each day.
The model was selected by mean PR-AUC in a 5-fold rolling-origin backtest, not on test.

| Model | Precision@10% | PR-AUC | ROC-AUC | Backtest PR-AUC (mean ± sd) |
|---|---|---|---|---|
| Baseline: 7-day mean OOS hours (best baseline in the backtest) | 0.549 (0.543 to 0.555) | 0.466 (0.462 to 0.471) | 0.636 | 0.476 ± 0.035 |
| Baseline: today's OOS hours (rule > 2h) | 0.576 (0.570 to 0.581) | 0.470 (0.467 to 0.474) | 0.649 | 0.473 ± 0.030 |
| LightGBM, daily | 0.607 (0.603 to 0.612) | 0.532 (0.528 to 0.536) | 0.699 | 0.559 ± 0.032 |
| LightGBM, combined | 0.615 (0.610 to 0.620) | 0.538 (0.535 to 0.542) | 0.708 | 0.567 ± 0.038 |
| Logistic regression, full | 0.639 | 0.562 | 0.727 | 0.578 ± 0.035 |
| **LightGBM, full** | **0.651 (0.646 to 0.656)** | **0.567 (0.563 to 0.571)** | **0.729** | **0.592 ± 0.036** |

1. **Daily aggregation carries the signal.** With LightGBM, daily features reach a test PR-AUC
   of 0.532, category features 0.478 and last-completed-week features 0.460, below the one-line
   rule baseline (0.470). Combining the three levels adds a small gain over daily (+0.007
   PR-AUC, CI 0.006 to 0.008).
2. **The added features matter more than the aggregation level.** The `full` config (combined
   plus intraday, history, entity and demand features) adds +0.037 precision@10%
   (CI 0.032 to 0.041) and +0.028 PR-AUC over combined on test, and beats combined in all 5
   backtest folds. The entity group contributes the most: how often the same product is out of
   stock in the other stores today (leave-one-out, so the row's own store is excluded). The
   late-night out-of-stock hours of D come next. The censored-demand estimates add nothing
   measurable (less than 0.001 PR-AUC when removed).
3. **Model choice matters less than the information.** On `full`, logistic regression is close
   to LightGBM on backtest precision@10% (0.690 against 0.692) and trails it by 0.014 backtest
   PR-AUC and 0.012 test precision@10%.
4. **The baselines are strong and close to each other.** Stockouts persist, and every simple
   ranking (today's hours, 7-day mean, series history, a two-feature LR) lands at backtest PR-AUC
   0.46 to 0.48. On test the best model adds about 0.10 PR-AUC and 0.08 to 0.10 precision@10%
   over the two baselines in the table.
5. **Validation is more optimistic than test for every model, the baselines included.** Test
   precision@10% is 0.651 against 0.720 on validation, and the backtest folds range from 0.658
   to 0.752. Because the baselines drop too, this points to a harder late June and early July
   period rather than to overfitting, though one test week cannot settle it.
6. **What the best model relies on** (importance is not causation): the product's stockout rate
   in the other stores and the late-night out-of-stock hours (response triggers), the product
   category (descriptive) and discount (actionable, 6th by SHAP). Average temperature ranks 5th;
   it follows the season, so part of its weight may be a time trend rather than weather.

### Suggested operating setup

If this were run for a replenishment team: **LightGBM on the `full` feature set, used as a
ranked alert list with a fixed daily review capacity, starting at the top 10% of series (5,000
alerts per day).**

- **Lead time.** Every configuration produces its warning at the end of day D for day D+1, which
  leaves one overnight cycle to reorder, move stock or adjust a markdown. What differs is
  reaction speed. Weekly features change once a week and cannot react to a stockout that started
  today. The strongest signals (other stores running out today, the shelf being empty at 22:00)
  refresh daily, after closing.
- **False negatives versus false positives.** For perishables, a missed stockout costs a day of
  lost sales. A false alarm costs review time and, if acted on blindly, overstock that may spoil.
  At the F1-optimal threshold the model flags 49% of all store-products each day, which no team
  could review, so the cutoff is set by capacity rather than by F1. Each alert carries a
  calibrated probability (Platt, ECE 0.021 on test) and its top 3 contributors.
- **Complexity.** `full` needs four more SQL feature models than `combined`, each an aggregate
  over tables the pipeline already builds. Logistic regression on `full` is within 0.012
  precision@10% and simpler to explain, so it is a reasonable alternative. Daily-only LightGBM is
  a weaker one (-0.044 precision@10%).
- **Caveats.** The test window is one week. The store-cluster CIs cover variation across stores,
  not across weeks; the backtest spread (PR-AUC sd 0.036) is the better guide to week-to-week
  variation. The model is weak where stockouts are rare (for example first category 28,
  prevalence 3.1%, PR-AUC 0.187). Feature importance and SHAP values are associations: they say
  where the model looks, not what would happen if a discount or promotion were changed.

**Capacity trade-off on test.** At the top 10% of series per day, 65.1% of alerts are real
stockouts (54.9% for the best baseline, 57.6% for the rule "today > 2h"), and they catch 18.9%
of the next day's stockouts. The top 5% raises precision to 68.0% but catches only 9.9%. The top
20% gives precision 60.4% and recall 34.9%.
