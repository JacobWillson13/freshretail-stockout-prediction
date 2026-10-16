"""Generate notebooks/stockout_analysis.ipynb from the cell list below.

Usage:
  python notebooks/build_notebook.py
  jupyter nbconvert --to notebook --execute --inplace notebooks/stockout_analysis.ipynb \
      --ExecutePreprocessor.timeout=3600
"""
from pathlib import Path

import nbformat as nbf

CELLS: list[tuple[str, str]] = []


def md(text: str):
    CELLS.append(("md", text.strip("\n")))


def code(text: str):
    CELLS.append(("code", text.strip("\n")))


# --------------------------------------------------------------------------- intro
md("""
# Next-day stockout risk on FreshRetailNet-50K

**Problem.** A replenishment team can review a limited number of store-products each evening.
Which series will run out of stock tomorrow, ranked so the top of the list is worth their time?
A second question: which aggregation level (daily, weekly, category) carries the signal?

**Prediction.** For each store-product-day D, using only what is known at the end of D, predict
whether D+1 has **more than 2 out-of-stock hours inside the 06:00 to 22:00 operating window**.

Sections: 1 data checks, 2 EDA, 3 aggregation levels, 4 additional features, 5 models and
baselines, 6 backtest, 7 test results with CIs, 8 calibration, 9 slices, 10 importance and SHAP,
11 results and suggested operating setup. Figures are written to `reports/figures/`, tables to
`reports/tables/`.
""")

code("""
import json
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src import checks, eda, modeling
from src.data import WAREHOUSE
from src.plots import CONFIG_COLORS, NEUTRAL, SERIES, SEQ_CMAP, TEXT_2, save, style
from src.run_sql import build_warehouse

style()
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
TABLES = ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# Build the warehouse (raw parquet -> SQL models) when it is missing.
if not WAREHOUSE.exists():
    build_warehouse().close()
con = duckdb.connect(str(WAREHOUSE), read_only=True)
con.execute("SELECT COUNT(*) FROM model_dataset").fetchone()
""")

# --------------------------------------------------------------------------- 1 checks
md("""
## 1. Data checks

Row counts, coverage, date ranges, missing values, duplicate keys, array lengths and
series contiguity. Then the two checks the target depends on: which value of
`hours_stock_status` means out of stock, and which hour indices form the 06:00 to 22:00 window
counted by `stock_hour6_22_cnt`.
""")

code("""
dc = checks.run_all(con)
dc["files"]
""")

code("""
display(dc["series_overlap"])
display(dc["duplicates"])
display(dc["arrays"])
display(dc["contiguity"])
missing = dc["missing"]
print("Columns with any missing value:",
      missing[(missing.raw_train > 0) | (missing.raw_eval > 0)].column.tolist() or "none")
""")

md("""
### Stock status coding and the operating window

For several candidate hour ranges we count the 1s (and separately the 0s) of `hours_stock_status`
and check how often that count equals `stock_hour6_22_cnt`.
""")

code("""
dc["window_match"]
""")

code("""
display(dc["status_vs_sales"])
display(dc["full_day_status"])
""")

code("""
wm = dc["window_match"].set_index("hours (0-based, inclusive)")
svs = dc["status_vs_sales"].set_index("stock_status")
print(f"Count of 1s over hours 6..21 matches stock_hour6_22_cnt on "
      f"{wm.loc['6..21', 'share_rows_count_of_1_matches']:.2%} of train rows; "
      f"best alternative: {wm.drop('6..21').share_rows_count_of_1_matches.max():.2%}.")
print(f"Hours coded 1 record a sale {svs.loc[1, 'share_hours_with_sale']:.1%} of the time, "
      f"hours coded 0 {svs.loc[0, 'share_hours_with_sale']:.1%}.")
""")

md("""
**Verdict.**

- **1 = out of stock, 0 = in stock.** Hours coded 1 almost never record a sale, days coded 1 for
  all 16 window hours average about 0.05 units sold versus about 1 unit for days coded 0 for all
  16 hours, and a count of 1s (not 0s) reproduces `stock_hour6_22_cnt`. So
  `stock_hour6_22_cnt` is a count of **out-of-stock** hours despite its name.
- **The window is array positions 6 to 21 (0-based), i.e. the 16 hours from 06:00 to 22:00.**
  This range matches on 100% of rows; no other candidate exceeds 83%.
- Eval (2024-06-26 to 2024-07-02) starts the day after train ends (2024-06-25), with the same
  50,000 store-products, so eval is used as the final test set.
- No missing values, no duplicate keys, all arrays have 24 elements, and every series is 97
  contiguous days, so row-based rolling windows equal day-based windows.
""")

md("""
### Target prevalence and threshold

Share of labeled rows whose next day exceeds each threshold of out-of-stock window hours.
""")

code("""
dc["target_prevalence"]
""")

md("""
The target uses **more than 2 hours**. Moving from "any out-of-stock hour" (> 0) to > 2 drops about
one positive in seven: those are 1 to 2 hour gaps, often a shelf refill or a recording artifact,
which a replenishment team cannot act on. Three or more hours is at least 19% of the 16-hour
selling day, a material loss of selling time. Higher thresholds (> 4, > 8) shrink the positive
class without changing the nature of the problem. Prevalence at > 2 is about 35 to 37% and is
stable across train, validation and test. Overnight out-of-stock hours (positions 0 to 5 and
22 to 23) are common (about 57% of days) but mostly reflect closed hours. They are reported
separately below and are not part of the target.
""")

# --------------------------------------------------------------------------- 2 EDA
md("""
## 2. Exploratory analysis

All relationships below are associations measured on observational data, not causal effects.
Sample sizes (n) are shown with every comparison.
""")

code("""
dist = eda.oos_hours_distribution(con)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(dist.oos_hours, dist.share * 100, color=[SERIES[0] if h > 2 else NEUTRAL for h in dist.oos_hours],
       width=0.8)
ax.set_xlabel("Out-of-stock hours in the 06:00 to 22:00 window (per store-product-day)")
ax.set_ylabel("% of store-product-days")
ax.set_title("Most days have no stockout; a long tail runs to fully out of stock days")
ax.text(0.98, 0.9, f"> 2 hours: {dist[dist.oos_hours > 2].share.sum():.1%} of days\\n"
        f"> 0 hours: {dist[dist.oos_hours > 0].share.sum():.1%} of days\\nn = {dist.n.sum():,}",
        transform=ax.transAxes, ha="right", va="top", color=TEXT_2)
save(fig, "eda_01_oos_hours_distribution")
plt.show()
dist.to_csv(TABLES / "eda_oos_hours_distribution.csv", index=False)
""")

code("""
by_hour = eda.stockout_by_hour(con)
by_hour_cat = eda.stockout_by_hour_category(con)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
ax.axvspan(5.5, 21.5, color=SERIES[0], alpha=0.06, lw=0)
ax.plot(by_hour.hour, by_hour.oos_rate * 100, color=SERIES[0], marker="o", ms=4)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour of day (array position)")
ax.set_ylabel("% of store-product-hours out of stock")
ax.set_title("Stockout rate by hour, all categories")
ax.text(13.5, by_hour.oos_rate.max() * 100 + 1, "operating window 06:00 to 22:00", ha="center",
        color=TEXT_2, va="bottom", fontsize=8)
piv = by_hour_cat.pivot(index="first_category_id", columns="hour", values="oos_rate")
piv = piv.loc[piv.loc[:, 6:21].mean(axis=1).sort_values(ascending=False).index]
ax = axes[1]
im = ax.imshow(piv.values * 100, aspect="auto", cmap=SEQ_CMAP)
ax.set_yticks(range(len(piv)), piv.index, fontsize=7)
ax.set_xticks(range(0, 24, 2), range(0, 24, 2))
ax.grid(False)
ax.set_xlabel("Hour of day")
ax.set_ylabel("first_category_id (sorted by window rate)")
ax.set_title("Stockout rate by hour and category")
fig.colorbar(im, ax=ax, label="% out of stock")
save(fig, "eda_02_stockout_by_hour")
plt.show()
by_hour.to_csv(TABLES / "eda_stockout_by_hour.csv", index=False)
piv.to_csv(TABLES / "eda_stockout_by_hour_category.csv")
print(f"Window hours (6..21) mean rate {by_hour[by_hour.hour.between(6, 21)].oos_rate.mean():.1%}; "
      f"overnight mean rate {by_hour[~by_hour.hour.between(6, 21)].oos_rate.mean():.1%}")
""")

md("""
The hourly profile is U-shaped: out-of-stock rates are high overnight, fall in the morning after
restocking, and climb through the afternoon and evening as shelves sell down. The overnight rate
mostly reflects closed or unstocked hours, which is why the target only counts the operating
window. Categories differ by a factor of several in their window stockout rate, so category
context is plausibly useful.
""")

code("""
rel = {
    "discount": eda.rate_by(con, eda.DISCOUNT_BIN, "bin"),
    "promotion (activity_flag)": eda.rate_by(con, "activity_flag", "bin"),
    "holiday_flag": eda.rate_by(con, "holiday_flag", "bin"),
    "temperature (C, 5-degree bins)": eda.rate_by(con, eda.TEMP_BIN, "bin"),
    "precipitation": eda.rate_by(con, eda.PRECIP_BIN, "bin"),
}
fig, axes = plt.subplots(1, 5, figsize=(17, 3.9), sharey=True)
for ax, (name, d) in zip(axes, rel.items()):
    labels = d["bin"].astype(str).str.replace(r"^[a-z0-9] ", "", regex=True)
    ax.bar(range(len(d)), d.stockout_day_rate * 100, color=SERIES[0], width=0.7)
    ax.set_xticks(range(len(d)), labels, rotation=35, ha="right", fontsize=8)
    for i, (r, n) in enumerate(zip(d.stockout_day_rate, d.n)):
        ax.text(i, r * 100 + 0.5, f"n={n/1e3:,.0f}k", ha="center", fontsize=7, color=TEXT_2)
    ax.set_title(name, fontsize=10)
axes[0].set_ylabel("% of days with > 2 window stockout hours")
fig.suptitle("Same-day stockout rate by discount, promotion, holiday and weather (associations)",
             fontweight="bold")
save(fig, "eda_03_relationships")
plt.show()
pd.concat({k: v for k, v in rel.items()}, names=["factor"]).reset_index(level=0).to_csv(
    TABLES / "eda_relationships.csv", index=False)
for k, v in rel.items():
    print(k); display(v.round(3))
""")

md("""
Reading these as associations:

- **Discount.** The deepest discounts (< 0.5 of list price, small n) coincide with the highest
  same-day stockout rate. This may reflect markdowns on products that are already scarce or near
  expiry, rather than discounts causing stockouts.
- **Promotion days** (`activity_flag = 1`) have a stockout-day rate about 3.5 points higher than
  other days (40.2% against 36.7%), and **holidays** about 1 point higher (38.8% against 37.6%).
- **Weather.** Heavy rain is associated with slightly more stockouts. Cooler days (15 to 20 C)
  show the highest rate, but temperature is confounded with season and with the calendar.

All of these differences are small next to the persistence effect shown below.
""")

code("""
pers = eda.persistence(con)
pers_h = eda.persistence_by_hours(con)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(pers_h.oos_hours, pers_h.p_tomorrow_stockout * 100, color=SERIES[0], marker="o", ms=4)
ax.axvline(2.5, color=NEUTRAL, ls="--", lw=1)
ax.set_xlabel("Out-of-stock window hours today")
ax.set_ylabel("% with > 2 hours tomorrow")
ax.set_title("Stockouts persist: tomorrow's risk rises with today's out-of-stock hours")
ax.text(2.7, 20, "baseline rule: today > 2h", color=TEXT_2, fontsize=8)
save(fig, "eda_04_persistence")
plt.show()
pers.to_csv(TABLES / "eda_persistence.csv", index=False)
display(pers.round(3))
""")

code("""
ex = eda.example_series(con)
keys = ex[["store_id", "product_id"]].drop_duplicates().values.tolist()
fig, axes = plt.subplots(len(keys), 2, figsize=(14, 2.5 * len(keys)), sharex=True)
for (ax_o, ax_s), (s, p) in zip(axes, keys):
    d = ex[(ex.store_id == s) & (ex.product_id == p)]
    ax_o.bar(d.dt, d.oos_hours, color=[SERIES[1] if h > 2 else "#f2c3ad" for h in d.oos_hours], width=0.9)
    ax_o.set_ylabel("OOS window hours")
    ax_o.set_title(f"store {s}, product {p}: out-of-stock hours (dark = > 2h)", loc="left", fontsize=9)
    ax_s.plot(d.dt, d.sale_amount, color=SERIES[0], lw=1.5)
    ax_s.set_ylabel("daily sales")
    ax_s.set_title(f"store {s}, product {p}: daily sales", loc="left", fontsize=9)
    for ax in (ax_o, ax_s):
        ax.axvline(pd.Timestamp("2024-06-26"), color=NEUTRAL, ls="--", lw=1)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
fig.suptitle("Three example store-product series (dashed line: start of eval)", fontweight="bold")
save(fig, "eda_05_example_series")
plt.show()
""")

md("""
Stockout days cluster in runs, and the frequency of stockouts changes over time within a series
(for example a busier late June). Sales usually continue on stockout days, because the shelf
often empties part way through the day, but sales during out-of-stock hours are lost. Daily sales
are therefore censored demand, which is why the daily features include in-stock sales and sales
per available hour as well as raw sales.
""")

# --------------------------------------------------------------------------- 3 aggregation
md("""
## 3. Aggregation levels

Every feature set is joined onto the same daily prediction rows (store, product, D), produced by
the SQL layer in `sql/` (`stg_` -> `int_` -> `features_` -> `model_dataset`). Each non-daily
config also carries the same three calendar fields (day of week, holiday today, holiday
tomorrow), so the configs differ only in the aggregation level of the stock and sales signals.

**Leakage rule.** Every feature is known at the end of D. Nothing realized on D+1 is a feature,
except the D+1 holiday flag (a calendar fact). `tests/test_leakage.py` enforces this by
scrambling all data after a cutoff, rebuilding the warehouse, and checking that features up to
the cutoff do not change.
""")


code("""
split_summary = con.execute('''
    SELECT split, COUNT(*) AS rows, MIN(dt) AS first_D, MAX(dt) AS last_D,
           MIN(label_dt) AS first_label_day, MAX(label_dt) AS last_label_day, AVG(label) AS prevalence
    FROM model_dataset GROUP BY 1 ORDER BY first_D
''').df()
split_summary.to_csv(TABLES / "split_summary.csv", index=False)
split_summary
""")

md("""
**Temporal split** (by the label day D+1). Eval follows train, so the eval days are the test set.
Validation is the last 14 days of train, and training is everything before that. The first 13
days of each series are dropped as warm-up, so every row has 14 days of history. Features for
the first test rows use train-period history; that is allowed, because features only look
backward from D.
""")

# --------------------------------------------------------------------------- 4 features
md("""
## 4. Additional features: intraday, history, entity, demand

Four more groups are built in SQL (`sql/09` to `sql/12`) and joined in `features_all`. All of them
are known at the end of D; the leakage test scrambles every raw value after a cutoff and checks
that none of them changes.

- **intraday:** out of stock at close (hour 21), OOS hours 19 to 21, OOS hours at 22 and 23 (outside
  the target window but known at the end of D), first OOS hour, last in-stock hour, an OOS hour after
  the first sale (sell-out), OOS at hour 23, lag 1 and 7-day mean of OOS at close;
- **history:** expanding per-series mean OOS hours and share of days with more than 2 OOS hours
  (no days-observed count: every series starts on the same day, so it would act as a date index);
- **entity:** stockout rate of the same product in the other stores and of the store's other
  products in the same second category (both leave-one-out, so the row's own value is excluded),
  and of the store across all its products, for D and the trailing 7 days, plus the second and
  third category ids (LightGBM categoricals; logistic regression uses the rates only);
- **demand:** censored-demand estimate (in-stock sales rate x 16), estimated lost sales and their
  7-day means.

The config `full` is `combined` plus all four groups.
""")

code("""
groups = {"daily": modeling.DAILY, "weekly": modeling.WEEKLY, "category": modeling.CATEGORY,
          "calendar": modeling.CALENDAR, **modeling.NEW_GROUPS}
cfg_table = pd.DataFrame([
    {"config": k, "n_features": len(v), "features": ", ".join(v)} for k, v in modeling.CONFIGS.items()
])
cfg_table.to_csv(TABLES / "feature_configs.csv", index=False)
display(pd.DataFrame([{"group": g, "n": len(f), "features": ", ".join(f)} for g, f in groups.items()]))
new_cols = sum(modeling.NEW_GROUPS.values(), [])
con.execute(f"SELECT {', '.join(f'AVG(CAST({c} AS DOUBLE)) AS {c}' for c in new_cols)} "
            "FROM model_dataset").df().T.rename(columns={0: "mean"}).round(3)
""")

# --------------------------------------------------------------------------- 5 models
md("""
## 5. Models and baselines

The pipeline (`make backtest train report`) writes every table used below; this notebook reads them
and refits nothing, except when they are missing.

- **Baselines, scored as rankings:** today's OOS hours (the rule "today > 2h"), 7-day mean
  OOS hours, expanding per-series stockout share, and a logistic regression on OOS hours plus OOS at
  close.
- **Logistic regression** and **LightGBM** on daily, weekly, category, combined and full. Both model
  types train on the same rows: the full train split.
- **Ablation:** LightGBM on `full` minus one added group at a time.
- Thresholds (F1-optimal) and calibration are fitted on validation; the test week is scored once.
""")

code("""
import subprocess
needed = [modeling.BACKTEST_SUMMARY, TABLES / "model_results_all_splits.csv", TABLES / "test_metrics_ci.csv"]
if not all(p.exists() for p in needed) or not modeling.PRED_PATH.exists():
    for target in ["backtest", "train", "report"]:
        subprocess.run(["make", target, f"PYTHON={sys.executable}"], cwd=ROOT, check=True)
results = pd.read_csv(TABLES / "model_results_all_splits.csv")
preds = pd.read_parquet(modeling.PRED_PATH)
meta = json.loads((ROOT / "models" / "best_model.json").read_text())
best = meta["entry"]
comp = modeling.comparison_table(results, "test")
comp.drop(columns=["tp", "fp", "fn", "tn"]).round(3)
""")

# --------------------------------------------------------------------------- 6 backtest
md("""
## 6. Rolling-origin backtest and model selection

Five weekly folds over the train period. Each fold trains on every label date before its validation
week, holds out the last 7 of those days for LightGBM early stopping, and validates on the week. The
selected model is the one with the highest **mean backtest PR-AUC**; the test week plays no part
in the choice.
""")

code("""
bt_folds = pd.read_csv(TABLES / "backtest_folds.csv")
bt = pd.read_csv(modeling.BACKTEST_SUMMARY)
print(f"Selected: {best} ({meta['selected_by']})")
display(bt_folds.pivot_table(index="entry", columns="fold", values="pr_auc").round(3))
bt[["entry", "model", "config", "pr_auc_mean", "pr_auc_std", "precision_at_10_mean",
    "precision_at_10_std", "roc_auc_mean"]].round(4)
""")

code("""
from IPython.display import Image
Image(filename=str(ROOT / "reports" / "figures" / "backtest_01_folds.png"))
""")

# --------------------------------------------------------------------------- 7 test with CIs
md("""
## 7. Test results with confidence intervals

The test week (7 label days x 50,000 series) is scored once. Intervals come from a cluster bootstrap
over stores (1,000 resamples of the 898 stores, every model on the same draws), so they reflect
store-to-store variation, not only row noise. Precision@k is the share of real stockouts among the
top k% of series by score on each day, averaged over the 7 days: it is what a team with a fixed
daily review capacity would see. **Precision@10% (5,000 alerts per day) is the headline metric.**
""")

code("""
Image(filename=str(ROOT / "reports" / "figures" / "hero_precision_at_k.png"))
""")

code("""
head = pd.read_csv(TABLES / "headline_test.csv")
ci = pd.read_csv(TABLES / "test_metrics_ci.csv")
display(head[["role", "label", "precision_at_10", "precision_at_10_lo", "precision_at_10_hi",
              "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc"]].round(3))
ci[["label", "precision_at_10", "precision_at_10_lo", "precision_at_10_hi", "pr_auc", "pr_auc_lo",
    "pr_auc_hi", "roc_auc", "roc_auc_lo", "roc_auc_hi"]].round(3)
""")

code("""
diffs = pd.read_csv(TABLES / "test_paired_differences.csv")
diffs[diffs.metric.isin(["pr_auc", "precision_at_10", "roc_auc"])].round(4)
""")

code("""
Image(filename=str(ROOT / "reports" / "figures" / "ci_01_precision_at_10.png"))
""")

md("""
### Aggregation study and ablation

Which aggregation level carries the signal is answered with the four configs daily, weekly,
category and combined. The ablation removes one added group at a time from `full`.
""")

code("""
order = ["daily", "weekly", "category", "combined", "full"]
test = results[results.split == "test"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, metric in zip(axes, ["pr_auc", "precision_at_10"]):
    x = np.arange(len(order))
    for j, (kind, color) in enumerate([("logreg", SERIES[0]), ("lightgbm", SERIES[1])]):
        vals = [test[test.entry == f"{kind}_{c}"][metric].iloc[0] for c in order]
        ax.bar(x + (j - 0.5) * 0.38, vals, width=0.36, color=color, label=modeling.MODEL_NAMES[kind])
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 0.5) * 0.38, v + 0.004, f"{v:.3f}", ha="center", fontsize=7, color=TEXT_2)
    b = test[test.entry == "baseline_today"][metric].iloc[0]
    ax.axhline(b, color=NEUTRAL, ls="--", lw=1.2, label="rule baseline (today's OOS hours)")
    ax.set_xticks(x, order)
    ax.set_ylim(test[metric].min() - 0.05, test[metric].max() + 0.05)
    ax.set_title({"pr_auc": "PR-AUC", "precision_at_10": "Precision@10%"}[metric] + " on test")
axes[0].legend(fontsize=8, loc="upper left", frameon=True, facecolor="white", edgecolor="#e6e5df")
fig.suptitle("Test metrics by feature config and model", fontweight="bold")
save(fig, "model_01_metrics_by_config")
plt.show()
display(pd.read_csv(TABLES / "ablation_full.csv").round(4))
""")

code("""
fig, ax = plt.subplots(figsize=(7, 4.6))
y = preds.label.values
for e, color in [(best, SERIES[0]), ("lightgbm_combined", SERIES[3]), ("lightgbm_daily", SERIES[2]),
                 ("baseline_today", NEUTRAL)]:
    p, r, _ = precision_recall_curve(y, preds[f"score_{e}"])
    ax.plot(r, p, color=color, lw=1.6, label=" / ".join(modeling.describe_entry(e)))
ax.axhline(y.mean(), color="#c9c8c2", lw=1)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-recall curves on test (grey line = prevalence)")
ax.legend(fontsize=8)
save(fig, "model_02_pr_curves")
plt.show()
""")

# --------------------------------------------------------------------------- 8 calibration
md("""
## 8. Calibration

Platt and isotonic maps are fitted on the validation scores of the selected model and applied to
test. The map used for reporting is chosen by grouped (store) cross-validation on validation only.
""")

code("""
display(pd.read_csv(TABLES / "calibration_test.csv").round(4))
Image(filename=str(ROOT / "reports" / "figures" / "calibration_01_reliability.png"))
""")

# --------------------------------------------------------------------------- 9 slices
md("""
## 9. Where the model fails: slices

Test PR-AUC and precision@10% (computed within each slice) by first category, store volume decile
(mean daily sales over the train period), series stockout-history decile (share of train days with
more than 2 OOS hours) and day.
""")

code("""
sl = pd.read_csv(TABLES / "slices_test.csv")
for dim, d in sl.groupby("slice", sort=False):
    print(dim); display(d.drop(columns="slice").round(3))
""")

code("""
Image(filename=str(ROOT / "reports" / "figures" / "slices_01_pr_auc.png"))
""")

# --------------------------------------------------------------------------- 10 importance
md("""
## 10. What the model relies on: importance and SHAP

Permutation importance (validation sample) and Tree SHAP values (20,000 test rows) for the selected
model. Features are labeled by role: **actionable** (discount, promotion), **response
trigger** (recent and intraday stockouts, entity stockout rates, demand estimates) and
**descriptive** (weather, calendar, category ids). Importance describes what the model
uses to rank risk. **It is not causation.**
""")

code("""
imp = pd.read_csv(TABLES / "feature_importance_best.csv")
shap_imp = pd.read_csv(TABLES / "shap_importance_test.csv")
top = imp.head(15).iloc[::-1]
role_color = {"actionable": SERIES[1], "response trigger": SERIES[0], "descriptive": NEUTRAL}
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.barh(top.feature, top.perm_importance_mean, xerr=top.perm_importance_std,
        color=[role_color[r] for r in top.role], height=0.7)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=r) for r, c in role_color.items()], loc="lower right")
ax.set_title(f"Permutation importance, {best} (drop in validation PR-AUC; not causation)")
save(fig, "importance_01_top_features")
plt.show()
display(shap_imp.head(20).round(4))
shap_imp.groupby(["group"]).mean_abs_shap.sum().sort_values(ascending=False).round(4)
""")

code("""
role_summary = imp.groupby("role").agg(
    n_features=("feature", "size"),
    perm_importance_sum=("perm_importance_mean", "sum"),
    **({"gain_share_sum": ("gain_share", "sum")} if "gain_share" in imp else {}),
).sort_values("perm_importance_sum", ascending=False)
role_summary.to_csv(TABLES / "feature_importance_by_role.csv")
role_summary.round(4)
""")

code("""
display(Image(filename=str(ROOT / "reports" / "figures" / "shap_01_summary.png")))
display(Image(filename=str(ROOT / "reports" / "figures" / "shap_02_dependence_top3.png")))
""")

# --------------------------------------------------------------------------- 11 recommendation
md("""
## 11. Results and suggested operating setup
""")

md("""
RECOMMENDATION_PLACEHOLDER
""")


def build(path: Path):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in CELLS]
    nbf.write(nb, path)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    rec = here / "recommendation.md"
    if rec.exists():
        CELLS[-1] = ("md", rec.read_text().strip("\n"))
    build(here / "stockout_analysis.ipynb")
    print("wrote", here / "stockout_analysis.ipynb")
