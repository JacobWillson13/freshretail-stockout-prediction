"""Models, temporal evaluation, feature importance and the saved final model.

Reads model_dataset from the DuckDB warehouse. Usage: python -m src.modeling

Final fit protocol:
- train split: model fitting, for logistic regression and LightGBM alike (same rows);
- valid split: LightGBM early stopping, decision thresholds and calibration;
- test split: scored once, after the final model has been selected by the
  rolling-origin backtest (src/backtest.py), not by test results.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import calibration
from src.data import ROOT, WAREHOUSE
from src.metrics import best_f1_threshold, capacity_metrics, evaluate

TABLES = ROOT / "reports" / "tables"
MODELS_DIR = ROOT / "models"
PRED_PATH = ROOT / "data" / "predictions.parquet"
VALID_PRED_PATH = ROOT / "data" / "predictions_valid.parquet"
SEED = 42
warnings.filterwarnings("ignore", message=".*eval_set.*is deprecated")

# ---------------------------------------------------------------- feature sets
CALENDAR = ["day_of_week", "holiday_flag", "holiday_tomorrow"]

DAILY = [
    "oos_hours", "oos_fraction", "longest_oos_run", "in_stock_sales", "sale_amount",
    "sales_per_available_hour", "discount", "activity_flag",
    "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
    "oos_hours_lag1", "oos_hours_lag2", "oos_hours_lag7",
    "sale_amount_lag1", "sale_amount_lag7", "discount_lag1",
    "oos_hours_mean_7d", "oos_days_7d", "longest_oos_run_max_7d", "sales_mean_7d",
    "sales_std_7d", "discount_mean_7d", "activity_days_7d",
    "oos_hours_mean_14d", "oos_days_14d", "sales_mean_14d", "sales_std_14d",
    "sales_trend_7v14",
]

WEEKLY = [
    "wk_oos_hours_sum", "wk_oos_days", "wk_any_oos_days", "wk_sales_sum", "wk_sales_std",
    "wk_sales_cv", "wk_discount_mean", "wk_activity_days", "wk_oos_hours_change",
    "wk_oos_days_change", "wk_sales_change_pct",
]

CATEGORY = [
    "first_category_id", "cat_n_products", "cat_oos_rate", "cat_share_affected", "cat_sales",
    "cat_oos_rate_lag1", "cat_oos_rate_mean_7d", "cat_share_affected_mean_7d",
    "cat_sales_mean_7d", "cat_sales_trend",
]

# added feature groups (sql/09 to sql/12)
INTRADAY = [
    "oos_at_close", "oos_hours_last3", "oos_hours_late_night", "first_oos_hour",
    "last_in_stock_hour", "oos_after_sale", "ended_day_oos", "oos_at_close_lag1",
    "oos_at_close_mean_7d",
]
HISTORY = ["hist_oos_hours_mean", "hist_oos_day_share"]
ENTITY = [
    "second_category_id", "third_category_id",
    "prod_oos_rate", "prod_oos_rate_mean_7d", "store_oos_rate", "store_oos_rate_mean_7d",
    "store_cat2_oos_rate", "store_cat2_oos_rate_mean_7d",
]
DEMAND = ["demand_est", "lost_sales_est", "demand_est_mean_7d", "lost_sales_est_mean_7d"]

NEW_GROUPS = {"intraday": INTRADAY, "history": HISTORY, "entity": ENTITY, "demand": DEMAND}

CATEGORICAL = {"day_of_week", "first_category_id", "second_category_id", "third_category_id"}
# LightGBM native categoricals; day_of_week stays ordinal
LGBM_CATEGORICAL = {"first_category_id", "second_category_id", "third_category_id"}
# high-cardinality ids are not one-hot encoded for logistic regression; the entity rates
# carry that information instead
LOGREG_EXCLUDE = {"second_category_id", "third_category_id"}

CONFIGS = {
    "daily": DAILY + CALENDAR,
    "weekly": WEEKLY + CALENDAR,
    "category": CATEGORY + CALENDAR,
    "combined": DAILY + WEEKLY + CATEGORY + CALENDAR,
    "full": DAILY + WEEKLY + CATEGORY + INTRADAY + HISTORY + ENTITY + DEMAND + CALENDAR,
}

# full minus one new group at a time (LightGBM only)
ABLATIONS = {
    f"full_minus_{g}": [f for f in CONFIGS["full"] if f not in set(feats)]
    for g, feats in NEW_GROUPS.items()
}

# ranking baselines: a single feature used as the score (higher = more risk)
RULE_BASELINES = {
    "baseline_today": ("baseline: today's OOS hours (rule > 2h)", "oos_hours"),
    "baseline_mean7": ("baseline: 7-day mean OOS hours", "oos_hours_mean_7d"),
    "baseline_history": ("baseline: series stockout share (expanding)", "hist_oos_day_share"),
}
LR_BASELINE = ("baseline: LR on OOS hours + OOS at close", ["oos_hours", "oos_at_close"])
TODAY_RULE_THRESHOLD = 2.5   # the rule "today > 2h" on whole hours

MODEL_NAMES = {"logreg": "logistic regression", "lightgbm": "LightGBM"}

# importance labels: actionable levers, response triggers, descriptive context
ACTIONABLE = ("discount", "activity", "wk_discount", "wk_activity")
DESCRIPTIVE = ("precpt", "avg_temperature", "avg_humidity", "avg_wind_level", "holiday",
               "day_of_week", "first_category_id", "second_category_id", "third_category_id",
               "cat_n_products")


def feature_role(name: str) -> str:
    if name.startswith(ACTIONABLE):
        return "actionable"
    if name.startswith(DESCRIPTIVE):
        return "descriptive"
    return "response trigger"


def all_features() -> list[str]:
    return sorted(set(sum(CONFIGS.values(), [])))


def entry_name(kind: str, config: str) -> str:
    return f"{kind}_{config}"


# ---------------------------------------------------------------- data
ID_COLUMNS = ["store_id", "product_id", "dt"]


def feature_select_sql(cols: list[str]) -> str:
    """Cast numeric features to FLOAT (float32) in SQL to halve memory on the full data."""
    parts = []
    for c in cols:
        parts.append(f"CAST({c} AS INTEGER) AS {c}" if c in CATEGORICAL else f"CAST({c} AS FLOAT) AS {c}")
    return ", ".join(parts)


def load_dataset(con: duckdb.DuckDBPyConnection | None = None, where: str = "",
                 extra: tuple[str, ...] = ()) -> pd.DataFrame:
    own = con is None
    if own:
        con = duckdb.connect(str(WAREHOUSE), read_only=True)
    cols = all_features()
    extra_sql = "".join(f", {c}" for c in extra)
    df = con.execute(
        f"SELECT store_id, product_id, dt, label_dt, split, CAST(label AS TINYINT) AS label, "
        f"next_oos_hours{extra_sql}, {feature_select_sql(cols)} FROM model_dataset {where}"
    ).df()
    if own:
        con.close()
    # a fixed row order keeps LightGBM bagging reproducible; sorting in pandas is much faster
    # than ORDER BY over ~110 columns in DuckDB
    df.sort_values(["dt", "store_id", "product_id"], inplace=True, ignore_index=True)
    return df


# ---------------------------------------------------------------- models
def logreg_features(features: list[str]) -> list[str]:
    return [f for f in features if f not in LOGREG_EXCLUDE]


def make_logreg(features: list[str]) -> Pipeline:
    feats = logreg_features(features)
    num = [f for f in feats if f not in CATEGORICAL]
    cat = [f for f in feats if f in CATEGORICAL]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=300, C=1.0))])


def fit_logreg(X: pd.DataFrame, y) -> Pipeline:
    feats = logreg_features(list(X.columns))
    return make_logreg(feats).fit(X[feats], y)


LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    n_estimators=1000, random_state=SEED, verbose=-1, n_jobs=-1,
)


def fit_lgbm(X_tr, y_tr, X_es, y_es):
    """LightGBM with early stopping (patience 50) on the (X_es, y_es) holdout."""
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    cats = [c for c in X_tr.columns if c in LGBM_CATEGORICAL]
    model.fit(
        X_tr, y_tr, eval_set=[(X_es, y_es)], eval_metric="average_precision",
        categorical_feature=cats or "auto",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def predict_score(model, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, Pipeline):
        X = X[logreg_features(list(X.columns))]
    return model.predict_proba(X)[:, 1].astype("float32")


def fit_entry(kind: str, feats: list[str], tr: pd.DataFrame, es: pd.DataFrame):
    """Fit one model on `tr`; `es` is the early-stopping holdout (LightGBM only)."""
    if kind == "logreg":
        return fit_logreg(tr[feats], tr.label), {}
    model = fit_lgbm(tr[feats], tr.label, es[feats], es.label)
    return model, {"best_iteration": int(model.best_iteration_)}


# ---------------------------------------------------------------- experiment grid
def final_entries(include_ablations: bool = True) -> list[tuple[str, str, list[str]]]:
    """(entry, kind, features) for every fitted model in the final protocol."""
    out = [("baseline_lr", "logreg", LR_BASELINE[1])]
    for config, feats in CONFIGS.items():
        for kind in ("logreg", "lightgbm"):
            out.append((entry_name(kind, config), kind, feats))
    if include_ablations:
        for name, feats in ABLATIONS.items():
            out.append((entry_name("lightgbm", name), "lightgbm", feats))
    return out


def describe_entry(entry: str) -> tuple[str, str]:
    """(model label, config label) for tables."""
    if entry in RULE_BASELINES:
        return RULE_BASELINES[entry][0], RULE_BASELINES[entry][1]
    if entry == "baseline_lr":
        return LR_BASELINE[0], "+".join(LR_BASELINE[1])
    kind, config = entry.split("_", 1)
    return MODEL_NAMES[kind], config


def score_row(entry: str, split: str, frame: pd.DataFrame, score, threshold: float, **info) -> dict:
    model, config = describe_entry(entry)
    return {"entry": entry, "model": model, "config": config, "split": split,
            **evaluate(frame.label.values, score, threshold),
            **capacity_metrics(frame.label.values, score, frame.dt.values), **info}


def run_experiments(df: pd.DataFrame | None = None, include_ablations: bool = True,
                    verbose: bool = True):
    """Fit every baseline and model on train, pick thresholds on valid, score valid and test.

    Returns (results table, test predictions, valid predictions, fitted models dict).
    """
    if df is None:
        df = load_dataset()
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    rows, models = [], {}
    keys = ["store_id", "product_id", "dt", "label", "next_oos_hours"]
    preds, preds_va = te[keys].copy(), va[keys].copy()

    for entry, (_, col) in RULE_BASELINES.items():
        s_va, s_te = va[col].fillna(0).values, te[col].fillna(0).values
        thr = TODAY_RULE_THRESHOLD if entry == "baseline_today" else best_f1_threshold(va.label.values, s_va)
        for split, frame, s in (("valid", va, s_va), ("test", te, s_te)):
            rows.append(score_row(entry, split, frame, s, thr, n_features=1, fit_seconds=0.0))
        preds[f"score_{entry}"], preds_va[f"score_{entry}"] = s_te, s_va

    for entry, kind, feats in final_entries(include_ablations):
        start = time.time()
        model, info = fit_entry(kind, feats, tr, va)
        secs = time.time() - start
        s_va, s_te = predict_score(model, va[feats]), predict_score(model, te[feats])
        thr = best_f1_threshold(va.label.values, s_va)
        n_feats = len(logreg_features(feats)) if kind == "logreg" else len(feats)
        for split, frame, s in (("valid", va, s_va), ("test", te, s_te)):
            rows.append(score_row(entry, split, frame, s, thr, n_features=n_feats,
                                  fit_seconds=secs, **info))
        preds[f"score_{entry}"], preds_va[f"score_{entry}"] = s_te, s_va
        models[entry] = model
        if verbose:
            r = rows[-1]
            print(f"  {entry:<34} valid PR-AUC {rows[-2]['pr_auc']:.3f}  test PR-AUC {r['pr_auc']:.3f}  "
                  f"P@10% {r['precision_at_10']:.3f}  ({secs:.0f}s)", flush=True)

    return pd.DataFrame(rows), preds, preds_va, models


# ---------------------------------------------------------------- selection and saving
BACKTEST_SUMMARY = TABLES / "backtest_summary.csv"


def pick_best(results: pd.DataFrame, backtest_summary: Path = BACKTEST_SUMMARY) -> tuple[str, str]:
    """Final model: highest mean backtest PR-AUC among the fitted models.

    Falls back to validation PR-AUC when the backtest has not been run. The test set never
    picks the winner. Returns (entry, criterion).
    """
    if backtest_summary.exists():
        bt = pd.read_csv(backtest_summary)
        bt = bt[~bt.entry.str.startswith("baseline")]
        return bt.sort_values("pr_auc_mean", ascending=False).entry.iloc[0], "mean backtest PR-AUC"
    v = results[(results.split == "valid") & ~results.entry.str.startswith("baseline")
                & ~results.entry.str.contains("minus")]
    return v.sort_values("pr_auc", ascending=False).entry.iloc[0], "validation PR-AUC (no backtest)"


def entry_features(entry: str) -> tuple[str, list[str]]:
    kind, config = entry.split("_", 1)
    feats = CONFIGS.get(config) or ABLATIONS[config]
    return kind, (logreg_features(feats) if kind == "logreg" else feats)


def _date_range(frame: pd.DataFrame) -> list[str]:
    return [str(pd.Timestamp(frame.label_dt.min()).date()), str(pd.Timestamp(frame.label_dt.max()).date())]


def save_model(entry: str, model, results: pd.DataFrame, df: pd.DataFrame, preds_va: pd.DataFrame,
               criterion: str, models_dir: Path = MODELS_DIR) -> dict:
    """Save the final model and its metadata (features, threshold, calibration, window)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    kind, feats = entry_features(entry)
    va_rows = results[(results.entry == entry) & (results.split == "valid")].iloc[0]
    s_va = preds_va[f"score_{entry}"].values
    method, cv_brier = calibration.choose_method(s_va, preds_va.label.values, preds_va.store_id.values)
    cal = calibration.FITTERS[method](s_va, preds_va.label.values)
    tr = df[df.split == "train"]
    va = df[df.split == "valid"]
    meta = {
        "entry": entry, "kind": kind, "model": describe_entry(entry)[0],
        "config": describe_entry(entry)[1], "features": feats,
        "categorical_features": [f for f in feats if f in LGBM_CATEGORICAL] if kind == "lightgbm" else [],
        "threshold_f1_valid": float(va_rows.threshold),
        "best_iteration": int(va_rows.best_iteration) if pd.notna(va_rows.get("best_iteration")) else None,
        "selected_by": criterion,
        "calibration": cal,
        "calibration_cv_brier_valid": cv_brier,
        "training_window": {"train_label_dates": _date_range(tr), "valid_label_dates": _date_range(va),
                            "train_rows": int(len(tr)), "valid_rows": int(len(va))},
        "target": "D+1 has more than 2 out-of-stock hours in the 06:00 to 22:00 window",
    }
    joblib.dump(model, models_dir / "best_model.joblib")
    (models_dir / "best_model.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------- importance and tables
def importance(model, df: pd.DataFrame, feats: list[str], n_sample: int = 100_000,
               n_repeats: int = 3) -> pd.DataFrame:
    """LightGBM gain plus permutation importance (PR-AUC drop) on a validation sample."""
    va = df[df.split == "valid"].sample(n=n_sample, random_state=SEED)
    perm = permutation_importance(
        model, va[feats], va.label, scoring="average_precision",
        n_repeats=n_repeats, random_state=SEED, n_jobs=1,
    )
    out = pd.DataFrame({
        "feature": feats,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    })
    if hasattr(model, "booster_"):
        gain = model.booster_.feature_importance(importance_type="gain")
        out["gain"] = gain
        out["gain_share"] = gain / gain.sum()
    out["role"] = out.feature.map(feature_role)
    return out.sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)


def comparison_table(results: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    cols = ["entry", "model", "config", "n_features", "threshold", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "precision_at_5", "precision_at_10", "precision_at_20",
            "recall_at_5", "recall_at_10", "recall_at_20", "alert_rate", "tp", "fp", "fn", "tn"]
    t = results[results.split == split][cols].copy()
    return t.reset_index(drop=True)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    print(f"model_dataset: {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e9:.2f} GB", flush=True)
    results, preds, preds_va, models = run_experiments(df)
    results.to_csv(TABLES / "model_results_all_splits.csv", index=False)
    comparison_table(results).to_csv(TABLES / "model_comparison_test.csv", index=False)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(PRED_PATH, index=False)
    preds_va.to_parquet(VALID_PRED_PATH, index=False)
    entry, criterion = pick_best(results)
    meta = save_model(entry, models[entry], results, df, preds_va, criterion)
    kind, feats = entry_features(entry)
    imp = importance(models[entry], df, feats)
    imp.to_csv(TABLES / "feature_importance_best.csv", index=False)
    (TABLES / "best_model.json").write_text(json.dumps(
        {k: meta[k] for k in ("entry", "kind", "config", "selected_by", "threshold_f1_valid",
                              "best_iteration", "training_window")}, indent=2))
    print(f"selected {entry} by {criterion}")
    print(comparison_table(results).drop(columns=["tp", "fp", "fn", "tn"]).round(3).to_string(index=False))
    print(imp.head(15).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
