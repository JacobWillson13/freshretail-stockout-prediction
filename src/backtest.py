"""Rolling-origin backtest over the train period.

Five weekly folds, all inside the train and validation label dates (the eval week is
never touched here). For each fold:
- validation: one week of label dates;
- training window: every label date before the validation week;
- early stopping: the final 7 label days of the training window are held out for
  LightGBM early stopping, so the validation week plays no part in fitting;
- fitting rows: the training window minus that holdout, for every model, so logistic
  regression and LightGBM see the same rows.

The model used on the test set is the one with the highest mean backtest PR-AUC
(src/modeling.py reads reports/tables/backtest_summary.csv).

Usage: python -m src.backtest
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import modeling as m
from src.metrics import ranking_metrics

FOLD_CONFIGS = ("daily", "combined", "full")
N_FOLDS = 5
ES_DAYS = 7
METRICS = ["pr_auc", "roc_auc", "precision_at_5", "precision_at_10", "precision_at_20",
           "recall_at_5", "recall_at_10", "recall_at_20"]


def make_folds(label_dates, n_folds: int = N_FOLDS, fold_days: int = 7,
               es_days: int = ES_DAYS) -> list[dict]:
    """Weekly folds ending at the last label date given (oldest fold first)."""
    dates = np.array(sorted(pd.unique(pd.to_datetime(pd.Series(label_dates)))))
    need = n_folds * fold_days
    if len(dates) < need + es_days + 1:
        raise ValueError(f"{len(dates)} label dates are too few for {n_folds} folds")
    folds = []
    for i in range(n_folds):
        val = dates[len(dates) - need + i * fold_days: len(dates) - need + (i + 1) * fold_days]
        val_start = pd.Timestamp(val[0])
        es_start = val_start - pd.Timedelta(days=es_days)
        folds.append({"fold": i + 1, "fit_end": es_start - pd.Timedelta(days=1), "es_start": es_start,
                      "val_start": val_start, "val_end": pd.Timestamp(val[-1])})
    return folds


def fold_entries(configs=FOLD_CONFIGS) -> list[tuple[str, str, list[str]]]:
    out = [("baseline_lr", "logreg", m.LR_BASELINE[1])]
    for config in configs:
        for kind in ("logreg", "lightgbm"):
            out.append((m.entry_name(kind, config), kind, m.CONFIGS[config]))
    return out


def run_backtest(df: pd.DataFrame, n_folds: int = N_FOLDS, es_days: int = ES_DAYS,
                 configs=FOLD_CONFIGS, verbose: bool = True) -> pd.DataFrame:
    dev = df[df.split != "test"]
    folds = make_folds(dev.label_dt, n_folds=n_folds, es_days=es_days)
    rows = []
    for f in folds:
        fit = dev[dev.label_dt <= f["fit_end"]]
        es = dev[(dev.label_dt >= f["es_start"]) & (dev.label_dt < f["val_start"])]
        va = dev[(dev.label_dt >= f["val_start"]) & (dev.label_dt <= f["val_end"])]
        info = {"fold": f["fold"], "fit_rows": len(fit), "es_rows": len(es), "val_rows": len(va),
                "fit_end": f["fit_end"].date(), "val_start": f["val_start"].date(),
                "val_end": f["val_end"].date(), "val_prevalence": float(va.label.mean())}
        if verbose:
            print(f"fold {f['fold']}: fit label dates <= {info['fit_end']}, early stop "
                  f"{f['es_start'].date()}..{(f['val_start'] - pd.Timedelta(days=1)).date()}, validate "
                  f"{info['val_start']}..{info['val_end']}", flush=True)
        y, day = va.label.values, va.dt.values
        for entry, (_, col) in m.RULE_BASELINES.items():
            rows.append({"entry": entry, **info, **ranking_metrics(y, va[col].fillna(0).values, day)})
        for entry, kind, feats in fold_entries(configs):
            start = time.time()
            model, extra = m.fit_entry(kind, feats, fit, es)
            score = m.predict_score(model, va[feats])
            rows.append({"entry": entry, **info, **ranking_metrics(y, score, day), **extra,
                         "fit_seconds": time.time() - start})
            if verbose:
                print(f"  {entry:<22} PR-AUC {rows[-1]['pr_auc']:.3f}  "
                      f"P@10% {rows[-1]['precision_at_10']:.3f}  ({time.time() - start:.0f}s)", flush=True)
            del model
    out = pd.DataFrame(rows)
    labels = out.entry.map(m.describe_entry)
    out.insert(1, "model", labels.str[0])
    out.insert(2, "config", labels.str[1])
    return out


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    g = folds.groupby(["entry", "model", "config"], sort=False)[METRICS]
    mean, std = g.mean().add_suffix("_mean"), g.std(ddof=1).add_suffix("_std")
    out = pd.concat([mean, std], axis=1)
    out = out[[c for met in METRICS for c in (f"{met}_mean", f"{met}_std")]]
    out["n_folds"] = g.size()
    return out.reset_index().sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


def main():
    m.TABLES.mkdir(parents=True, exist_ok=True)
    df = m.load_dataset()
    folds = run_backtest(df)
    folds.to_csv(m.TABLES / "backtest_folds.csv", index=False)
    summary = summarize(folds)
    summary.to_csv(m.BACKTEST_SUMMARY, index=False)
    print(summary[["entry", "pr_auc_mean", "pr_auc_std", "precision_at_10_mean",
                   "precision_at_10_std"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
