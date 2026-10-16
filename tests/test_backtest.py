"""Rolling-origin folds: weekly validation, early-stopping holdout before it, no overlap."""
import pandas as pd

from src import backtest
from src.modeling import load_dataset


def test_make_folds_are_ordered_and_disjoint():
    dates = pd.date_range("2024-04-11", "2024-06-25")
    folds = backtest.make_folds(dates)
    assert len(folds) == 5
    assert folds[-1]["val_end"] == pd.Timestamp("2024-06-25")
    assert folds[0]["val_start"] == pd.Timestamp("2024-05-22")
    for f in folds:
        assert (f["val_end"] - f["val_start"]).days == 6
        assert (f["val_start"] - f["es_start"]).days == 7
        assert f["fit_end"] < f["es_start"]
    for a, b in zip(folds, folds[1:]):
        assert b["val_start"] == a["val_end"] + pd.Timedelta(days=1)


def test_backtest_runs_on_synthetic_warehouse(warehouse):
    df = load_dataset(warehouse)
    out = backtest.run_backtest(df, n_folds=2, es_days=3, configs=("daily",), verbose=False)
    assert set(out.fold) == {1, 2}
    assert set(out.entry) >= {"baseline_today", "baseline_mean7", "baseline_history", "baseline_lr",
                              "logreg_daily", "lightgbm_daily"}
    # validation weeks never reach the test label dates
    test_start = df[df.split == "test"].label_dt.min()
    assert (pd.to_datetime(out.val_end) < test_start).all()
    summary = backtest.summarize(out)
    assert (summary.n_folds == 2).all() and "pr_auc_std" in summary
