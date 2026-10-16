"""End to end on the synthetic warehouse: fit, save the model, score a day, write alerts."""
import json

import pandas as pd
import pytest

from src import modeling as m
from src import predict
from src.data import connect, register_raw_frames
from src.run_sql import run_models
from tests.conftest import make_rows, split_frames


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    db = tmp / "warehouse.duckdb"
    con = connect(db)
    register_raw_frames(con, *split_frames(make_rows(seed=11, n_stores=6, n_products=8)))
    run_models(con, verbose=False)
    df = m.load_dataset(con)
    con.close()
    results, _, preds_va, models = m.run_experiments(df, include_ablations=False, verbose=False)
    entry, criterion = m.pick_best(results, backtest_summary=tmp / "missing.csv")
    meta = m.save_model(entry, models[entry], results, df, preds_va, criterion, models_dir=tmp / "models")
    # both model types must be scorable, whichever wins on the synthetic data
    other = "lightgbm_full" if meta["kind"] == "logreg" else "logreg_full"
    m.save_model(other, models[other], results, df, preds_va, criterion, models_dir=tmp / "other")
    return tmp, db, meta


def test_saved_metadata(trained):
    tmp, _, meta = trained
    saved = json.loads((tmp / "models" / "best_model.json").read_text())
    assert saved["entry"] == meta["entry"]
    assert saved["features"] and saved["calibration"]["method"] in ("platt", "isotonic")
    assert saved["training_window"]["train_rows"] > 0
    assert saved["selected_by"].startswith(("mean backtest", "validation"))


@pytest.mark.parametrize("model_dir", ["models", "other"])
@pytest.mark.parametrize("which", ["labeled", "unlabeled"])
def test_predict_writes_ranked_alerts(trained, which, model_dir):
    tmp, db, _ = trained
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    last = con.execute("SELECT MAX(dt) FROM features_all").fetchone()[0]
    labeled = con.execute("SELECT MAX(dt) FROM model_dataset").fetchone()[0]
    n_series = con.execute("SELECT COUNT(*) FROM features_all WHERE dt = ?", [last]).fetchone()[0]
    con.close()
    date = str(labeled if which == "labeled" else last)
    path = predict.score_date(date, top_k=10, model_path=tmp / model_dir / "best_model.joblib",
                              warehouse=db, out_dir=tmp / "alerts" / model_dir)
    out = pd.read_csv(path)
    assert path.name == f"alerts_{date}.csv"
    assert list(out.columns[:5]) == ["store_id", "product_id", "score", "calibrated_prob", "rank"]
    assert [c for c in out.columns if c.startswith("top_contributor")] == [
        "top_contributor_1", "top_contributor_2", "top_contributor_3"]
    assert len(out) == min(10, n_series)
    assert out["rank"].tolist() == list(range(1, len(out) + 1))
    assert out.score.is_monotonic_decreasing
    assert out.calibrated_prob.between(0, 1).all()
    assert out.top_contributor_1.str.contains("=").all()


def test_predict_rejects_unknown_date(trained):
    tmp, db, _ = trained
    with pytest.raises(SystemExit):
        predict.score_date("1999-01-01", model_path=tmp / "models" / "best_model.joblib",
                           warehouse=db, out_dir=tmp / "alerts")
