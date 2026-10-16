"""Score every series for one prediction day and write a ranked alert list.

The alert for day D is produced at the end of D and covers D+1. Features come from
features_all, which holds every scorable day, including the last observed day whose
label (D+1) does not exist yet.

Usage:
  python -m src.predict --date 2024-07-02 --top-k 5000 [--model models/best_model.joblib]

Writes reports/alerts/alerts_<date>.csv with store_id, product_id, score, calibrated_prob,
rank and the top 3 contributors (Tree SHAP values in log-odds for LightGBM, coefficient x
standardized value for logistic regression).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import calibration
from src import modeling as m
from src.data import ROOT, WAREHOUSE

ALERTS_DIR = ROOT / "reports" / "alerts"
DEFAULT_MODEL = m.MODELS_DIR / "best_model.joblib"


def load_model(path: Path = DEFAULT_MODEL):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    return joblib.load(path), meta


def load_features(con, date: str, feats: list[str]) -> pd.DataFrame:
    df = con.execute(
        f"SELECT store_id, product_id, dt, {m.feature_select_sql(feats)} FROM features_all "
        f"WHERE dt = CAST(? AS DATE) ORDER BY store_id, product_id", [date]
    ).df()
    if df.empty:
        lo, hi = con.execute("SELECT MIN(dt), MAX(dt) FROM features_all").fetchone()
        raise SystemExit(f"No feature rows for {date}; scorable dates are {lo} to {hi}")
    return df


def contributions(model, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Per-row additive contributions to the log-odds, one column per source feature."""
    feats = list(X.columns)
    if hasattr(model, "booster_"):
        contrib = model.booster_.predict(X, pred_contrib=True)
        return contrib[:, :-1], feats   # last column is the expected value
    if isinstance(model, Pipeline):
        pre, clf = model.named_steps["pre"], model.named_steps["clf"]
        Z = pre.transform(X)
        Z = Z.toarray() if hasattr(Z, "toarray") else Z
        parts = Z * clf.coef_[0]
        names = pre.get_feature_names_out()
        source = [_source_feature(n, feats) for n in names]
        out = np.zeros((len(X), len(feats)))
        for j, f in enumerate(source):
            out[:, feats.index(f)] += parts[:, j]
        return out, feats
    raise TypeError(f"unsupported model type {type(model)}")


def _source_feature(name: str, feats: list[str]) -> str:
    base = name.split("__", 1)[-1].removeprefix("missingindicator_")
    return max((f for f in feats if base == f or base.startswith(f + "_")), key=len)


def top_contributors(contrib: np.ndarray, X: pd.DataFrame, n: int = 3) -> list[list[str]]:
    """Readable 'feature=value (+0.123)' strings for the n largest positive contributions."""
    feats = list(X.columns)
    values = X.to_numpy()
    order = np.argsort(-contrib, axis=1)[:, :n]
    out = []
    for i, idx in enumerate(order):
        row = []
        for j in idx:
            v = values[i, j]
            v_str = "NULL" if pd.isna(v) else f"{v:.3g}"
            row.append(f"{feats[j]}={v_str} ({contrib[i, j]:+.3f})")
        out.append(row)
    return out


def score_date(date: str, top_k: int = 5000, model_path: Path = DEFAULT_MODEL,
               warehouse: Path = WAREHOUSE, out_dir: Path = ALERTS_DIR) -> Path:
    model, meta = load_model(model_path)
    feats = meta["features"]
    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        df = load_features(con, date, feats)
    finally:
        con.close()
    X = df[feats]
    score = m.predict_score(model, X)
    prob = calibration.apply(meta.get("calibration"), score)
    alerts = df[["store_id", "product_id"]].copy()
    alerts["score"] = score
    alerts["calibrated_prob"] = prob
    order = np.lexsort((alerts.product_id.values, alerts.store_id.values, -alerts.score.values))
    alerts = alerts.iloc[order].head(top_k).reset_index(drop=True)
    alerts.insert(4, "rank", np.arange(1, len(alerts) + 1))
    X_top = X.iloc[order[:top_k]].reset_index(drop=True)
    contrib, _ = contributions(model, X_top)
    tops = top_contributors(contrib, X_top)
    for k in range(3):
        alerts[f"top_contributor_{k + 1}"] = [t[k] if len(t) > k else "" for t in tops]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"alerts_{date}.csv"
    alerts.to_csv(path, index=False, float_format="%.6f")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the ranked next-day stockout alert list for one day")
    ap.add_argument("--date", required=True, help="prediction day D (YYYY-MM-DD); alerts cover D+1")
    ap.add_argument("--top-k", type=int, default=5000, help="number of alerts to write")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--warehouse", type=Path, default=WAREHOUSE)
    ap.add_argument("--out-dir", type=Path, default=ALERTS_DIR)
    args = ap.parse_args(argv)
    path = score_date(args.date, args.top_k, args.model, args.warehouse, args.out_dir)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
