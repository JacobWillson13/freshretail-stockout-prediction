"""Synthetic FreshRetailNet-shaped data and an in-memory warehouse built from it."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import connect, register_raw_frames  # noqa: E402
from src.run_sql import run_models  # noqa: E402

START = pd.Timestamp("2024-03-28")
TRAIN_DAYS = 35
EVAL_DAYS = 7


def make_rows(seed: int = 0, n_stores: int = 2, n_products: int = 4,
              n_days: int = TRAIN_DAYS + EVAL_DAYS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for store in range(n_stores):
        for product in range(n_products):
            for d in range(n_days):
                dt = START + pd.Timedelta(days=d)
                status = (rng.random(24) < rng.uniform(0.0, 0.5)).astype(int)
                sale = np.where(status == 0, rng.gamma(1.0, 0.1, 24), 0.0).round(3)
                rows.append({
                    "city_id": 0, "store_id": store, "management_group_id": 0,
                    "first_category_id": product % 2, "second_category_id": product,
                    "third_category_id": product, "product_id": product,
                    "dt": dt.strftime("%Y-%m-%d"),
                    "sale_amount": float(sale.sum()),
                    "hours_sale": sale.tolist(),
                    "stock_hour6_22_cnt": int(status[6:22].sum()),
                    "hours_stock_status": status.tolist(),
                    "discount": float(rng.choice([1.0, 0.8, 0.9])),
                    "holiday_flag": int(dt.dayofweek >= 5),
                    "activity_flag": int(rng.random() < 0.2),
                    "precpt": float(rng.gamma(1.0, 2.0)),
                    "avg_temperature": float(rng.normal(25, 3)),
                    "avg_humidity": float(rng.uniform(50, 90)),
                    "avg_wind_level": float(rng.uniform(0, 3)),
                })
    return pd.DataFrame(rows)


def split_frames(df: pd.DataFrame):
    cut = (START + pd.Timedelta(days=TRAIN_DAYS - 1)).strftime("%Y-%m-%d")
    return df[df.dt <= cut].reset_index(drop=True), df[df.dt > cut].reset_index(drop=True)


def build(df: pd.DataFrame):
    con = connect(None)
    train, eval_ = split_frames(df)
    register_raw_frames(con, train, eval_)
    run_models(con, verbose=False)
    return con


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_rows()


@pytest.fixture(scope="session")
def warehouse(raw_df):
    return build(raw_df)
