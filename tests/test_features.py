"""Added features: intraday timing, long memory, entity context and censored demand."""
import numpy as np
import pandas as pd
import pytest

from src.modeling import ABLATIONS, CONFIGS, NEW_GROUPS
from tests.conftest import build, make_rows


def _status(oos_hours: set[int]) -> list[int]:
    return [1 if h in oos_hours else 0 for h in range(24)]


def _sales(by_hour: dict[int, float]) -> list[float]:
    return [float(by_hour.get(h, 0.0)) for h in range(24)]


# day index -> (out-of-stock hours, hourly sales)
DAYS = {
    # OOS overnight 0..5, in the window at 10, 11, 19, 21, and at 22; sales at 7 and 13
    0: ({0, 1, 2, 3, 4, 5, 10, 11, 19, 21, 22}, {7: 1.0, 13: 2.0}),
    # fully in stock, one sale at 8
    1: (set(), {8: 1.0}),
    # out of stock all 24 hours, no sale
    2: (set(range(24)), {}),
    # OOS 6..8 with a sale recorded at 7 during the OOS hours, then in stock with a sale at 10;
    # the last OOS window hour (8) comes after the first sale (7)
    3: ({6, 7, 8}, {7: 5.0, 10: 1.0}),
    # OOS 6..8 before any sale, in stock from 9 with a sale at 10, OOS at 23 only
    4: ({6, 7, 8, 23}, {10: 1.0}),
}


@pytest.fixture(scope="module")
def con():
    df = make_rows(seed=7, n_stores=1, n_products=1, n_days=20)
    for idx, (oos, sales) in DAYS.items():
        st = _status(oos)
        df.at[idx, "hours_stock_status"] = st
        df.at[idx, "hours_sale"] = _sales(sales)
        df.at[idx, "sale_amount"] = float(sum(sales.values()))
        df.at[idx, "stock_hour6_22_cnt"] = int(sum(st[6:22]))
    return build(df)


def _intraday(con) -> pd.DataFrame:
    return con.execute("SELECT * FROM features_intraday ORDER BY dt").df()


def test_intraday_hand_built_day(con):
    r = _intraday(con).iloc[0]
    assert r.oos_at_close == 1
    assert r.oos_hours_last3 == 2            # 19 and 21, not 20
    assert r.oos_hours_late_night == 1       # 22 only
    assert r.first_oos_hour == 10            # overnight hours 0..5 are outside the window
    assert r.last_in_stock_hour == 20
    assert r.oos_after_sale == 1             # OOS at 10 after the sale at 7
    assert r.ended_day_oos == 0


def test_intraday_no_stockout_and_full_stockout(con):
    f = _intraday(con)
    in_stock, out = f.iloc[1], f.iloc[2]
    assert in_stock.oos_at_close == 0 and in_stock.oos_hours_last3 == 0
    assert pd.isna(in_stock.first_oos_hour)
    assert in_stock.last_in_stock_hour == 21
    assert in_stock.oos_after_sale == 0 and in_stock.ended_day_oos == 0
    assert out.oos_at_close == 1 and out.oos_hours_last3 == 3 and out.oos_hours_late_night == 2
    assert out.first_oos_hour == 6
    assert pd.isna(out.last_in_stock_hour)
    assert out.oos_after_sale == 0           # no sale at all
    assert out.ended_day_oos == 1


def test_oos_after_sale_needs_oos_after_the_first_sale(con):
    f = _intraday(con)
    assert f.iloc[3].oos_after_sale == 1     # sale at 7, still OOS at 8
    assert f.iloc[4].oos_after_sale == 0     # OOS 6..8, first sale at 10
    assert f.iloc[4].ended_day_oos == 1 and f.iloc[4].oos_hours_late_night == 1


def test_oos_at_close_lag_and_rolling_mean(con):
    f = _intraday(con)
    assert pd.isna(f.oos_at_close_lag1.iloc[0])
    np.testing.assert_array_equal(f.oos_at_close_lag1.iloc[1:].values, f.oos_at_close.iloc[:-1].values)
    expected = f.oos_at_close.rolling(7, min_periods=1).mean()
    np.testing.assert_allclose(f.oos_at_close_mean_7d.values, expected.values)


def test_demand_estimates(con):
    d = con.execute("SELECT * FROM features_demand ORDER BY dt").df()
    # day 0: 12 available window hours, in-stock sales 3.0 -> 0.25/h -> demand 4.0, lost 1.0
    assert d.demand_est[0] == pytest.approx(4.0)
    assert d.lost_sales_est[0] == pytest.approx(1.0)
    # day 1: in stock all day, demand = observed sales (1.0), nothing lost
    assert d.demand_est[1] == pytest.approx(1.0)
    assert d.lost_sales_est[1] == pytest.approx(0.0)
    # day 2: no available hour -> NULL demand and NULL lost sales
    assert np.isnan(d.demand_est[2]) and np.isnan(d.lost_sales_est[2])
    # day 3: a sale recorded during an OOS hour makes window sales exceed the estimate: floor 0
    assert d.demand_est[3] == pytest.approx(16 / 13)
    assert d.lost_sales_est[3] == 0.0
    # 7-day means skip NULL days
    assert d.demand_est_mean_7d[2] == pytest.approx((4.0 + 1.0) / 2)


def test_history_is_expanding(warehouse):
    h = warehouse.execute(
        "SELECT h.*, d.oos_hours FROM features_history h JOIN int_daily d USING (store_id, product_id, dt) "
        "ORDER BY store_id, product_id, dt"
    ).df()
    g = h.groupby(["store_id", "product_id"])
    assert "hist_days_observed" not in h.columns   # dropped: it acts as a date index
    np.testing.assert_allclose(h.hist_oos_hours_mean.values,
                               g.oos_hours.transform(lambda s: s.expanding().mean()).values)
    share = (h.oos_hours > 2).astype(float).groupby([h.store_id, h.product_id]).transform(
        lambda s: s.expanding().mean())
    np.testing.assert_allclose(h.hist_oos_day_share.values, share.values)


@pytest.fixture(scope="module")
def entity_con():
    # 3 stores x 4 products, second category = product % 2, so every store x second category
    # group holds 2 products and every product is sold in 3 stores
    df = make_rows(seed=13, n_stores=3, n_products=4)
    df["second_category_id"] = df.product_id % 2
    return build(df)


def _loo(frame: pd.DataFrame, keys: list[str]) -> np.ndarray:
    g = frame.groupby(keys).oos_day
    n = g.transform("size")
    return ((g.transform("sum") - frame.oos_day) / (n - 1)).where(n > 1).values


def _entity_vs_pandas(con) -> pd.DataFrame:
    d = con.execute(
        "SELECT store_id, product_id, second_category_id, dt, oos_hours FROM int_daily").df()
    e = con.execute("SELECT * FROM features_entity").df()
    d["oos_day"] = (d.oos_hours > 2).astype(float)
    return e.merge(d, on=["store_id", "product_id", "dt"], suffixes=("", "_d")).sort_values(
        ["store_id", "product_id", "dt"]).reset_index(drop=True)


def test_entity_rates_are_leave_one_out(entity_con):
    m = _entity_vs_pandas(entity_con)
    np.testing.assert_allclose(m.prod_oos_rate.values, _loo(m, ["product_id", "dt"]))
    np.testing.assert_allclose(m.store_cat2_oos_rate.values,
                               _loo(m, ["store_id", "second_category_id", "dt"]))
    # the store rate keeps every product of the store
    np.testing.assert_allclose(m.store_oos_rate.values,
                               m.groupby(["store_id", "dt"]).oos_day.transform("mean").values)
    # 7-day means average the row's own leave-one-out daily rates (NULL days skipped)
    series = m.groupby(["store_id", "product_id"])
    for col in ("prod_oos_rate", "store_cat2_oos_rate"):
        roll = series[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        np.testing.assert_allclose(m[f"{col}_mean_7d"].values, roll.values)
    s = m.drop_duplicates(["store_id", "dt"])
    roll = s.groupby("store_id").store_oos_rate.transform(lambda x: x.rolling(7, min_periods=1).mean())
    np.testing.assert_allclose(s.store_oos_rate_mean_7d.values, roll.values)


def test_leave_one_out_is_null_for_single_member_groups(warehouse):
    # in the default synthetic data each store x second category holds one product
    r = warehouse.execute(
        "SELECT COUNT(*), COUNT(store_cat2_oos_rate), COUNT(store_cat2_oos_rate_mean_7d), "
        "COUNT(prod_oos_rate) FROM features_entity").fetchone()
    assert r[1] == 0 and r[2] == 0
    assert r[3] == r[0]                      # 2 stores per product: always one other store


def test_features_all_has_every_model_feature_and_unlabeled_last_day(warehouse):
    cols = set(warehouse.execute("SELECT * FROM features_all LIMIT 0").df().columns)
    assert set(CONFIGS["full"]) <= cols
    last = warehouse.execute(
        "SELECT (SELECT COUNT(*) FROM features_all WHERE dt = (SELECT MAX(dt) FROM stg_sales)), "
        "(SELECT COUNT(*) FROM model_dataset WHERE dt = (SELECT MAX(dt) FROM stg_sales))"
    ).fetchone()
    assert last[0] > 0 and last[1] == 0


def test_config_groups():
    full = set(CONFIGS["full"])
    assert set(CONFIGS["combined"]) < full
    for g, feats in NEW_GROUPS.items():
        assert set(feats) <= full
        assert set(ABLATIONS[f"full_minus_{g}"]) == full - set(feats)
