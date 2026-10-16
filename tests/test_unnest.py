"""The hourly unnesting model keeps every array element at the right hour."""
import numpy as np


def test_one_row_per_hour(warehouse, raw_df):
    n = warehouse.execute("SELECT COUNT(*) FROM int_hourly").fetchone()[0]
    assert n == 24 * len(raw_df)
    hours = warehouse.execute("SELECT MIN(hour), MAX(hour), COUNT(DISTINCT hour) FROM int_hourly").fetchone()
    assert hours == (0, 23, 24)


def test_values_match_array_positions(warehouse, raw_df):
    row = raw_df.iloc[17]
    got = warehouse.execute(
        "SELECT hour, stock_status, sale FROM int_hourly "
        "WHERE store_id = ? AND product_id = ? AND dt = CAST(? AS DATE) ORDER BY hour",
        [int(row["store_id"]), int(row["product_id"]), row["dt"]],
    ).df()
    assert got.stock_status.tolist() == list(row.hours_stock_status)
    np.testing.assert_allclose(got.sale.values, np.array(row.hours_sale))


def test_window_flags_and_count(warehouse):
    # in_window covers hours 6..21 and reproduces stock_hour6_22_cnt on every row
    bad = warehouse.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT h.store_id, h.product_id, h.dt,
                 SUM(CASE WHEN h.in_window THEN h.stock_status ELSE 0 END) AS n1,
                 ANY_VALUE(s.stock_hour6_22_cnt) AS cnt
          FROM int_hourly h JOIN stg_sales s USING (store_id, product_id, dt)
          GROUP BY 1, 2, 3
        ) WHERE n1 <> cnt
        """
    ).fetchone()[0]
    assert bad == 0
    window = warehouse.execute(
        "SELECT MIN(hour), MAX(hour) FROM int_hourly WHERE in_window"
    ).fetchone()
    assert window == (6, 21)


def test_daily_stock_metrics(warehouse):
    from tests.conftest import build, make_rows

    df = make_rows(seed=1, n_stores=1, n_products=1, n_days=20)
    status = [1] * 6 + [0, 1, 1, 1, 0, 1, 1, 0] + [0] * 8 + [1, 1]   # window: 6..21
    df.at[5, "hours_stock_status"] = status
    df.at[5, "hours_sale"] = [0.0 if s else 0.5 for s in status]
    df.at[6, "hours_stock_status"] = [1] * 24
    df.at[6, "hours_sale"] = [0.0] * 24
    con = build(df)
    r = con.execute(
        "SELECT oos_hours, longest_oos_run, oos_hours_overnight, available_hours, "
        "in_stock_sales, sales_per_available_hour FROM int_daily_stock ORDER BY dt"
    ).df()
    # window hours 6..21 = [0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0]
    assert r.oos_hours[5] == 5
    assert r.longest_oos_run[5] == 3
    assert r.oos_hours_overnight[5] == 6 + 2
    assert r.available_hours[5] == 11
    assert abs(r.sales_per_available_hour[5] - 0.5) < 1e-9
    # fully out of stock day: no available hours, so sales per available hour is NULL
    assert r.oos_hours[6] == 16 and r.available_hours[6] == 0
    assert np.isnan(r.sales_per_available_hour[6])
