"""EDA queries over the warehouse. Each returns a DataFrame with sample sizes (n).

Relationships here are associations, not causes.
"""
from __future__ import annotations

import pandas as pd

OOS_DAY = "CASE WHEN oos_hours > 2 THEN 1.0 ELSE 0.0 END"


def oos_hours_distribution(con) -> pd.DataFrame:
    return con.execute(
        "SELECT oos_hours, COUNT(*) AS n FROM int_daily GROUP BY 1 ORDER BY 1"
    ).df().assign(share=lambda d: d.n / d.n.sum())


def stockout_by_hour(con) -> pd.DataFrame:
    return con.execute(
        "SELECT hour, AVG(stock_status) AS oos_rate, COUNT(*) AS n "
        "FROM int_hourly GROUP BY 1 ORDER BY 1"
    ).df()


def stockout_by_hour_category(con) -> pd.DataFrame:
    return con.execute(
        "SELECT first_category_id, hour, AVG(stock_status) AS oos_rate, COUNT(*) AS n "
        "FROM int_hourly GROUP BY 1, 2 ORDER BY 1, 2"
    ).df()


def rate_by(con, expr: str, label: str) -> pd.DataFrame:
    """Share of store-product-days with > 2 window out-of-stock hours, by a grouping expression."""
    return con.execute(
        f"""
        SELECT {expr} AS {label}, COUNT(*) AS n, AVG({OOS_DAY}) AS stockout_day_rate,
               AVG(oos_hours) AS mean_oos_hours
        FROM int_daily GROUP BY 1 ORDER BY 1
        """
    ).df()


DISCOUNT_BIN = """CASE WHEN discount >= 1.0 THEN 'a 1.00 (none)'
     WHEN discount >= 0.9 THEN 'b 0.90-0.99'
     WHEN discount >= 0.8 THEN 'c 0.80-0.89'
     WHEN discount >= 0.7 THEN 'd 0.70-0.79'
     WHEN discount >= 0.5 THEN 'e 0.50-0.69'
     ELSE 'f <0.50' END"""
TEMP_BIN = "CAST(FLOOR(avg_temperature / 5) * 5 AS INTEGER)"
PRECIP_BIN = """CASE WHEN precpt = 0 THEN '0 none'
     WHEN precpt < 2 THEN '1 light (<2)'
     WHEN precpt < 10 THEN '2 moderate (2-10)'
     ELSE '3 heavy (>=10)' END"""


def persistence(con) -> pd.DataFrame:
    """Next-day stockout rate given today's state (labeled rows)."""
    return con.execute(
        """
        SELECT CASE WHEN oos_hours > 2 THEN 'today > 2h' ELSE 'today <= 2h' END AS today,
               COUNT(*) AS n, AVG(label) AS p_tomorrow_stockout
        FROM model_dataset GROUP BY 1 ORDER BY 1
        """
    ).df()


def persistence_by_hours(con) -> pd.DataFrame:
    return con.execute(
        "SELECT oos_hours, COUNT(*) AS n, AVG(label) AS p_tomorrow_stockout "
        "FROM model_dataset GROUP BY 1 ORDER BY 1"
    ).df()


def example_series(con, n: int = 3) -> pd.DataFrame:
    """Three series with a mid-range stockout rate (deterministic pick)."""
    keys = con.execute(
        f"""
        WITH s AS (
          SELECT store_id, product_id, AVG({OOS_DAY}) AS r, SUM(sale_amount) AS sales
          FROM int_daily GROUP BY 1, 2
        )
        SELECT store_id, product_id FROM s
        WHERE r BETWEEN 0.3 AND 0.6 AND sales > 50
        ORDER BY hash(store_id * 100000 + product_id) LIMIT {n}
        """
    ).df()
    frames = []
    for _, k in keys.iterrows():
        frames.append(con.execute(
            "SELECT store_id, product_id, dt, source, oos_hours, oos_hours_overnight, "
            "sale_amount, discount FROM int_daily WHERE store_id = ? AND product_id = ? ORDER BY dt",
            [int(k.store_id), int(k.product_id)],
        ).df())
    return pd.concat(frames, ignore_index=True)
